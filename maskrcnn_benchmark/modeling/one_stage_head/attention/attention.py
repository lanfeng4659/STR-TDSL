import torch
from torch import nn
from torch.nn import functional as F
import cv2
import os
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import build_fcos
from maskrcnn_benchmark.utils.text_util import TextGenerator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, cat_boxlist_texts
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torch.autograd import Variable
import string
import random
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from .box_aug import make_box_aug
from .transformer import TextTransformer
from .position_encoding import build_position_encoding
from .selector import build_selector
INF = 10000000000

class SelfAttention(nn.Module):
    def __init__(self, in_channels, embed_dim,steps=10):
        super(SelfAttention, self).__init__()
        convs = []
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        # convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.steps = steps
        self.in_channels = in_channels
        self.q_proj = nn.Linear(in_channels*steps, embed_dim, bias=False)
        self.k_proj = nn.Linear(in_channels//2*steps, embed_dim, bias=False)
        self.v_proj = nn.Linear(in_channels//2*steps, embed_dim, bias=False)

        self.k_conv = conv_func(in_channels, in_channels//2, 3, stride=(2, 1))
        self.v_conv = conv_func(in_channels, in_channels//2, 3, stride=(2, 1))
        self.self_attn = nn.MultiheadAttention(1024, 1)
    def forward(self,q,k,v):
        assert q.size(2)==k.size(1)==v.size(1) 
        k_ = self.k_proj(self.k_conv(k).mean(dim=2).view(-1,self.steps*self.in_channels//2))[:,None,:]
        v_ = self.v_proj(self.v_conv(v).mean(dim=2).view(-1,self.steps*self.in_channels//2))[:,None,:]
        q_ = self.q_proj(q.view(-1,self.in_channels*self.steps))[:,None,:]
        attn_output, attn_output_weights = self.self_attn(q_, k_, v_)
        # print(q.shape,v.shape,k.shape,attn_output_weights.shape)
        return attn_output[:,0,:]

class RNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels,bidirectional=True):
        super(RNNDecoder, self).__init__()
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels, 256, out_channels)
    def forward(self, x, targets=None):
        x = self.convs(x)
        x = x.mean(dim=2)  # NxCxW
        # assert x.size(-2) == 1, "the height of conv must be 1"
        # x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # WxNxC
        x = self.rnn(x)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        return x

class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
class WordEmbedding(nn.Module):
    def __init__(self,
                      out_channels=512,
                      embedding_dim=300,
                      char_vector_dim=256,
                      out_embedding_dim=1024,
                      max_length=10,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True):
        super(WordEmbedding, self).__init__()
        self.max_length = int(max_length)
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.char_embedding = nn.Embedding(len(self.lexicon), embedding_dim)
        self.char_encoder = nn.Sequential(
            nn.Linear(embedding_dim, char_vector_dim),
            nn.ReLU(inplace=True)
        )
        # self.rnn = nn.LSTM(char_vector_dim, out_channels,num_layers=1,bidirectional=bidirectional)
        self.rnn = BidirectionalLSTM(char_vector_dim, 256, out_channels)
        self.proj = nn.Linear(out_channels*max_length, out_embedding_dim, bias=False)
    def forward(self,inputs):
        '''
        word: b, 256
        embedding: b, 256, 300
        h_t: b, out_channels
        '''
        # print(inputs)
        embeddings_batch = []
        for word in inputs:
            assert len(word)>0,word
            embeddings = self.char_embedding(word)
            embeddings_batch.append(
                nn.functional.interpolate(
                    embeddings[None,None,...], 
                    size=(self.max_length,self.embedding_dim), 
                    mode='bilinear', 
                    align_corners=True)
            )
        embeddings_batch = torch.cat(embeddings_batch,dim=1)[0] # [b, self.max_length, embedding_dim]
        char_vector_ori = self.char_encoder(embeddings_batch)
        char_vector = char_vector_ori.permute(1, 0, 2).contiguous() # [w, b, c]
        # print(char_vector.shape)
        x = self.rnn(char_vector)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        out = self.proj(x.view(x.size(0),-1))
        return out,char_vector_ori

class AttentionHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AttentionHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        resolution = cfg.MODEL.ALIGN.POOLER_RESOLUTION
        canonical_scale = cfg.MODEL.ALIGN.POOLER_CANONICAL_SCALE
        self.use_word_aug = cfg.MODEL.ALIGN.USE_WORD_AUG
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.pooler = Pooler(
            output_size=resolution,
            scales=self.scales,
            sampling_ratio=1,
            canonical_scale=canonical_scale,
            mode='align')
        
        out_channels = 128
        in_channels = 256
        self.text_generator = TextGenerator()
        self.box_augumentor = make_box_aug()

        self.selector = build_selector(cfg)
        self.atten = SelfAttention(256,1024)

        self.position_encoding = build_position_encoding(256)
        # self.text_transformer = TextTransformer(d_model=256)
        self.word_embedding = WordEmbedding(out_channels=out_channels,
                      embedding_dim=256,
                      char_vector_dim=256,
                      max_length=resolution[1],
                      lexicon = self.text_generator.chars,
                      bidirectional=True)
        self.feat_dim = 1920
        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
    @torch.no_grad()
    def get_word_embedding(self,texts,device):
        words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(device) for text in texts]
        words_embedding = self.word_embedding(words).detach()
        return words_embedding
    def compute_loss(self, embedding1, embedding2, words1,words2):
        k = 1
        iou = self.compute_similarity(embedding1, embedding2, k)
        # print(iou.shape, len(words1),len(words2))
        similarity = self.text_generator.calculate_similarity_matric(words1, words2)
        
        loss = self.sim_loss_func(iou, torch.tensor(similarity).type_as(iou))
        loss = loss.max(dim=1)[0].mean()
        # print(loss)
        return loss
    def compute_similarity(self,embedding1, embedding2,k=1):
        embedding1_nor = nn.functional.normalize((embedding1*k).tanh().view(embedding1.size(0),-1))
        embedding2_nor = nn.functional.normalize((embedding2*k).tanh().view(embedding2.size(0),-1))
        inter = embedding1_nor.mm(embedding2_nor.t())
        return inter

    def forward(self, x,maps, samples,images=None,is_words=None):
        """
        offset related operations are messy
        images: used for test pooler
        """        
        if self.training:
            targets = samples["retrieval_samples"]
            texts = []
            new_proposals = []
            # with torch.no_grad():
            #     boxes = self.selector(maps["locations"],maps["box_cls"],maps["box_regression"],maps["centerness"],maps["image_sizes"])
            boxes = targets
            x = [torch.cat([v,self.position_encoding(v)],dim=1) for v in x]
            c = x[0].size(1)//2
            all_img_embeddings = []
            all_word_embeddings = []
            for box,target in zip(boxes, targets):
                rois_with_position = self.pooler(x, [self.box_augumentor(target)])
                assert rois_with_position.size(0) > 0
                rois = rois_with_position[:,:c,:,:]
                positions = rois_with_position[:,c:,:,:]
                text_per_img = target.get_field("texts").tolist()
                assert len(text_per_img) > 0
                texts.extend(text_per_img)
                words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(rois.device) for text in text_per_img]
                words_embedding,before_rnn = self.word_embedding(words)
                queries = before_rnn.detach() # T, B, C
                keys = rois+positions
                values = rois
                img_embedding_per_img = self.atten(queries,keys,values)
                all_img_embeddings.append(img_embedding_per_img)
                all_word_embeddings.append(words_embedding)
            all_img_embeddings = torch.cat(all_img_embeddings,dim=0)
            
            
            # print(boxes)
            
            if all_img_embeddings.size(0)==0:
                zero_loss = x[0].sum()*0
                loss = {"loss_wi":zero_loss,"loss_ww":zero_loss,"loss_ii":zero_loss}
                return None,loss
            # print(new_proposals)
            # self.test_pooler(images, new_proposals)
            # imgs_embedding = self.image_embedding(rois)
            # assert imgs_embedding.size(0) == len(texts),print(imgs_embedding.size(0),len(texts))
            word_texts = texts.copy()
            aug_texts = []
            if self.use_word_aug:
                aug_texts.extend([self.text_generator(text) for text in texts])
                words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(all_img_embeddings.device) for text in aug_texts]
                word_texts.extend(aug_texts)
                words_embedding,before_rnn = self.word_embedding(words)
                all_word_embeddings.append(words_embedding)
            all_word_embeddings = torch.cat(all_word_embeddings,dim=0)
            
            # print(words_embedding.shape,before_rnn.shape)
            # imgs_embedding = 

            loss_fn = self.compute_loss
            wi_loss = loss_fn(all_word_embeddings.detach(), all_img_embeddings, word_texts, texts)
            ww_loss = loss_fn(all_word_embeddings, all_word_embeddings, word_texts, word_texts)
            ii_loss = loss_fn(all_img_embeddings, all_img_embeddings, texts, texts)
            
            loss = {"loss_wi":wi_loss*1,"loss_ww":ww_loss*10,"loss_ii":ii_loss*1}
            return None,loss
        else:
            select_boxes = []
            proposals = samples["retrieval_samples"]
            x = [torch.cat([v,self.position_encoding(v)],dim=1) for v in x]
            c = x[0].size(1)//2
            all_img_embeddings = []
            all_word_embeddings = []
            for proposals_per_im in proposals:
                rois_with_position = self.pooler(x, [proposals_per_im])
                assert rois_with_position.size(0) > 0
                rois = rois_with_position[:,:c,:,:]
                positions = rois_with_position[:,c:,:,:]
                text_per_img = proposals_per_im.get_field("texts").tolist()
                assert len(text_per_img) > 0
                words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(rois.device) for text in text_per_img]
                words_embedding,before_rnn = self.word_embedding(words)
                queries = before_rnn.detach() # T, B, C
                keys = rois+positions
                values = rois
                img_embedding_per_img = self.atten(queries,keys,values)
                imgs_embedding_nor = nn.functional.normalize((img_embedding_per_img*1).tanh().view(img_embedding_per_img.size(0),-1))
                words_embedding_nor = nn.functional.normalize((words_embedding*1).tanh().view(words_embedding.size(0),-1))
                proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
                if is_words:
                    proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)

            return proposals, {"select_boxes":select_boxes}
class AttentionModule(torch.nn.Module):
    """
    Module for BezierAlign computation. Takes feature maps from the backbone and
    BezierAlign outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels,proposal_matcher):
        super(AttentionModule, self).__init__()

        self.cfg = cfg.clone()
        
        self.detector = build_fcos(cfg, in_channels)
        self.proposal_matcher = proposal_matcher
        self.scales = cfg.MODEL.ATTENTION.POOLER_SCALES
        self.use_box_aug = cfg.MODEL.ATTENTION.USE_BOX_AUG
        self.use_retrieval = cfg.MODEL.ATTENTION.USE_RETRIEVAL
        if self.use_retrieval:
            self.head = AttentionHead(cfg, in_channels)
        self.batch_size_per_image = 256 
        self.positive_fraction = 0.25


    def forward(self, images, features, targets=None, vis=False,is_words=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
            vis (bool): visualise offsets

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        
        # print(targets)
        targets = [target.to(features[0].device) for target in targets]
        if self.training:
            # self.visual(images, boxes)
            rec_features = features[:len(self.scales)]
            maps, losses = self.detector(images, features[1:], targets)
            if not self.use_retrieval:
                return None, losses
            new_boxes = targets
            _, loss_dict = self.head(rec_features,maps, {"retrieval_samples":new_boxes},images)

            # self.visual(images, boxes)
            
            
            for k, v in loss_dict.items():
                losses.update({k: v})
            # print(losses)
            return None, losses

        else:
            boxes, losses = self.detector(images, features[1:], targets)
            if not self.use_retrieval:
                return boxes, losses
            # print(images)
            # new_boxes = self.select_positive_boxes(boxes, targets)
            
            rec_features = features[:len(self.scales)]

            new_boxes = []
            for box, target in zip(boxes, targets):
                scores = box.get_field("scores")
                # pos_idxs = torch.nonzero(scores>0.08).view(-1)
                pos_idxs = torch.nonzero(scores>0.05).view(-1)
                # pos_idxs = torch.nonzero(scores>0.2).view(-1)#75.43
                box = box[pos_idxs]
                box.add_field("texts", target.get_field("texts"))
                box.add_field("scale", target.get_field("scale"))
                box.add_field("path", target.get_field("path"))
                # box.add_field("y_trues", target.get_field("y_trues"))
                new_boxes.append(box)
                # new_boxes.append(box)
            image_names = [os.path.basename(str(image.get_field("path"))) for image in new_boxes]
            # self.test_visual(images, new_boxes[0].bbox,new_boxes[0].bbox,image_names[0])
            # pos_idxs = torch.nonzero(scores>thresholds).view(-1)
            # results, other = self.head(rec_features,{"retrieval_samples":new_boxes},images)
            results, other = self.head(rec_features,None,{"retrieval_samples":new_boxes},images,is_words=is_words)
            # results, other = self.head(rec_features,{"retrieval_samples":targets},images,is_words=is_words)
            # self.test_visual(images, new_boxes[0].bbox,other["select_boxes"][0],image_names[0])
            # results, _ = self.head(rec_features,targets,images)
            return results, other

        # preds, _ = self.head(rec_features, boxes)
        


@registry.ONE_STAGE_HEADS.register("attention")
def build_attention_head(cfg, in_channels):
    return AttentionModule(cfg, in_channels, 
                        Matcher(
                        0.7,
                        0.5,
                        allow_low_quality_matches=False)
                    )

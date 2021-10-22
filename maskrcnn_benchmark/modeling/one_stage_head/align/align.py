import os
import torch
from torch import nn
from torch.nn import functional as F
import cv2
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
INF = 10000000000


class CTCPredictor(nn.Module):
    def __init__(self, in_channels, class_num):
        super(CTCPredictor, self).__init__()
        self.class_num = class_num
        self.clf = nn.Linear(in_channels, self.class_num)

    def forward(self, x, targets=None):
        x = self.clf(x)
        if self.training:
            x = F.log_softmax(x, dim=-1).permute(1,0,2)
            # print(targets.shape)
            input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
            target_lengths, targets_sum = self.prepare_targets(targets)
            # print(x.shape,targets.shape,target_lengths.shape)
            # loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True) / 10
            loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True)
            # loss = F.ctc_loss(x, targets_sum, input_lengths, target_lengths, blank=self.class_num-1, zero_infinity=True)/2
            return loss
        return x
    def prepare_targets(self, targets):
        target_lengths = (targets != self.class_num - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        sum_targets = torch.cat(sum_targets)
        return target_lengths, sum_targets
class RNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels,bidirectional=True,use_look_up=False,use_res_link=False,use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(RNNDecoder, self).__init__()
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        convs = []
        self.use_rnn = use_rnn
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels, 256, out_channels) if self.use_rnn else nn.Linear(in_channels,out_channels)
    def forward(self, x, dictionary=None):
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
                      max_length=10,
                      lexicon = string.ascii_lowercase+string.digits,
                      bidirectional=True,
                      use_res_link=False,
                      use_rnn = True,use_pyramid=False,pyramid_layers=None):
        super(WordEmbedding, self).__init__()
        self.use_rnn = use_rnn
        self.max_length = int(max_length)
        self.lexicon = lexicon
        self.embedding_dim=embedding_dim
        self.char_embedding = nn.Embedding(len(self.lexicon), embedding_dim)
        self.char_encoder = nn.Sequential(
            nn.Linear(embedding_dim, char_vector_dim),
            nn.ReLU(inplace=True)
        )
        # self.rnn = nn.LSTM(char_vector_dim, out_channels,num_layers=1,bidirectional=bidirectional)
        self.rnn = BidirectionalLSTM(char_vector_dim, 256, out_channels) if self.use_rnn else nn.Linear(char_vector_dim,out_channels)
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
        char_vector = self.char_encoder(embeddings_batch)
        char_vector = char_vector.permute(1, 0, 2).contiguous() # [w, b, c]
        x = self.rnn(char_vector)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        return x

class AlignHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AlignHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        resolution = cfg.MODEL.ALIGN.POOLER_RESOLUTION
        canonical_scale = cfg.MODEL.ALIGN.POOLER_CANONICAL_SCALE
        # print(resolution)
        self.use_ctc_loss = cfg.MODEL.ALIGN.USE_CTC_LOSS
        self.is_chinese = cfg.MODEL.ALIGN.IS_CHINESE
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.pooler = Pooler(
            output_size=resolution,
            scales=self.scales,
            sampling_ratio=1,
            canonical_scale=canonical_scale,
            mode='align')
        
        out_channels = 128
        in_channels = 256
        if self.is_chinese:
            # lexicon = np.load("./datasets/rctw/chars.npy").tolist()
            lexicon = np.load("/workspace/wanghao/projects/Pytorch-yolo-phoc/selected_chars.npy").tolist()
            print(len(lexicon))
            self.text_generator = TextGenerator(ratios=[1,0,1,5],chars=lexicon)
        else:
            self.text_generator = TextGenerator()
        self.image_embedding = RNNDecoder(in_channels, out_channels,bidirectional=True)
        self.word_embedding = WordEmbedding(out_channels=out_channels,
                      embedding_dim=256,
                      char_vector_dim=256,
                      max_length=resolution[1],
                      lexicon = self.text_generator.chars,
                      bidirectional=True)
        frames = resolution[1]
        if self.use_ctc_loss:
            self.ctc_head = CTCPredictor(out_channels,len(self.text_generator.chars)+1)

        self.feat_dim = 128*frames 

        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
    @torch.no_grad()
    def get_word_embedding(self,texts,device):
        words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(device) for text in texts]
        words_embedding = self.word_embedding(words).detach()
        return words_embedding

    def compute_similarity(self,embedding1, embedding2,k=1):
        embedding1_nor = nn.functional.normalize((embedding1*k).tanh().view(embedding1.size(0),-1))
        embedding2_nor = nn.functional.normalize((embedding2*k).tanh().view(embedding2.size(0),-1))
        inter = embedding1_nor.mm(embedding2_nor.t())
        return inter



    def forward(self, x, samples,images=None,is_words=None):
        """
        offset related operations are messy
        images: used for test pooler
        """        
        if self.training:
            return None,None
        else:
            select_boxes = []
            proposals = samples["retrieval_samples"]
            for proposals_per_im in proposals:
                if not self.is_chinese:
                    idxs, texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                else:
                    texts = proposals_per_im.get_field("texts").tolist()
                if is_words:
                    if len(texts) == 0:
                        words_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
                    else:
                        if not self.is_chinese:
                            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x[0].device) for text in texts]
                        else:
                            words = [torch.tensor(self.text_generator.label_map(text)).long().to(x[0].device) for text in texts]
                            # print(len(words))
                        words_embedding = self.word_embedding(words)

                        k = 1
                        words_embedding_nor = nn.functional.normalize((words_embedding*k).tanh().view(words_embedding.size(0),-1))
                        # words_embedding_nor = nn.functional.normalize(words_embedding.view(words_embedding.size(0),-1))
                if proposals_per_im.bbox.size(0) == 0:
                    imgs_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
                else:

                    rois = self.pooler(x, [proposals_per_im])
                    select_boxes.append(proposals_per_im.bbox)
                    imgs_embedding = self.image_embedding(rois)

                    k = 1

                    imgs_embedding_nor = nn.functional.normalize((imgs_embedding*k).tanh().view(imgs_embedding.size(0),-1))
                    # imgs_embedding_nor = nn.functional.normalize(imgs_embedding.view(imgs_embedding.size(0),-1))
                proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
                if is_words:
                    proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)

            return proposals, {"select_boxes":select_boxes}

class AlignModule(torch.nn.Module):
    """
    Module for BezierAlign computation. Takes feature maps from the backbone and
    BezierAlign outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels,proposal_matcher):
        super(AlignModule, self).__init__()

        self.cfg = cfg.clone()
        
        self.detector = build_fcos(cfg, in_channels)
        self.proposal_matcher = proposal_matcher
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.use_box_aug = cfg.MODEL.ALIGN.USE_BOX_AUG
        self.use_retrieval = cfg.MODEL.ALIGN.USE_RETRIEVAL
        if self.use_retrieval:
            self.head = AlignHead(cfg, in_channels)


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

            return None, None

        else:
            boxes, losses = self.detector(images, features[1:], targets)
            rec_features = features[:len(self.scales)]

            new_boxes = []
            for box, target in zip(boxes, targets):
                scores = box.get_field("scores")
                # pos_idxs = torch.nonzero(scores>0.08).view(-1)
                # pos_idxs = torch.nonzero(scores>0.05).view(-1)
                # pos_idxs = torch.nonzero(scores>0.05).view(-1)
                pos_idxs = torch.nonzero(scores>target.get_field("det_thred")).view(-1)#75.43
                # pos_idxs = torch.nonzero(scores>0.23).view(-1)#75.43
                box = box[pos_idxs]
                box.add_field("texts", target.get_field("texts"))
                box.add_field("scale", target.get_field("scale"))
                box.add_field("path", target.get_field("path"))
                # box.add_field("y_trues", target.get_field("y_trues"))
                new_boxes.append(box)
            image_names = [os.path.basename(str(image.get_field("path"))) for image in new_boxes]
            results, other = self.head(rec_features,{"retrieval_samples":new_boxes},images,is_words=is_words)
            return results, other

        # preds, _ = self.head(rec_features, boxes)
        


@registry.ONE_STAGE_HEADS.register("align")
def build_align_head(cfg, in_channels):
    return AlignModule(cfg, in_channels, 
                        Matcher(
                        0.7,
                        0.5,
                        allow_low_quality_matches=False)
                    )

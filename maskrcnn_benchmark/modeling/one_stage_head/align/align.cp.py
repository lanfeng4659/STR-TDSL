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
from .box_aug import make_box_aug
INF = 10000000000



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
        self.use_character_awareness = cfg.MODEL.ALIGN.USE_CHARACTER_AWARENESS
        self.is_chinese = cfg.MODEL.ALIGN.IS_CHINESE
        self.use_along_loss = cfg.MODEL.ALIGN.USE_ALONG_LOSS
        self.use_n_gram_ed = cfg.MODEL.ALIGN.USE_N_GRAM_ED
        self.use_common_space = cfg.MODEL.ALIGN.USE_COMMON_SPACE
        self.use_hanming = cfg.MODEL.ALIGN.USE_HANMING
        self.use_step = cfg.MODEL.ALIGN.USE_STEP
        self.use_char_count = cfg.MODEL.ALIGN.USE_CHAR_COUNT
        self.use_contrastive_loss = cfg.MODEL.ALIGN.USE_CONTRASTIVE_LOSS
        self.use_textness = cfg.MODEL.ALIGN.USE_TEXTNESS
        self.use_iou_predictor = cfg.MODEL.ALIGN.USE_IOU_PREDICTOR
        self.use_word_aug = cfg.MODEL.ALIGN.USE_WORD_AUG
        self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        self.pooler = Pooler(
            output_size=resolution,
            scales=self.scales,
            sampling_ratio=1,
            canonical_scale=canonical_scale,
            mode='align')
        self.textness_pooler = Pooler(
            output_size=(7,7),
            scales=self.scales,
            sampling_ratio=1,
            canonical_scale=canonical_scale,
            mode='align')
        self.image_pooler = Pooler(
            output_size=(64,128),
            scales=self.scales,
            sampling_ratio=1,
            canonical_scale=canonical_scale,
            mode='align')
        
        out_channels = 128
        in_channels = 256
        if self.is_chinese:
            lexicon = np.load("./datasets/rctw/chars.npy").tolist()
            self.text_generator = TextGenerator(ratios=[1,0,1,5],chars=lexicon)
        else:
            self.text_generator = TextGenerator()
        self.box_augumentor = make_box_aug()
        self.image_embedding = RNNDecoder(in_channels, out_channels,bidirectional=True)
        self.word_embedding = WordEmbedding(out_channels=out_channels,
                      embedding_dim=256,
                      char_vector_dim=256,
                      max_length=resolution[1],
                      lexicon = self.text_generator.chars,
                      bidirectional=True)
        if self.use_character_awareness:
            conv_func = conv_with_kaiming_uniform(True, True, False, False)
            self.char_conv = nn.Sequential(
                conv_func(in_channels, 64, 3, stride=(2, 1)),
            )
            self.character_awareness = nn.Sequential(
                nn.Linear(64*30,len(self.text_generator.chars)),
                )
        if self.use_textness:
            self.textness = nn.Sequential(
                nn.Linear(in_channels*7*7,128),
                nn.Linear(128,2))
        if self.use_iou_predictor:
            self.iou_predictor = nn.Sequential(
                nn.Linear(in_channels*7*7,128),
                nn.ReLU(),
                nn.Linear(128,1),
                )
        if self.use_char_count:
            conv_func = conv_with_kaiming_uniform(True, True, False, False)
            self.char_conv = nn.Sequential(
                conv_func(in_channels, out_channels, 3, stride=(2, 1)),
            )
            self.char_count_predictor = nn.Sequential(
                nn.Linear(128*30,20),
                )
        self.feat_dim = 1920
        if self.use_common_space:
            self.common_space = nn.Linear(out_channels, 64)
            self.feat_dim = 960
        # self.embedding_nor = nn.Linear(out_channels * 2, 32)
        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
        self.criterion = nn.CrossEntropyLoss()
        self.character_awarenness_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
    @torch.no_grad()
    def get_word_embedding(self,texts,device):
        words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(device) for text in texts]
        words_embedding = self.word_embedding(words).detach()
        return words_embedding
    def compute_contrastive_loss(self, embedding1, words1):
        all_similarity = []
        for idx,text in enumerate(words1):
            texts = [text]
            for i in range(20):
                texts.append(self.text_generator(text))
            words_embedding = self.get_word_embedding(texts,embedding1.device)
            all_similarity.append(self.compute_similarity(embedding1[idx].view(1,-1), words_embedding))
        all_similarity = torch.cat(all_similarity)
        labels = torch.zeros(all_similarity.shape[0], dtype=torch.long).to(embedding1.device)
        # print(all_similarity.shape, labels.shape)
        loss = self.criterion(all_similarity, labels)
        return loss
    def hanming_distance(self,a,b):
        c = a.mm(b.t())
        d = (a.sum(dim=1)[:,None].repeat((1,b.size(0))) + b.sum(dim=1)[None,:].repeat((a.size(0),1))).detach()
        return 2*c/d
    def compute_loss(self, embedding1, embedding2, words1,words2):
        # print(embedding1)
        if self.use_n_gram_ed:
            b,t,c = embedding1.size()
            k = 50 if self.use_step else 1
            iou = self.compute_similarity(embedding1, embedding2, k)
            former_iou = self.compute_similarity(embedding1[:,:t//2,:], embedding2[:,:t//2,:], k)
            latter_iou = self.compute_similarity(embedding1[:,t//2:,:], embedding2[:,t//2:,:], k)

            similarity = self.text_generator.calculate_similarity_matric(words1, words2)
            former_similarity = self.text_generator.calculate_similarity_matric([word[:(len(word)//2+1)] for word in words1], [word[:(len(word)//2+1)] for word in words2])
            latter_similarity = self.text_generator.calculate_similarity_matric([word[(len(word)//2):] for word in words1], [word[(len(word)//2):] for word in words2])
            
            all_loss = self.sim_loss_func(iou, torch.tensor(similarity).type_as(iou)).max(dim=1)[0].mean()
            former_loss = self.sim_loss_func(former_iou, torch.tensor(former_similarity).type_as(iou)).max(dim=1)[0].mean()
            latter_loss = self.sim_loss_func(latter_iou, torch.tensor(latter_similarity).type_as(iou)).max(dim=1)[0].mean()
            loss = (all_loss+former_loss+latter_loss)/3
            # print(loss)
        if self.use_hanming:
            embedding1_nor = embedding1.sigmoid().view(embedding1.size(0),-1)
            embedding2_nor = embedding2.sigmoid().view(embedding2.size(0),-1)
            inter = self.hanming_distance(embedding1_nor,embedding2_nor)
            # print(inter.max())
            similarity = self.text_generator.calculate_similarity_matric(words1, words2)
            # print(similarity.max())
            loss = self.sim_loss_func(inter, torch.tensor(similarity).type_as(inter))
            loss = loss.max(dim=1)[0].mean()
        else:
            k = 50 if self.use_step else 1
            iou = self.compute_similarity(embedding1, embedding2, k)
            # print(iou.shape, len(words1),len(words2))
            similarity = self.text_generator.calculate_similarity_matric(words1, words2)
            
            loss = self.sim_loss_func(iou, torch.tensor(similarity).type_as(iou))
            loss = loss.max(dim=1)[0].mean()
        # print(loss)
        return loss
    def compute_along_loss(self, embedding1, embedding2, words1,words2):
        def similarity(x,y):
            _,t,c = x.size()
            sims = []
            for i in range(1,t+1):
                x_t = torch.nn.functional.normalize(x[:,:i,:].tanh().view(x.size(0),-1))
                y_t = torch.nn.functional.normalize(y[:,:i,:].tanh().view(y.size(0),-1))
                sims.append(x_t.mm(y_t.t()))
            return torch.stack(sims,dim=-1)
        iou = similarity(embedding1, embedding2)
        similarity_map = torch.zeros([len(words1),len(words2),embedding1.size(1)])
        similarity_map = self.text_generator.calculate_along_similarity_matric(similarity_map,words1, words2).to(embedding1.device)
        
        loss = self.sim_loss_func(iou, similarity_map).mean(dim=-1)
        loss = loss.max(dim=1)[0].mean()
        # print(loss)
        return loss
    def compute_similarity(self,embedding1, embedding2,k=1):
        embedding1_nor = nn.functional.normalize((embedding1*k).tanh().view(embedding1.size(0),-1))
        embedding2_nor = nn.functional.normalize((embedding2*k).tanh().view(embedding2.size(0),-1))
        inter = embedding1_nor.mm(embedding2_nor.t())
        return inter
    # def compute_similarity(self,embedding1, embedding2,k=1):
    #     embedding1_nor = nn.functional.normalize((embedding1*k).sigmoid().view(embedding1.size(0),-1))
    #     embedding2_nor = nn.functional.normalize((embedding2*k).sigmoid().view(embedding2.size(0),-1))
    #     inter = embedding1_nor.mm(embedding2_nor.t())
    #     return inter
    # def test_pooler(self, images, proposals):
    #     import cv2
    #     import shutil
    #     images = to_image_list(images).tensors
    #     # print(images.shape)
    #     rois = self.image_pooler([images[:,:,::4,::4],images[:,:,::8,::8],images[:,:,::16,::16]], proposals)
    #     # texts = []
    #     rois = rois.permute(0,2,3,1).float()
    #     rois_de = denormalize(rois).data.cpu().numpy().astype(np.uint8)
    #     start = 0
    #     save_path = 'rois'
    #     for idx, proposals_per_im in enumerate(proposals):
    #         num = proposals_per_im.get_field("positive_num").data.cpu().numpy()[0]
    #         if num==0:
    #             continue
    #         print("have rois")
    #         end = start+num
    #         texts = proposals_per_im.get_field("texts")
            
    #         for image_np, text in zip(rois_de[start:end,:,:,:], texts[:num]):
    #             img_save_path = os.path.join(save_path, text+'.jpg')
    #             cv2.imwrite(img_save_path, image_np)
    #         start+=proposals_per_im.bbox.size(0)
    def test_pooler(self, images, proposals):
        def show_roi(image,rois,texts,size):
            h,w,c = image.shape
            rois_num = rois.shape[0]
            roi_h, roi_w = size
            num_per_col = int(h//roi_h)
            num_col = int(rois_num//num_per_col+1)*2
            new_h,new_w = h, w+num_col*roi_w
            mask = np.zeros([new_h, new_w,c])
            mask[:h,:w,:] = image
            for idx, (roi,text) in enumerate(zip(rois, texts)):
                idx_h = (idx%num_per_col)*roi_h
                idx_w = (idx//num_per_col)*roi_w*2 + w
                print(idx_h,roi_h,idx_w,roi_w,roi.shape)
                mask[idx_h:idx_h+roi_h,idx_w:idx_w+roi_w,:] = roi
                cv2.putText(mask,text.lower(),(idx_w+roi_w,idx_h+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            return mask
            
        import cv2
        import shutil
        save_path='rois'
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        #     os.makedirs(save_path)
        images = to_image_list(images).tensors
        images_bhwc = images.permute(0,2,3,1)
        images_bhwc_de = denormalize(images_bhwc)

        for per_image, proposals_per_img in zip(images_bhwc_de, proposals):
            # if proposals_per_img.get_field("positive_num").size(0)==0:
            #     continue
            # num = proposals_per_img.get_field("positive_num").data.cpu().numpy()[0]
            # if num==0:
            #     continue
            num = proposals_per_img.bbox.size(0)
            texts = proposals_per_img.get_field("texts")[:num]
            # texts = proposals_per_img.get_field("texts")
            # if num==0:
            #     continue
            per_image_bchw = per_image.permute(2,0,1)
            rois_per_image = self.image_pooler([per_image_bchw[None,:,::4,::4],per_image_bchw[None,:,::8,::8],per_image_bchw[None,:,::16,::16]], [proposals_per_img])
            rois_per_image = rois_per_image.permute(0,2,3,1).data.cpu().numpy().astype(np.uint8)
            boxes = proposals_per_img.bbox[:num,:].data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
            image_np = per_image.data.cpu().numpy()
            cv2.drawContours(image_np, boxes, -1, color=(255,0,0), thickness=1)
            new_image = show_roi(image_np,rois_per_image[:num,:,:,:],texts,(64,128))
            img_save_path = os.path.join(save_path, '{}.jpg'.format(np.random.randint(0,999)))
            cv2.imwrite(img_save_path, new_image)
        
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        #     os.makedirs(save_path)
        
        # print(rois.shape)
    def test_textness(self, images, proposals):
        import cv2
        import shutil
        images = to_image_list(images).tensors
        # print(images.shape)
        rois = self.image_pooler([images[:,:,::4,::4],images[:,:,::8,::8],images[:,:,::16,::16]], proposals)
        # texts = []
        pos_neg = []
        for proposals_per_im in proposals:
            pos_neg.extend(proposals_per_im.get_field("textness").data.cpu().numpy().tolist())
            # pos_neg.extend()
        rois = rois.permute(0,2,3,1).float()
        rois_de = denormalize(rois).data.cpu().numpy().astype(np.uint8)
        save_path = 'textness'
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        #     os.makedirs(save_path)
        for image_np, pn in zip(rois_de, pos_neg):
            img_save_path = os.path.join(save_path, '{}_{}.jpg'.format(np.random.randint(0,999),pn))
            cv2.imwrite(img_save_path, image_np)
        # print(rois.shape)
    def test_visual(self,images,boxes,image_name):
        image_tensor = images.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)[0]
        boxes = boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        cv2.drawContours(image_de, boxes, -1, color=(255,0,0), thickness=1)
        img_path = os.path.join('temp',image_name)
        cv2.imwrite(img_path, image_de)
    def iou_head(self,x,samples):
        if not self.training:
            rois = self.textness_pooler(x, samples)
            iou_pred = self.iou_predictor(rois.view(rois.size(0),-1)).view(-1)
            return iou_pred
        new_samples = [self.box_augumentor(sample) for sample in samples]
        iou_target = torch.cat([sample.get_field("iou") for sample in new_samples])
        # print("iou_target:",iou_target)
        rois = self.textness_pooler(x, new_samples)
        if rois.size(0)==0:
            iou_loss = x[0].sum()*0
            return iou_loss
        # imgs_embedding = self.image_embedding(rois)
        iou_pred = self.iou_predictor(rois.view(rois.size(0),-1)).view(-1)
        iou_loss = F.smooth_l1_loss(iou_pred.sigmoid(), iou_target)
        return iou_loss
    def char_count_head(self,rois,texts=None):
        vector = self.char_conv(rois)
        preds = self.char_count_predictor(vector.view(vector.size(0),-1))
        # print(preds.shape)
        if self.training:
            labels = torch.tensor([len(text) for text in texts], dtype=torch.long).to(rois.device).clamp(0,20-1)
            loss = F.cross_entropy(preds,labels)
            return loss
        else:
            return preds
    def character_awareness_head(self,rois,texts=None):
        vector = self.char_conv(rois)
        preds = self.character_awareness(vector.view(vector.size(0),-1))
        # print(preds.shape)
        if self.training:
            labels = torch.tensor(self.text_generator.phoc_level_1(texts), dtype=torch.int32).to(rois.device).view(-1)
            # print(preds.shape,labels.shape)
            loss = self.character_awarenness_loss_func(preds.view(-1,1),labels)/ max(labels.view(-1).size(0), 1.0)
            return loss
        else:
            preds = preds.sigmoid()
            masks = torch.tensor(self.text_generator.phoc_level_1(texts), dtype=torch.float32).to(rois.device) #N_text
            char_scores = preds.mm(masks.t())/masks.sum(dim=1).view(1,-1) #N_roi, N_text
            # print(char_scores.shape,char_scores.max())
            return char_scores

    def forward(self, x, samples,images=None,is_words=None):
        """
        offset related operations are messy
        images: used for test pooler
        """        
        if self.training:
            proposals = samples["retrieval_samples"]
            texts = []
            new_proposals = []
            for proposals_per_im in proposals:
                # idxs, new_texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                # texts.extend(new_texts)
                # new_proposals.append(proposals_per_im[idxs])

                # idxs, new_texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                texts.extend(proposals_per_im.get_field("texts").tolist())
                new_proposals.append(proposals_per_im)
            rois = self.pooler(x, new_proposals)
            # print(rois.shape)
            # exit()
            if self.use_iou_predictor:
                iou_loss = self.iou_head(x, new_proposals)
            if rois.size(0)==0:
                zero_loss = x[0].sum()*0
                loss = {"loss_wi":zero_loss,"loss_ww":zero_loss,"loss_ii":zero_loss}
                if self.use_contrastive_loss:
                    loss["loss_ct"] = zero_loss
                if self.use_textness:
                    loss["loss_roi_cls"] = zero_loss
                if self.use_iou_predictor:
                    loss["loss_iou"] = zero_loss
                return None,loss
            # print(new_proposals)
            # self.test_pooler(images, new_proposals)
            imgs_embedding = self.image_embedding(rois)
            if self.use_common_space:
                imgs_embedding = self.common_space(imgs_embedding)
            assert imgs_embedding.size(0) == len(texts),print(imgs_embedding.size(0),len(texts))
            word_texts = texts.copy()
            if self.use_word_aug:
                word_texts.extend([self.text_generator(text) for text in texts])
            if not self.is_chinese:
                words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(rois.device) for text in word_texts]
            else:
                words = [torch.tensor(self.text_generator.label_map(text)).long().to(rois.device) for text in word_texts]
            words_embedding = self.word_embedding(words)
            if self.use_common_space:
                words_embedding = self.common_space(words_embedding)
            # words_embedding_nor = nn.functional.normalize(words_embedding.tanh()).view(words_embedding.size(0),-1)
            loss_fn = self.compute_along_loss if self.use_along_loss else self.compute_loss
            wi_loss = loss_fn(words_embedding.detach(), imgs_embedding, word_texts, texts)
            ww_loss = loss_fn(words_embedding, words_embedding, word_texts, word_texts)
            ii_loss = loss_fn(imgs_embedding, imgs_embedding, texts, texts)
            
            loss = {"loss_wi":wi_loss*10,"loss_ww":ww_loss*10,"loss_ii":ii_loss*10}
            if self.use_character_awareness:
                # print("char:",rois.shape)
                # exit()
                loss["loss_ca"] = self.character_awareness_head(rois, texts)*10
            if self.use_char_count:
                # print("char:",rois.shape)
                # exit()
                loss["loss_cc"] = self.char_count_head(rois, texts)
            if self.use_contrastive_loss:
                loss["loss_ct"] = self.compute_contrastive_loss(imgs_embedding, texts)
            if self.use_iou_predictor:
                loss["loss_iou"] = iou_loss*10
                # print(loss)
            if self.use_textness:
                textness_samples = samples["textness_samples"]
                # self.test_pooler(images, textness_samples)
                # self.test_textness(images, textness_samples)
                # adaptive_avg_pool2d(input, output_size)
                rois = self.textness_pooler(x, textness_samples)
                # print(rois.shape)
                rois = rois.view(rois.size(0),-1)
                # print(imgs_embedding.shape)
                logits = self.textness(rois)
                labels = torch.cat([sample.get_field("textness") for sample in textness_samples]).to(logits.device).long()
                # print(logits.shape, labels.shape)
                # print(labels)
                cls_loss = F.cross_entropy(logits, labels)
                loss["loss_roi_cls"] = cls_loss*10
            # print(loss)
            return None,loss
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
                        if self.use_hanming:
                            words_embedding_nor = words_embedding.sigmoid().view(words_embedding.size(0),-1)
                        else:
                            k = 50 if self.use_step else 1
                            words_embedding_nor = nn.functional.normalize((words_embedding*k).tanh().view(words_embedding.size(0),-1))
                        # words_embedding_nor = nn.functional.normalize(words_embedding.view(words_embedding.size(0),-1))
                if proposals_per_im.bbox.size(0) == 0:
                    imgs_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
                else:
                    
                    if self.use_textness:
                        rois = self.pooler(x, [proposals_per_im])
                        rois_for_logits = self.textness_pooler(x, [proposals_per_im]).view(rois.size(0),-1)
                        scores = F.softmax(self.textness(rois_for_logits),dim=1)[:,1]
                        # print(scores)
                        pos_idxs = torch.nonzero((scores>0.95)).view(-1)
                        # pos_idxs = torch.nonzero((scores>0)).view(-1)
                        # print(pos_idxs.numel(),proposals_per_im)
                        rois = rois[pos_idxs,:,:,:] if pos_idxs.numel()>0 else rois
                        select_boxes.append(proposals_per_im.bbox[pos_idxs,:])
                        
                    else:
                        rois = self.pooler(x, [proposals_per_im])
                        select_boxes.append(proposals_per_im.bbox)
                    if self.use_character_awareness:
                        char_awareness = self.character_awareness_head(rois, texts)
                        proposals_per_im.add_field("char_awareness", char_awareness)
                    if self.use_char_count:
                        counts = self.char_count_head(rois)
                        proposals_per_im.add_field("char_counts", counts)
                    if self.use_iou_predictor:
                        iou_scores = self.iou_head(x, [proposals_per_im])
                        proposals_per_im.add_field("iou", iou_scores)
                    imgs_embedding = self.image_embedding(rois)
                    if self.use_hanming:
                        imgs_embedding_nor = imgs_embedding.sigmoid().view(imgs_embedding.size(0),-1)
                    else:
                        k = 50 if self.use_step else 1
                        imgs_embedding_nor = nn.functional.normalize((imgs_embedding*k).tanh().view(imgs_embedding.size(0),-1))
                    # imgs_embedding_nor = nn.functional.normalize(imgs_embedding.view(imgs_embedding.size(0),-1))
                proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
                if is_words:
                    proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)

            return proposals, {"select_boxes":select_boxes}

def denormalize(image):
    std_ = torch.tensor([[57.375, 57.12, 58.395]]).to(image.device)
    mean_ = torch.tensor([[103.53, 116.28, 123.675]]).to(image.device)
    image.mul_(std_).add_(mean_)
    return image
import os
import numpy as np
from PIL import Image, ImageDraw
def vis_pss_map(img, pss, ori_h, ori_w):
    im = img.copy()
    img = Image.fromarray(im).convert('RGB').resize((ori_w, ori_h))
    pss_img = Image.fromarray((pss*255).astype(np.uint8)).convert('RGB').resize((ori_w, ori_h))
    pss_img = Image.blend(pss_img, img, 0.5)
    return pss_img
def vis_multi_image(image_list, shape=[1,-1]):
    image_num = len(image_list)
    h, w,_ = np.array(image_list[0]).shape
    #print h,w
    num_w = int(image_num/shape[0])
    num_h = shape[0]
    new_im = Image.new('RGB', (num_w*w,num_h*h))
    for idx, image in enumerate(image_list):
        idx_w = idx%num_w
        idx_h = int(idx/num_w)
        new_im.paste(image, (int(idx_w*w),int(idx_h*h)))
    return new_im
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
        self.batch_size_per_image = 256 
        self.positive_fraction = 0.25
        self.use_textness = cfg.MODEL.ALIGN.USE_TEXTNESS
    def visual(self,images,boxes):
        image_tensor = images.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)
        maps = boxes.view(-1,1).float()
        maps = maps.data.cpu().numpy()
        # print(maps.shape)
        nums = [6400,8000,8400,8500,8525]
        maps = [maps[:6400,:].reshape([80,80]),maps[6400:8000,:].reshape([40,40]),maps[8000:8400,:].reshape([20,20]),maps[8400:8500,:].reshape([10,10]),maps[8500:,:].reshape([5,5])]
        img_list = [Image.fromarray(image_de[0]).convert('RGB')]
        for single_map in maps:
            # single_map.reshape()
            img = vis_pss_map(image_de[0], single_map, 640,640)
            img_list.append(img)
        new_img = vis_multi_image(img_list,shape=[2,-1])
        img_path = os.path.join('temp','img_{}.jpg'.format(np.random.randint(0,999)))
        print(img_path)
        new_img.save(img_path)

        return None
    def test_visual(self,images,boxes,select_boxes,image_name):
        image_tensor = images.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)[0]
        boxes = boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        image = image_de.copy()
        cv2.drawContours(image, boxes, -1, color=(255,0,0), thickness=1)
        # if len(select_boxes)>0:
        #     boxes = select_boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        #     image2 = image_de.copy()
        #     cv2.drawContours(image2, boxes, -1, color=(255,0,0), thickness=1)
        #     image = np.concatenate((image, image2),axis=1)

        #     image = np.ascontiguousarray(image)
        # print(image.shape)
        img_path = os.path.join('temp',image_name)
        cv2.imwrite(img_path, image)

        
        

        return None
    def match_targets_to_proposals(self, proposal, target):
        # TODO  when use rotated box, use polygon iou
        # print(target.bbox, proposal.bbox)
        # print(proposal, target)
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        # FIXME need poly in next
        # target = target.copy_with_fields("labels")
        
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # matched_targets = target[matched_idxs.clamp(min=0)]
        # if target.bbox.shape[0]:
        #     # print(matched_idxs, matched_idxs.clamp(min=0))
        #     matched_targets = target[matched_idxs.clamp(min=0).data.cpu().numpy()]
        # else:
        #     # target.add_field ("labels", matched_idxs.clamp(min=1, max=1))
        #     fg_inds = torch.nonzero(matched_idxs >= 0).squeeze(1).data.cpu().numpy()
        #     # labels_per_image[bg_inds] = 0
        #     matched_targets = target[fg_inds]
        # matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_idxs
    def select_positive_boxes(self, boxes, targets):
        new_boxes = []
        for boxes_per_image, targets_per_image in zip(boxes, targets):
            if boxes_per_image.bbox.size(0) == 0:
                boxes_per_image = targets_per_image
            matched_idxs = self.match_targets_to_proposals(
                boxes_per_image, targets_per_image
            )
            # print(matched_targets, boxes_per_image)
            # assert matched_targets.bbox.size(0) == boxes_per_image.bbox.size(0)
            # matched_idxs = matched_targets.get_field("matched_idxs")
            positive = torch.nonzero(matched_idxs >= 0).squeeze(1).data.cpu().numpy()
            positive_matched_targets = targets_per_image[positive]
            positive_boxes = boxes_per_image[positive]
            positive_boxes.add_field("texts", positive_matched_targets.get_field("texts"))
            # targets_per_image.add_field("scores", positive_boxes.get_field("scores"))
            positive_boxes = targets_per_image if positive_boxes.bbox.size(0)==0 else positive_boxes
            new_boxes.append(positive_boxes)
        return new_boxes
    def prepare_training_samples_for_textness_retrieval(self, boxes, targets):
        retrieval_samples = []
        textness_samples = []
        for boxes_per_image, targets_per_image in zip(boxes, targets):
            # print(boxes_per_image)
            # not_difficult_idx = [i for i, text in enumerate(targets_per_image.get_field("texts")) if text != "###"]
            # if len(not_difficult_idx)==0:
            #     continue
            # targets_per_image = targets_per_image[not_difficult_idx]
            if boxes_per_image.bbox.size(0) == 0:
                boxes_per_image = targets_per_image
            matched_idxs = self.match_targets_to_proposals(
                boxes_per_image, targets_per_image
            )
            # print(matched_targets, boxes_per_image)
            # assert matched_targets.bbox.size(0) == boxes_per_image.bbox.size(0)
            # matched_idxs = matched_targets.get_field("matched_idxs")
            
            #retrieval_samples
            positive = torch.nonzero(matched_idxs >= 0).squeeze(1).data.cpu().numpy()
            positive_matched_targets = targets_per_image[matched_idxs[positive].data.cpu().numpy()]
            positive_boxes = boxes_per_image[positive]
            assert positive_matched_targets.bbox.size(0) == positive_boxes.bbox.size(0)
            
            # if matched_idxs.max()>=0:
            #     print(matched_idxs)
            #     print(positive_boxes.bbox.size(0))
            positive_boxes = positive_boxes.clone_without_fields()
            positive_boxes.add_field("texts", positive_matched_targets.get_field("texts"))
            
            targets_per_image2 = targets_per_image.clone_without_fields()
            targets_per_image2.add_field("texts", targets_per_image.get_field("texts"))
            # print(positive_boxes.fields(),targets_per_image.fields())
            positive_boxes = targets_per_image2 if positive_boxes.bbox.size(0)==0 else cat_boxlist_texts([positive_boxes, targets_per_image2])
            # positive_boxes = targets_per_image2
            # positive_boxes.add_field("positive_num",torch.tensor([positive_boxes.bbox.size(0)]*(positive_boxes.bbox.size(0)+1)))
            # positive_boxes.add_field("positive_num",torch.tensor([positive_boxes.bbox.size(0)-targets_per_image2.bbox.size(0)]*(positive_boxes.bbox.size(0)+1)))
            retrieval_samples.append(positive_boxes)

            #textness_samples
            positive_boxes = positive_boxes.clone_without_fields()
            negative = torch.nonzero(matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD).squeeze(1)
            # print(negative.numel())
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # print(positive.shape[0])
            num_pos = min(positive_boxes.bbox.size(0), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            neg_idx_per_image = negative[perm2].data.cpu().numpy()
            
            negative_boxes = boxes_per_image[neg_idx_per_image].clone_without_fields()
            positive_boxes.add_field("textness",torch.ones([positive_boxes.bbox.size(0)], dtype=torch.uint8))
            negative_boxes.add_field("textness",torch.zeros([negative_boxes.bbox.size(0)], dtype=torch.uint8))
            # print(positive_boxes, negative_boxes)
            samples = cat_boxlist([positive_boxes, negative_boxes])
            
            # print(samples)
            textness_samples.append(samples)



        return retrieval_samples,textness_samples


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
            boxes, losses = self.detector(images, features[1:], targets)
            if not self.use_retrieval:
                return None, losses
            if not self.use_textness:
                # _, loss_dict = self.head(rec_features, targets,images)
                # new_boxes = self.select_positive_boxes(boxes, targets) if self.use_box_aug else targets
                new_boxes = targets
                _, loss_dict = self.head(rec_features, {"retrieval_samples":new_boxes},images)

                # self.visual(images, boxes)
                
                
                for k, v in loss_dict.items():
                    losses.update({k: v})
                return None, losses
            else:
                retrieval_samples,textness_samples = self.prepare_training_samples_for_textness_retrieval(boxes, targets)
                _, loss_dict = self.head(rec_features, {"retrieval_samples":retrieval_samples,"textness_samples":textness_samples},images)

                # self.visual(images, boxes)
                
                
                for k, v in loss_dict.items():
                    losses.update({k: v})
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
            results, other = self.head(rec_features,{"retrieval_samples":new_boxes},images,is_words=is_words)
            # results, other = self.head(rec_features,{"retrieval_samples":targets},images,is_words=is_words)
            # self.test_visual(images, new_boxes[0].bbox,other["select_boxes"][0],image_names[0])
            # results, _ = self.head(rec_features,targets,images)
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

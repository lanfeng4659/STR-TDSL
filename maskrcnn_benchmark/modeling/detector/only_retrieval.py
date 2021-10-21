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
class GlobalLocalSimilarity(nn.Module):
    
    def __init__(self,divided_nums = [1,3,5]):
        super(GlobalLocalSimilarity, self).__init__()
        self.divided_nums = divided_nums
        self.normalize = nn.functional.normalize
    def compute_similarity(self,x,y,divided_num=1):
        x = x.view(x.size(0),divided_num,-1)
        y = y.view(y.size(0),divided_num,-1)
        sims = torch.stack([self.normalize(x[:,i,:]).mm(self.normalize(y[:,i,:]).t()) for i in range(divided_num)],dim=-1)
        return sims.mean(dim=-1)
        
    def forward(self, x,y):
        x_tanh = x.tanh()
        y_tanh = y.tanh()
        sims = torch.stack([self.compute_similarity(x_tanh, y_tanh, divided_num) for divided_num in self.divided_nums],dim=-1)
        return sims.mean(dim=-1)
class DynamicMaxSimilarity(nn.Module):
    
    def __init__(self,frame_num):
        super(DynamicMaxSimilarity, self).__init__()
        self.frame_num = frame_num
    # def sim(self,x,y):
    #     # print(x.shape)
    #     x_nor = torch.nn.functional.normalize(x.view(x.size(0),-1).tanh())
    #     y_nor = torch.nn.functional.normalize(y.view(y.size(0),-1).tanh())
    #     return x_nor.mm(y_nor.t())
    def sim(self,x,y):
        x_nor = torch.nn.functional.normalize(x.view(-1,x.size(-1)).tanh()) # x_bw,c
        y_nor = torch.nn.functional.normalize(y.view(-1,y.size(-1)).tanh()) # y_bw,c
        similarity = x_nor.mm(y_nor.t()) # (x_bw,y_bw)
        similarity = similarity.reshape([x.size(0),x.size(1),y.size(0),y.size(1)])
        return similarity.permute(0,2,1,3)
    def push_similarity(self,global_sim, local_sim, steps):
        return (global_sim*(steps-1)+local_sim)/steps
    def forward(self,a,b):
        si = torch.zeros([a.size(0),b.size(0),self.frame_num+1, self.frame_num+1]).type_as(a)
        local_similarity = self.sim(a,b)
        for i in range(1, self.frame_num+1):
            for j in range(1, self.frame_num+1):
                local_sim = local_similarity[:,:,i-1,j-1]
                all_sim = torch.stack([self.push_similarity(si[:,:,i-1,j], local_sim, max(i,j)), 
                                       self.push_similarity(si[:,:,i,j-1], local_sim, max(i,j)), 
                                       self.push_similarity(si[:,:,i-1,j-1], local_sim, max(i,j))]
                                       ,dim=-1)
                si[:,:,i,j] = torch.max(all_sim,dim=-1)[0]
        return si[:,:,-1,-1]
class PyramidFeatures(nn.Module):
    
    def __init__(self,layers):
        super(PyramidFeatures, self).__init__()
        self.layers = layers

    def forward(self, features):
        b,w,c = features.size()
        features = features.view(b,1,w,c)
        pyramids = torch.cat([nn.functional.adaptive_avg_pool2d(features,[l,c]) for l in self.layers], dim=2)
        output = torch.cat([features, pyramids],dim=2).squeeze(dim=1)
        return output
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)
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
        self.use_look_up = use_look_up
        self.use_res_link = use_res_link
        self.use_pyramid = use_pyramid
        if self.use_pyramid:
            self.pyramid = PyramidFeatures(pyramid_layers)
        if self.use_look_up:
            self.look_up_model = nn.MultiheadAttention(in_channels, 1)
            # self.selection = nn.Linear(2*in_channels,1)
        if self.use_res_link:
            self.pro = nn.Linear(in_channels, out_channels)
    def look_up(self,q,k,v):
        # print(q.shape,k.shape,v.shape)
        feat,_ = self.look_up_model(q,k,v)
        return feat
    def forward(self, x, dictionary=None):
        x = self.convs(x)
        x = x.mean(dim=2)  # NxCxW
        # assert x.size(-2) == 1, "the height of conv must be 1"
        # x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # WxNxC
        if self.use_look_up:
            print("look_up")
            dictionary = dictionary[:,None,:].repeat([1,x.size(1),1])
            feat = self.look_up(x,dictionary,dictionary)
            x = feat
            # selection = self.selection(torch.cat([feat,x],dim=-1)).sigmoid()
            # print(selection.shape,x.shape)
            # x = selection*x + (1-selection)*feat
        if self.use_res_link:
            x = self.rnn(x)+ self.pro(x)
        else:
            x = self.rnn(x)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        if self.use_pyramid:
            x = self.pyramid(x)
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
        self.use_res_link = use_res_link
        self.use_pyramid = use_pyramid
        if self.use_pyramid:
            self.pyramid = PyramidFeatures(pyramid_layers)
        if self.use_res_link:
            self.pro = nn.Linear(256, out_channels)
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
        
        if self.use_res_link:
            x = self.rnn(char_vector) + self.pro(char_vector)
        else:
            x = self.rnn(char_vector)
        x = x.permute(1,0,2).contiguous()  # [b,w,c]
        if self.use_pyramid:
            x = self.pyramid(x)
        return x

class AlignHead(nn.Module):
    def __init__(self, cfg, in_channels=256):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AlignHead, self).__init__()
        self.use_ctc_loss = False
        self.use_word_aug = False

        
        out_channels = 128
        in_channels = 256
        conv_func = conv_with_kaiming_uniform(True, True, False, False)
        self.conv = conv_func(2048, in_channels, 1, stride=(1, 1))
        self.text_generator = TextGenerator()
        self.image_embedding = RNNDecoder(in_channels, out_channels,bidirectional=True)
        self.word_embedding = WordEmbedding(out_channels=out_channels,
                      embedding_dim=256,
                      char_vector_dim=256,
                      max_length=15,
                      lexicon = self.text_generator.chars,
                      bidirectional=True)
        frames = 15
        if self.use_ctc_loss:
            self.ctc_head = CTCPredictor(out_channels,len(self.text_generator.chars)+1)
        self.feat_dim = 128*frames 
        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
    @torch.no_grad()
    def get_word_embedding(self,texts,device):
        words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(device) for text in texts]
        words_embedding = self.word_embedding(words).detach()
        return words_embedding

    def compute_loss(self, embedding1, embedding2, words1,words2):
        k = 1
        iou = self.compute_similarity(embedding1, embedding2, k)
        similarity = self.text_generator.calculate_similarity_matric(words1, words2)
        similarity = torch.tensor(similarity).type_as(iou)
        # print(iou.max(),iou.min())
        loss = self.sim_loss_func(iou, similarity)
        loss = loss.max(dim=1)[0].mean()
        # print(loss)
        return loss
    def compute_similarity(self,embedding1, embedding2,k=1):
        embedding1_nor = nn.functional.normalize((embedding1*k).tanh().view(embedding1.size(0),-1))
        embedding2_nor = nn.functional.normalize((embedding2*k).tanh().view(embedding2.size(0),-1))
        inter = embedding1_nor.mm(embedding2_nor.t())
        return inter


    def forward(self, x, targets=None,is_words=None):
        """
        offset related operations are messy
        images: used for test pooler
        """ 
        x = self.conv(x)       
        if self.training:
            texts = [target.get_field("text") for target in targets]

            
            word_texts = texts.copy()
            imgs_texts = texts.copy()

            imgs_embedding = self.image_embedding(x)
            
            if self.use_word_aug:
                word_texts.extend([self.text_generator(text) for text in texts])
            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x.device) for text in word_texts]
            words_embedding = self.word_embedding(words)
            # words_embedding_nor = nn.functional.normalize(words_embedding.tanh()).view(words_embedding.size(0),-1)
            loss_fn = self.compute_loss
            wi_loss = loss_fn(words_embedding.detach(), imgs_embedding, word_texts, imgs_texts)
            ww_loss = loss_fn(words_embedding, words_embedding, word_texts, word_texts)
            ii_loss = loss_fn(imgs_embedding, imgs_embedding, imgs_texts, imgs_texts)
            
            loss = {"loss_wi":wi_loss*10,"loss_ww":ww_loss*10,"loss_ii":ii_loss*10}
            if self.use_ctc_loss:
                
                max_len = imgs_embedding.size(1)
                selected_idx = [i for i,text in enumerate(texts) if len(text)<max_len]
                selected_texts = [texts[i] for i in selected_idx]
                class_num = len(self.text_generator.chars)+1
                words = torch.tensor([self.text_generator.label_map_with_padding(text, max_len=max_len, padding=class_num-1) for text in selected_texts]).long().to(rois.device)
                loss["loss_rc"] = self.ctc_head(imgs_embedding[selected_idx], words)
            return None,loss
        else:
            # select_boxes = []
            # proposals = samples["retrieval_samples"]
            # for proposals_per_im in proposals:
            #     if not self.is_chinese:
            #         idxs, texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
            #     else:
            #         texts = proposals_per_im.get_field("texts").tolist()
            #     if is_words:
                    
            #         if len(texts) == 0:
            #             words_embedding_nor = torch.zeros([0,self.feat_dim]).type_as(x[0])
            #         else:
            #             if not self.is_chinese:
            #                 words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x[0].device) for text in texts]
            #             else:
            #                 words = [torch.tensor(self.text_generator.label_map(text)).long().to(x[0].device) for text in texts]
            #                 # print(len(words))
            #             words_embedding = self.word_embedding(words)
            #             if self.use_hanming:
            #                 words_embedding_nor = words_embedding.sigmoid().view(words_embedding.size(0),-1)
            #             else:
            #                 k = 50 if self.use_step else 1
            #                 if self.use_dynamic_similarity:
            #                     words_embedding_nor = words_embedding
            #                 else:
            #                     words_embedding_nor = nn.functional.normalize((words_embedding*k).tanh().view(words_embedding.size(0),-1))

            imgs_embedding = self.image_embedding(x)
            k=1
            imgs_embedding_nor = nn.functional.normalize((imgs_embedding*k).tanh().view(imgs_embedding.size(0),-1))
                # imgs_embedding_nor = nn.functional.normalize(imgs_embedding.view(imgs_embedding.size(0),-1))
            # proposals_per_im.add_field("imgs_embedding_nor", imgs_embedding_nor)
            # if is_words:
                # proposals_per_im.add_field("words_embedding_nor", words_embedding_nor)
            texts = targets[0].get_field("texts").tolist()
            # print(len(texts))
            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(x.device) for text in texts]
            words_embedding = self.word_embedding(words)
            words_embedding_nor = nn.functional.normalize((words_embedding*k).tanh().view(words_embedding.size(0),-1))
            return {"imgs_embedding_nor":imgs_embedding_nor,"words_embedding_nor":words_embedding_nor},None

            # return {"imgs_embedding_nor":imgs_embedding_nor},None


    
def build_retrieval_head(cfg):
    return AlignHead(cfg)

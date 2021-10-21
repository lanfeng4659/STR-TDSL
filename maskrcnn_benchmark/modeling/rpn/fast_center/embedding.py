import torch
from torch import nn
from torch.nn import functional as F
import cv2
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.utils.text_util import TextGenerator
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, cat_boxlist_texts
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torch.autograd import Variable
import string
import random
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.layers import SigmoidFocalLoss
import torch.nn.functional as F
INF = 10000000000
# class DepthwiseXCorr(nn.Module):
#     def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
#         super(DepthwiseXCorr, self).__init__()
#         self.conv_kernel = nn.Sequential(
#                 nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
#                 nn.BatchNorm2d(hidden),
#                 nn.ReLU(inplace=True),
#                 )
#         self.conv_search = nn.Sequential(
#                 nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
#                 nn.BatchNorm2d(hidden),
#                 nn.ReLU(inplace=True),
#                 )
#         self.head = nn.Sequential(
#                 nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(hidden, out_channels, kernel_size=1)
#                 )
        

#     def forward(self, kernel, search):
#         kernel = self.conv_kernel(kernel)
#         search = self.conv_search(search)
#         feature = xcorr_depthwise(search, kernel)
#         out = self.head(feature)
#         return out

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
        embeddings_batch = []
        for word in inputs:
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

class WordHead(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SiamHead, self).__init__()
        
        out_channels = 128
        in_channels = 256
        self.text_generator = TextGenerator()
        self.word_embedding = WordEmbedding(out_channels=out_channels,
                      embedding_dim=256,
                      char_vector_dim=256,
                      max_length=15,
                      lexicon = self.text_generator.chars,
                      bidirectional=True)
    def forward(self, words):
        """
        offset related operations are messy
        images: used for test pooler
        """        
        for _ in x:
            print(_.shape)
        if self.training:
            texts = []
            new_proposals = []
            for proposals_per_im in proposals:
                idxs, new_texts = self.text_generator.filter_words(proposals_per_im.get_field("texts").tolist())
                # texts.extend(new_texts)
                new_proposals.append(proposals_per_im[idxs])
            words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(rois.device) for text in texts]
            words_embedding = self.word_embedding(words)
            

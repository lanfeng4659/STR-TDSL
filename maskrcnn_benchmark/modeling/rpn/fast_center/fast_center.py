import math
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling.backbone.fbnet_builder import ShuffleV2Block
from maskrcnn_benchmark.utils.text_util import TextGenerator
from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator
from .predictors import make_offset_predictor
from .embedding import WordEmbedding
import time

def snv2_block(in_channels, out_channels, kernel_size, stride):
    return ShuffleV2Block(in_channels, out_channels, expansion=2, stride=stride, kernel=kernel_size)


class FastCenterHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FastCenterHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        if cfg.MODEL.FCOS.USE_LIGHTWEIGHT:
            conv_block = snv2_block
        else:
            conv_block = conv_with_kaiming_uniform(
                cfg.MODEL.FCOS.USE_GN, cfg.MODEL.FCOS.USE_RELU,
                cfg.MODEL.FCOS.USE_DEFORMABLE, cfg.MODEL.FCOS.USE_BN)

        for head in ['bbox']:
            tower = []
            for i in range(cfg.MODEL.FCOS.NUM_CONVS):
                tower.append(
                    conv_block(in_channels, in_channels, 3, 1))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        retrieval_conv = conv_with_kaiming_uniform(cfg.MODEL.FCOS.USE_GN, cfg.MODEL.FCOS.USE_RELU,cfg.MODEL.FCOS.USE_DEFORMABLE, cfg.MODEL.FCOS.USE_BN)
        self.retrieval_head = nn.Sequential(
            retrieval_conv(in_channels, in_channels,3,1),
            retrieval_conv(in_channels, 1920,1,1)
        )

        # self.poly_pred = nn.Conv2d(
        #     in_channels, 8, kernel_size=3, stride=1,
        #     padding=1)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        retrievals = []
        for l, feature in enumerate(x):
            bbox_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(bbox_tower))
            centerness.append(self.centerness(bbox_tower))
            bbox_reg.append(F.relu(self.bbox_pred(bbox_tower)))
            retrievals.append(self.retrieval_head(bbox_tower))
            
            # poly_reg.append(self.poly_pred(bbox_tower))
        return logits, bbox_reg, centerness, retrievals


class FastCenterModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FastCenterModule, self).__init__()

        self.cfg = cfg.clone()

        head = FastCenterHead(cfg, in_channels)

        box_selector_train = make_fcos_postprocessor(cfg, is_train=True)
        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_iters = 0
        self.text_generator = TextGenerator()
        self.word_embedding = WordEmbedding(
            out_channels=128,
            embedding_dim=256,
            char_vector_dim=256,
            max_length=15,
            lexicon = self.text_generator.chars,
            bidirectional=True)
        self.sim_loss_func = nn.SmoothL1Loss(reduction='none')
    # def retrieval(self, image_embedding, word_embedding):
    def compute_retrieval_similarity_gt(self,target,retrieval_texts,device):
        gt_texts = target.get_field("texts").tolist()

        # print(len(retrieval_texts), len(gt_texts))
        similarity = self.text_generator.calculate_similarity_matric(retrieval_texts, gt_texts)
        # print(similarity)
        target.add_field("similarity",torch.tensor(similarity).to(device))
        target.add_field("retrieval_texts",retrieval_texts)
        # print("similarity_gt.shape:", similarity.shape)
    """
    This is memory comsuming.
    """
    def compute_retrieval_similarity_pred(self,target,image_embedding, word_embedding):
        image_embedding_nor = nn.functional.normalize(image_embedding.tanh(), dim=0)
        word_embedding_nor = nn.functional.normalize(word_embedding.tanh(), dim=1)
        similarity = word_embedding_nor.mm(image_embedding_nor)
        # print(image_embedding_nor.shape,word_embedding_nor.shape,similarity.shape)
        target.add_field("similarity_pred",similarity)
        target.add_field("images_embedding_nor",image_embedding_nor)
        target.add_field("words_embedding_nor",word_embedding_nor)
    def compute_retrieval_similarity_pred_fast(self,target,image_embedding, word_embedding):
        image_embedding_nor = nn.functional.normalize(image_embedding.tanh(), dim=0)
        word_embedding_nor = nn.functional.normalize(word_embedding.tanh(), dim=1)
        target.add_field("images_embedding_nor",image_embedding_nor)
        target.add_field("words_embedding_nor",word_embedding_nor)

    def forward(self, images, features, targets=None, vis=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, image_embedding = self.head(features)
        
        
        if self.training:
            # all_word_embedding = []
            # all_words = []
            new_targets = [target[self.text_generator.filter_words(target.get_field("texts").tolist())[0]] for target in targets]
            for idx,target in enumerate(new_targets):
                texts = target.get_field("texts").tolist()
                words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(features[0].device) for text in texts]
                words_embedding_per_img = self.word_embedding(words).view(len(words),-1)
                image_embedding_per_img = torch.cat([per_level[idx].view(1920,-1) for per_level in image_embedding],dim=-1)
                self.compute_retrieval_similarity_gt(target,texts,device=features[0].device)
                self.compute_retrieval_similarity_pred_fast(target,image_embedding_per_img, words_embedding_per_img)
            targets = new_targets
        
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness,
                image_embedding,
                targets, images.image_sizes
            )
        else:
            # scale regression targets
            box_regression = [r * s for r, s in zip(box_regression, self.fpn_strides)]
            # new_targets = [target[self.text_generator.filter_words(target.get_field("texts").tolist())[0]] for target in targets]
            for idx,target in enumerate(targets):
                texts = target.get_field("texts").tolist()
                words = [torch.tensor(self.text_generator.label_map(text.lower())).long().to(features[0].device) for text in texts]
                words_embedding_per_img = self.word_embedding(words).view(len(words),-1)
                image_embedding_per_img = torch.cat([per_level[idx].view(1920,-1) for per_level in image_embedding],dim=-1)
                self.compute_retrieval_similarity_gt(target,texts,device=features[0].device)
                self.compute_retrieval_similarity_pred(target,image_embedding_per_img, words_embedding_per_img)
            boxes, _ = self._forward_test(locations, box_cls, box_regression,centerness, images.image_sizes)
            # print(len(box_cls), len(box_regression), len(centerness), len(image_embedding),len(features))
            return boxes, {"box_cls":box_cls,"centerness":centerness, "targets":targets}

    def _forward_train(self, locations, box_cls, box_regression,centerness,retrievals,
                       targets, image_sizes):
        loss_box_cls, loss_box_reg, loss_centerness,loss_sim,is_in_bboxes = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, retrievals, targets
        )
        all_word_embedding = torch.cat([target.get_field("words_embedding_nor") for target in targets])
        all_texts = []
        for target in targets:
            all_texts.extend(target.get_field("retrieval_texts"))
        similarity = self.text_generator.calculate_similarity_matric(all_texts, all_texts)
        similarity = torch.tensor(similarity).to(all_word_embedding.device).float()
        iou = all_word_embedding.mm(all_word_embedding.t())
        loss = self.sim_loss_func(iou, similarity)
        loss_sim_ww = loss.max(dim=1)[0].mean()
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_sim_wi": loss_sim*10,
            "loss_sim_ww": loss_sim_ww*10,
        }
        return None, losses

    def _forward_test(
            self, locations, box_cls, box_regression,
            centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fast_center(cfg, in_channels):
    return FastCenterModule(cfg, in_channels)

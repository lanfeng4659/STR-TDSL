# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..backbone.necks import build_neck
from maskrcnn_benchmark.modeling.one_stage_head import build_one_stage_head
from .only_retrieval import build_retrieval_head

class OneStage(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(OneStage, self).__init__()
        self.retrieval_only = cfg.MODEL.RETRIEVAL_ONLY
        self.backbone = build_backbone(cfg)
        if self.retrieval_only:
            self.retrieval_head = build_retrieval_head(cfg)
        else:
            self.neck = build_neck(cfg)
            self.decoder = build_one_stage_head(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, vis=False,is_words=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        if self.retrieval_only:
            feats = self.backbone(images.tensors)[-1]
            # print(feats[-1].shape)
            result,decoder_losses = self.retrieval_head(feats,targets)
        else:
            features = self.neck(self.backbone(images.tensors))
            result, decoder_losses = self.decoder(images, features, targets, vis=vis,is_words=is_words)
        # import pdb; pdb.set_trace()
        if self.training:
            losses = {}
            losses.update(decoder_losses)
            return losses

        return result

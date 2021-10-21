import math
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.structures.image_list import to_image_list
from .inference import make_east_postprocessor
from .loss import make_east_loss_evaluator

from maskrcnn_benchmark.layers import Scale
from maskrcnn_benchmark.layers import DFConv2d
from .visualization_map import Visualizater

class EASTHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(EASTHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.EAST.NUM_CLASSES - 1
        # self.fpn_strides = cfg.MODEL.EAST.FPN_STRIDES
        # self.norm_reg_targets = cfg.MODEL.EAST.NORM_REG_TARGETS
        self.use_dcn_in_tower = False

        cls_tower = []
        quad_tower = []
        for i in range(cfg.MODEL.EAST.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.EAST.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            quad_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            quad_tower.append(nn.GroupNorm(32, in_channels))
            quad_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('quad_tower', nn.Sequential(*quad_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        self.quad_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        # self.centerness = nn.Conv2d(
        #     in_channels, 1, kernel_size=3, stride=1,
        #     padding=1
        # )

        # initialization
        for modules in [self.cls_tower, self.quad_tower,
                        self.cls_logits, self.quad_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.EAST.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        quad_reg = []
        centerness = []
        for l, feature in enumerate([x[0]]):
            # print(feature.shape)
            cls_tower = self.cls_tower(feature)
            box_tower = self.quad_tower(feature)

            logits.append(self.cls_logits(cls_tower).sigmoid())
            quad_reg.append(self.quad_pred(box_tower).sigmoid()*512)

            # quad_pred = self.scales[l](self.quad_pred(box_tower))
            # if self.norm_reg_targets:
            #     quad_pred = F.relu(quad_pred)
            #     if self.training:
            #         quad_reg.append(quad_pred)
            #     else:
            #         quad_reg.append(quad_pred * self.fpn_strides[l])
            # else:
            #     quad_reg.append(torch.exp(quad_pred))
        return logits, quad_reg


class EASTModule(torch.nn.Module):
    """
    Module for EAST computation. Takes feature maps from the backbone and
    EAST outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(EASTModule, self).__init__()

        head = EASTHead(cfg, in_channels)

        poly_selector_test = make_east_postprocessor(cfg)

        loss_evaluator = make_east_loss_evaluator(cfg)
        self.head = head
        self.poly_selector_test = poly_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.EAST.FPN_STRIDES
        self.visualizater = Visualizater()
    def visulizate(self, images, masks, centerness=None):
        images_tensor = images.tensors.clone()
        # print(images_tensor)
        images_pil = self.visualizater.conver_images_to_pil(images_tensor)
        masks_pil = self.visualizater.convert_masks_to_pil(masks.sigmoid())
        images_masks = self.visualizater.render_masks_to_images(images_pil.copy(), masks_pil)
        shows = self.visualizater.cat_images(zip(images_pil, images_masks), shape=[1,-1])
        self.visualizater.save(shows, folder='vis', names=None)
    def forward(self, images, features, targets=None):
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
        quad_cls, quad_regression = self.head(features)
        # print(quad_cls[0].shape, quad_regression[0].shape)
        # self.visulizate(images, quad_cls[0])
        # print(quad_cls[0].shape, quad_regression[0].shape, centerness[0].shape)
        # locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                quad_cls, quad_regression, targets
            )
        else:
            self.visulizate(images, quad_cls[0])
            return self._forward_test(
                quad_cls, quad_regression, images.image_sizes, targets
            )

    def _forward_train(self, quad_cls, quad_regression, targets):
        # return None, None
        loss_cls, loss_quad_reg = self.loss_evaluator(
            quad_cls, quad_regression, targets
        )
        losses = {
            "loss_cls": loss_cls,
            "loss_reg": loss_quad_reg,
            # "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, box_cls, box_regression, image_sizes, targets=None):
        boxes = self.poly_selector_test(box_cls, box_regression, image_sizes, targets=targets)
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
        # print(w*stride)
        stride = 4
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
        # locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        locations = torch.stack((shift_x, shift_y), dim=1)
        return locations

def build_east(cfg, in_channels):
    return EASTModule(cfg, in_channels)

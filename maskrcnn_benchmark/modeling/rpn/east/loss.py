"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor
def dice_loss(pred_pss, true_pss, training_mask):
    eps = 1e-5
    training_mask = training_mask.type_as(pred_pss)
    pred_pss = pred_pss.view_as(true_pss)
    intersection = torch.sum(true_pss * pred_pss * training_mask, (3,2,1))
    union = torch.sum(true_pss * training_mask, (3,2,1)) + torch.sum(pred_pss * training_mask, (3,2,1)) + eps
    loss = 1. - (2 * intersection/union)
    loss = torch.mean(loss)
    return loss
def detect_quad_loss(pred_pss, pred_quad, true_pss, true_quad, norm, training_mask):
    positive_mask = true_pss.type_as(training_mask) * training_mask
    geo_loss = quad_loss(pred_quad,true_quad,norm,positive_mask)
    pss_loss = dice_loss(pred_pss, true_pss, training_mask)
    # geo_loss *= 30
    # total_loss = geo_loss + pss_loss
    return geo_loss, pss_loss
def quad_loss(pred_quad,true_quad,norm,pss_map):
    b,c,h,w = pred_quad.shape
    norm = norm + 1
    loss = torch.nn.functional.smooth_l1_loss(pred_quad, true_quad, reduction ='none')/norm
    loss = loss.mean(dim=1).view((b,1,h,w))
    loss = loss * pss_map
    pss_map = pss_map.view((b, -1))
    loss = loss.view((b, -1))
    mean_loss = torch.sum(loss, dim=1)/(torch.sum(pss_map, dim=1)+1)
    return torch.mean(mean_loss)
def iou_loss(pred_geo, true_geo):
    pred_t, pred_r, pred_b, pred_l = torch.split(pred_geo,1,dim=1)
    true_t, true_r, true_b, true_l = torch.split(true_geo,1,dim=1)

    pred_area = (pred_t + pred_b) * (pred_l + pred_r)
    true_area = (true_t + true_b) * (true_l + true_r)

    min_h = torch.min(pred_t, true_t) + torch.min(pred_b, true_b)
    min_w = torch.min(pred_l, true_l) + torch.min(pred_r, true_r)
    insection = min_h * min_w
    union = pred_area + true_area - insection
    iou = (insection + 1.0)/(union + 1.0)

    loss = -torch.log(iou)

    return loss
def detect_box_loss(pred_pss, pred_quad, true_pss, true_quad, training_mask):
    positive_mask = true_pss.type_as(training_mask) * training_mask
    geo_loss = iou_loss(pred_quad,true_quad)
    geo_loss = torch.sum(geo_loss*positive_mask, (3,2,1))/(torch.sum(positive_mask, (3,2,1))+1e-3)
    geo_loss = torch.mean(geo_loss)
    pss_loss = dice_loss(pred_pss, true_pss, training_mask)
    return geo_loss, pss_loss
class EASTLossComputation(object):
    """
    This class computes the EAST losses.
    """

    def __init__(self, cfg):
        # self.cls_loss_func = SigmoidFocalLoss(
        #     cfg.MODEL.EAST.LOSS_GAMMA,
        #     cfg.MODEL.EAST.LOSS_ALPHA
        # )
        # self.fpn_strides = cfg.MODEL.EAST.FPN_STRIDES
        # self.center_sampling_radius = cfg.MODEL.EAST.CENTER_SAMPLING_RADIUS
        # self.iou_loss_type = cfg.MODEL.EAST.IOU_LOSS_TYPE
        # self.norm_reg_targets = cfg.MODEL.EAST.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        # self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def __call__(self, box_cls, quad_regression, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        all_pss_gt, all_trbl_gt, all_mask_gt = [], [], []
        for target in targets:
            pss_maps,trbl,training_mask = target.generate_quad_gt()
            all_pss_gt.append(torch.tensor(pss_maps).type_as(box_cls[0]))
            # all_center_gt.append(torch.tensor(centerness).type_as(box_cls[0]))
            all_trbl_gt.append(torch.tensor(trbl).type_as(box_cls[0]))
            # all_norm_gt.append(torch.tensor(norm).type_as(box_cls[0]))
            all_mask_gt.append(torch.tensor(training_mask).type_as(box_cls[0]))
        all_pss_gt = torch.stack(all_pss_gt)
        # all_center_gt = torch.stack(all_center_gt)
        all_trbl_gt = torch.stack(all_trbl_gt)
        # all_norm_gt = torch.stack(all_norm_gt)
        all_mask_gt = torch.stack(all_mask_gt)

        # pss = box_cls[0].sigmoid()[:,::4,::4]
        # print(box_cls[0].shape, quad_regression[0].shape, all_pss_gt.shape, all_trbl_gt.shape, all_mask_gt.shape)
        geo_loss, pss_loss = detect_box_loss(box_cls[0], quad_regression[0], all_pss_gt, all_trbl_gt, all_mask_gt)
        # if torch.isnan(quad_loss):
        #     quad_loss = quad_loss * 0
        # if torch.isnan(cls_loss):
        #     cls_loss = cls_loss * 0
        # if torch.isnan(centerness_loss):
        #     centerness_loss = centerness_loss * 0
        # print(quad_loss.shape, cls_loss.shape,centerness_loss.shape)
        return pss_loss, geo_loss


def make_east_loss_evaluator(cfg):
    loss_evaluator = EASTLossComputation(cfg)
    return loss_evaluator

import torch
from torch import nn
from torch.nn import functional as F
import cv2
from .retrieval_pixel import SiamHead
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.rpn.fcos.fcos import build_fcos
from maskrcnn_benchmark.modeling.rpn.east.east import build_east
from maskrcnn_benchmark.modeling.rpn.fast_center.fast_center import build_fast_center

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


class SiamModule(torch.nn.Module):
    """
    Module for BezierAlign computation. Takes feature maps from the backbone and
    BezierAlign outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(SiamModule, self).__init__()

        self.cfg = cfg.clone()
        self.head = SiamHead(cfg, in_channels)
        self.detector = build_fast_center(cfg, in_channels)
        self.visualizator = Visualizator()
        # self.proposal_matcher = proposal_matcher
        # self.scales = cfg.MODEL.ALIGN.POOLER_SCALES
        # self.use_box_aug = cfg.MODEL.ALIGN.USE_BOX_AUG

    def test_visual(self,images,boxes):
        image_tensor = images.tensors.permute(0,2,3,1).float()
        print(image_tensor.shape)
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)[0]
        boxes = boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        cv2.drawContours(image_de, boxes, -1, color=(255,0,0), thickness=1)
        img_path = os.path.join('temp','img_{}.jpg'.format(np.random.randint(0,999)))
        cv2.imwrite(img_path, image_de)
        return None


    def forward(self, images, features, targets=None, vis=False):
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
            boxes, losses = self.detector(images, features[1:], targets)
            # _, loss_dict = self.head(features, targets,images)

            # for k, v in loss_dict.items():
            #     losses.update({k: v})
            return None, losses
        else:
            boxes, info = self.detector(images, features[1:], targets)
            # self.test_visual(images, boxes[0].bbox)
            # image = self.visualizator.convert_image_cuda_numpy(images)
            # image_boxes = self.visualizator.visual_boxes(image.copy(), boxes[0].bbox)
            # image_cls = self.visualizator.visual_cls(image.copy(), info['box_cls'])
            # image_center = self.visualizator.visual_cls(image.copy(), info['centerness'])
            # image_similarity = self.visualizator.visual_similarity(image.copy(), info['box_cls'], info['targets'][0].get_field("similarity_pred"))
            # image_cls.extend(image_similarity)
            # img_save = self.visualizator.vis_multi_image([Image.fromarray(image[:,:,(2,1,0)]) for image in image_cls],shape=[2,-1])
            # self.visualizator.save(img_save)

            results = info['targets']
            # new_results = []
            for idx, result in enumerate(results):
                image_embedding_per_img = result.get_field("images_embedding_nor")
                box_cls = torch.cat([per_level[idx].view(-1) for per_level in info['box_cls']])
                pos_ids = torch.nonzero(box_cls.sigmoid() > 0.5).squeeze(1)
                result.add_field("images_embedding_nor", image_embedding_per_img[:,pos_ids])
            return results, {}

@registry.ONE_STAGE_HEADS.register("siam")
def build_siam_head(cfg, in_channels):
    return SiamModule(cfg, in_channels)

class Visualizator(object):
    def __init__(self):
        print("visualizator")
    def draw_heatmap(self,image,heatmap,alpha = 0.5):
        overlay = image.copy()
        # heatmap = cv2.cvtColor(np.asarray(heatmap),cv2.COLOR_RGB2BGR)
        # cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 0), -1)
        # print(heatmap.shape, image.shape)
        # image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
        # print(heatmap, image)
        image = cv2.addWeighted(heatmap, alpha, image, 1-alpha, 0)
        return image
    def convert_image_cuda_numpy(self,image):
        image_tensor = image.tensors.permute(0,2,3,1).float()
        image_de = denormalize(image_tensor).data.cpu().numpy().astype(np.uint8)[0]
        return image_de
    def visual_boxes(self,image,boxes):
        boxes = boxes.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
        cv2.drawContours(image, boxes, -1, color=(255,0,0), thickness=1)
        return image
    def visual_cls(self,image,classifications):
        h,w,_ = image.shape
        images = []
        for cls_per_level in classifications[:-2]:
            cls_per_level = F.interpolate(cls_per_level.sigmoid(), size=(h,w)).data.cpu().numpy()[0,0,:,:]
            cls_per_level =  np.repeat((cls_per_level*255).astype(np.uint8)[:,:,None], 3, axis=-1)
            images.append(self.draw_heatmap(image.copy(),cls_per_level))
        return images
    def visual_similarity(self,image, classifications, similarity):
        h,w,_ = image.shape
        images = []
        sizes = [[cls_per_level.size(2),cls_per_level.size(3)] for cls_per_level in classifications]
        nums = [size[0]*size[1] for size in sizes]
        similaritys = torch.split(similarity, nums, dim=1)
        for idx, sim_per_level in enumerate(similaritys[:-2]):
            # print(sim_per_level.shape,sizes[idx])
            # print(sim_per_level.max(), sim_per_level.min())
            sim_per_level = sim_per_level.max(dim=0)[0].reshape([1,1,sizes[idx][0],sizes[idx][1]])
            
            sim_per_level = F.interpolate(sim_per_level, size=(h,w)).data.cpu().numpy()[0,0,:,:]
            cls_per_level = F.interpolate(classifications[idx].sigmoid(), size=(h,w)).data.cpu().numpy()[0,0,:,:]
            sim_per_level = (sim_per_level * cls_per_level)**0.5
            sim_per_level =  np.repeat((sim_per_level*255).astype(np.uint8)[:,:,None], 3, axis=-1)
            images.append(self.draw_heatmap(image.copy(),sim_per_level))
        return images
    

    def vis_multi_image(self, image_list, shape=[1,-1]):
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
    def save(self, img):
        img_path = os.path.join('temp','img_{}.jpg'.format(np.random.randint(0,999)))
        img.save(img_path)
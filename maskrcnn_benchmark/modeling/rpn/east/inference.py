import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
# from atss_core.structures.image_instance import ImageInstance
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, remove_small_boxes, boxlist_nms
# from atss_core.structures.imageinstance_ops import cat_imageinstance
# from atss_core.structures.imageinstance_ops import imageinstance_ml_nms
# from atss_core.structures.imageinstance_ops import remove_small_instances


class EASTPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(EASTPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
    def restore_box(self, quad, size, scale):
        h,w = size
        scale_factor_h, scale_factor_w = scale
        
        x_axis = torch.arange(w, dtype=torch.float32, device=quad.device).view(1,w).repeat(h,1)*scale_factor_w
        y_axis = torch.arange(h, dtype=torch.float32, device=quad.device).view(h,1).repeat(1,w)*scale_factor_h
        # pred_t, pred_r, pred_b, pred_l = torch.split(pred_geo,1,dim=1)
        quad[:,0,:,:] = y_axis - quad[:,0,:,:] #
        quad[:,1,:,:] = x_axis + quad[:,1,:,:]
        quad[:,2,:,:] = y_axis + quad[:,2,:,:] #
        quad[:,3,:,:] = x_axis - quad[:,3,:,:]
        return quad[:,(3,0,1,2),:,:]
    def forward_for_single_feature_map(
            self, quad_cls,
            quad_regression,
            image_sizes):
        """
        Arguments:
            anchors: list[quadList]
            quad_cls: tensor of size N, A * C, H, W
            quad_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = quad_cls.shape
        # print(locations)
        # put in the same format as locations
        boxes = self.restore_box(quad_regression, (H,W), scale=(4,4))
        quad_cls = quad_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        quad_cls = quad_cls.reshape(N, -1, C)
        boxes = boxes.view(N, 4, H, W).permute(0, 2, 3, 1)
        boxes = boxes.reshape(N, -1, 4)
        # centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # centerness = centerness.reshape(N, -1).sigmoid()
        # print(quad_cls.shape, self.pre_nms_thresh)
        candidate_inds = quad_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        # print(pre_nms_top_n, self.pre_nms_top_n)
        # multiply the classification scores with centerness scores
        # quad_cls = quad_cls * centerness[:, :, None]
        
        results = []
        for i in range(N):
            per_quad_cls = quad_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_quad_cls = per_quad_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_quad_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_boxes = boxes[i]
            per_boxes = per_boxes[per_quad_loc]
            # per_locations = locations[per_quad_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]
            # print(per_candidate_inds.sum().item(), per_pre_nms_top_n.item())
            if per_candidate_inds.sum().item() >= per_pre_nms_top_n.item():
                per_quad_cls, top_k_indices = \
                    per_quad_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_boxes = per_boxes[top_k_indices]
                
            # print(per_boxes.shape)
            # boxes = torch.stack([detections[:,::2].min(dim=1)[0], detections[:,1::2].min(dim=1)[0], detections[:,::2].max(dim=1)[0], detections[:,1::2].max(dim=1)[0]],dim=1)
            # print(boxes.shape)

            h, w = image_sizes[i]
            imageinstance = BoxList(per_boxes, (int(w), int(h)), mode="xyxy")
            imageinstance.add_field("labels", per_class)
            imageinstance.add_field("scores", per_quad_cls)
            # imageinstance.add_field("polys", detections)
            # imageinstance = imageinstance.clip_to_image(remove_empty=False)
            imageinstance = remove_small_boxes(imageinstance, self.min_size)
            results.append(imageinstance)

        return results

    def forward(self, box_cls, box_regression, image_sizes, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for _, (o, b) in enumerate(zip(box_cls, box_regression)):
            # target = targets[_]
            # pss_maps,trbl,training_mask = target.generate_quad_gt()
            # pss_maps = torch.tensor(pss_maps).type_as(box_cls[0])
            # trbl = torch.tensor(trbl).type_as(box_cls[0])
            sampled_boxes.append(self.forward_for_single_feature_map(o, b, image_sizes))
            # sampled_boxes.append(self.forward_for_single_feature_map(pss_maps[None,:,:,:], trbl[None,:,:,:], image_sizes))

        imageinstances = list(zip(*sampled_boxes))
        imageinstances = [cat_boxlist(imageinstance) for imageinstance in imageinstances]
        # if not self.bbox_aug_enabled:
        #     imageinstances = self.select_over_all_levels(imageinstances)
        imageinstances = [boxlist_nms(image_instance, 0.4) for image_instance in imageinstances]

        return imageinstances

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    # def select_over_all_levels(self, boxlists):
    #     num_images = len(boxlists)
    #     results = []
    #     for i in range(num_images):
    #         # multiclass nms
    #         result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
    #         number_of_detections = len(result)

    #         # Limit to max_per_image detections **over all classes**
    #         if number_of_detections > self.fpn_post_nms_top_n > 0:
    #             cls_scores = result.get_field("scores")
    #             image_thresh, _ = torch.kthvalue(
    #                 cls_scores.cpu(),
    #                 number_of_detections - self.fpn_post_nms_top_n + 1
    #             )
    #             keep = cls_scores >= image_thresh.item()
    #             keep = torch.nonzero(keep).squeeze(1)
    #             result = result[keep]
    #         results.append(result)
    #     return results


def make_east_postprocessor(config):
    pre_nms_thresh = config.MODEL.EAST.INFERENCE_TH
    pre_nms_top_n = config.MODEL.EAST.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.EAST.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = EASTPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.EAST.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector

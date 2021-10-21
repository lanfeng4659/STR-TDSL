# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
ROTATE_90 = 2
import cv2
# from geo_map_cython_lib import gen_geo_map
from maskrcnn_benchmark.utils.text_util import TextGenerator
import pyclipper
import Polygon as plg
import numpy as np
def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri
def shrink(bbox, rate, max_shr=20):
    rate = rate * rate
    area = plg.Polygon(bbox).area()
    peri = perimeter(bbox)

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
    
    shrinked_bbox = pco.Execute(-offset)
    if len(shrinked_bbox) == 0:
        return bbox
    
    shrinked_bbox = np.array(shrinked_bbox[0])
    if shrinked_bbox.shape[0] <= 2:
        shrinked_bbox = bbox
    return shrinked_bbox
def get_ordered_polys(cnt):
    points = list(cnt)
    ps = sorted(points,key = lambda x:x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0]
        py1 = ps[0][1]
        px4 = ps[1][0]
        py4 = ps[1][1]
    else:
        px1 = ps[1][0]
        py1 = ps[1][1]
        px4 = ps[0][0]
        py4 = ps[0][1]
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0]
        py2 = ps[2][1]
        px3 = ps[3][0]
        py3 = ps[3][1]
    else:
        px2 = ps[3][0]
        py2 = ps[3][1]
        px3 = ps[2][0]
        py3 = ps[2][1]

    return np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]])

class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
        self.text_generator = TextGenerator()

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def pop_field(self, field):
        return self.extra_fields.pop(field)

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v
    def clone_except_fields(self,fields=[]):
        bbox = BoxList(self.bbox, self.size, self.mode)
        # print(item)
        # item = item.data.cpu().numpy()
        for k, v in self.extra_fields.items():
            if k in fields:
                continue
            bbox.add_field(k, v)
        return bbox
    def clone_without_fields(self):
        bbox = BoxList(self.bbox, self.size, self.mode)
        return bbox

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor) and k != 'rles' and not isinstance(v,np.ndarray):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox.add_field("ratios",ratios)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and k != 'rles' and not isinstance(v,np.ndarray):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def pad(self, new_size):
        bbox = BoxList(self.bbox, new_size, mode=self.mode)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and not isinstance(v,np.ndarray):
                v = v.pad(new_size)
            bbox.add_field(k, v)

        return bbox

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        elif method == ROTATE_90:
            transposed_xmin = ymin
            transposed_xmax = ymax
            transposed_ymin = image_width - xmax
            transposed_ymax = image_width - xmin

            self.size=(image_height, image_width)

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and not isinstance(v,np.ndarray) and not isinstance(v,tuple) and v!=None:
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # def crop(self, box, remove_empty=False):
    #     """
    #     Crops a rectangular region from this bounding box. The box is a
    #     4-tuple defining the left, upper, right, and lower pixel
    #     coordinate.
    #     """
    #     xmin, ymin, xmax, ymax = self._split_into_xyxy()
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
    #     cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
    #     cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
    #     cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

    #     # TODO should I filter empty boxes here?
    #     if False:
    #         is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

    #     cropped_box = torch.cat(
    #         (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
    #     )
    #     bbox = BoxList(cropped_box, (w, h), mode="xyxy")
    #     # bbox._copy_extra_fields(self)
    #     for k, v in self.extra_fields.items():
    #         if not isinstance(v, torch.Tensor):
    #             v = v.crop(box)
    #         bbox.add_field(k, v)

    #     if remove_empty:
    #         box = bbox.bbox
    #         keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
    #         bbox = bbox[keep]
    #     return bbox.convert(self.mode)
    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if True:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)
            not_empty=torch.nonzero(is_empty.view(-1)==0).view(-1)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and not isinstance(v,np.ndarray):
                v = v.crop(box)
            bbox.add_field(k, v)
        
        # # label empty as ignore
        # is_hard = (cropped_xmin==0)|(cropped_xmax==w)|(cropped_ymin==0)|(cropped_ymax==h)
        # is_hard_idx =torch.nonzero(is_hard.view(-1)==1).view(-1)
        # hard=bbox.get_field('difficult')
        # hard[is_hard_idx]=1
        # bbox.add_field('difficult',hard)
        # filter empty
        # print(not_empty,bbox,bbox.get_field("texts"))
        bbox=bbox[not_empty]
        # print(not_empty,bbox,bbox.get_field("texts"))
        return bbox.convert(self.mode)
    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        # print(item)
        # item = item.data.cpu().numpy()
        for k, v in self.extra_fields.items():
            # print(k,len(v),item)
            # value = [v[item]] if len(item)==1 else v[item]
            # bbox.add_field(k, [v[i] for i in item])
            if isinstance(v,np.ndarray):
                # print(v)
                # if len(v)==0:
                #     print(v)
                bbox.add_field(k, np.array([v[i] for i in item]))
            else:
                bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            keep = torch.nonzero(keep).view(-1)
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
    def generate_quad_gt(self, min_text_size=40,difficult_label='###'):
        w,h= self.size
        text_boxes, text_polys, text_tags = self.bbox.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]), self.get_field("polys").data.cpu().numpy(), self.get_field("texts")
        # text_polys, text_tags = self.bbox.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]), self.get_field("texts") #check_and_validate_polys(boxes, tags, (h,w))
        idx_maps = np.zeros((h//4, w//4), dtype=np.uint8)
        pss_maps = np.zeros((1,h//4, w//4), dtype=np.uint8)
        geo_maps = np.zeros((h//4, w//4, 4), dtype=np.float32)
        # centerness = np.zeros((h, w, 1), dtype=np.float32)
        training_mask = np.ones((1,h//4,w//4), dtype=np.uint8)
        areas = []

        if len(text_polys) > 0:
            for poly_idx, poly_tag in enumerate(zip(text_boxes,text_polys, text_tags)):
                # print(poly_tag)
                box = poly_tag[0].reshape([-1,2])/4
                poly = poly_tag[1].reshape([-1,2])/4
                text  = poly_tag[2]
                if cv2.arcLength(poly.astype(np.int32),True) < min_text_size or text==difficult_label:
                    cv2.fillPoly(training_mask[0], poly.reshape([1,-1,2]).astype(np.int32), 0)
                else:
                    cv2.fillPoly(training_mask[0], poly.reshape([1,-1,2]).astype(np.int32), 0)
                    shrinked_poly = shrink(poly.astype(np.int32),0.3)
                    cv2.fillPoly(idx_maps, shrinked_poly.reshape([1,-1,2]).astype(np.int32), poly_idx+1)
                    cv2.fillPoly(training_mask[0], shrinked_poly.reshape([1,-1,2]).astype(np.int32), 1)
                    cv2.fillPoly(pss_maps[0], shrinked_poly.reshape([1,-1,2]).astype(np.int32), 1)

                    xy_in_poly = np.argwhere(idx_maps == (poly_idx + 1))
                    
                #   print(order_poly)
                    
                    order_poly = get_ordered_polys(poly)
                    gen_geo_map.gen_trbl_map(geo_maps, xy_in_poly, box)
                    # gen_geo_map.centerness(centerness, xy_in_poly,order_poly.reshape([-1,2]).astype(np.float32))
        # out = np.split(geo_maps,[8,9],axis=2)
        # print(out[0].shape, out[1].shape)
        # quad = out[0].transpose((2,0,1))
        # norm = out[1].transpose((2,0,1))
        # centerness = centerness.transpose((2,0,1))
        ltrb = geo_maps.transpose((2,0,1))*4
        return pss_maps,ltrb,training_mask#, text_polys, text_tags
    def generate_det_retrieval_gt(self, min_text_size=40,difficult_label='###'):
        w,h= self.size
        text_boxes, text_polys, text_tags = self.bbox.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]),\
                                             self.get_field("polys").data.cpu().numpy(), self.get_field("texts")
        idxs, valid_texts = self.text_generator.filter_words(text_tags.tolist())
        self.add_field("valid_texts", np.array(valid_texts))
        # text_polys, text_tags = self.bbox.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]), self.get_field("texts") #check_and_validate_polys(boxes, tags, (h,w))
        idx_maps = np.zeros((h//4, w//4), dtype=np.uint8)
        pss_maps = np.zeros((1,h//4, w//4), dtype=np.uint8)
        geo_maps = np.zeros((h//4, w//4, 4), dtype=np.float32)
        similarity = np.zeros((h//4, w//4, len(idxs)), dtype=np.float32)
        training_mask = np.ones((1,h//4,w//4), dtype=np.uint8)
        areas = []

        if len(text_polys) > 0:
            for poly_idx, poly_tag in enumerate(zip(text_boxes,text_polys, text_tags)):
                # print(poly_tag)
                box = poly_tag[0].reshape([-1,2])/4
                poly = poly_tag[1].reshape([-1,2])/4
                text  = poly_tag[2]
                if cv2.arcLength(poly.astype(np.int32),True) < min_text_size or text==difficult_label or poly_idx not in idxs:
                    cv2.fillPoly(training_mask[0], poly.reshape([1,-1,2]).astype(np.int32), 0)
                else:
                    cv2.fillPoly(training_mask[0], poly.reshape([1,-1,2]).astype(np.int32), 0)
                    shrinked_poly = shrink(poly.astype(np.int32),0.3)
                    cv2.fillPoly(idx_maps, shrinked_poly.reshape([1,-1,2]).astype(np.int32), poly_idx+1)
                    cv2.fillPoly(training_mask[0], shrinked_poly.reshape([1,-1,2]).astype(np.int32), 1)
                    cv2.fillPoly(pss_maps[0], shrinked_poly.reshape([1,-1,2]).astype(np.int32), 1)

                    xy_in_poly = np.argwhere(idx_maps == (poly_idx + 1))
                    
                    order_poly = get_ordered_polys(poly)
                    gen_geo_map.gen_trbl_map(geo_maps, xy_in_poly, box)
        for i, idx1 in enumerate(idxs):
            for idx2 in idxs:
                ew = self.text_generator.editdistance(text_tags[idx1], text_tags[idx2])
                fill_mask = (idx_maps==(idx2+1)).astype(np.float32)
                similarity[:,:,i] = similarity[:,:,i]*(1-fill_mask) + fill_mask*ew
        self.add_field("distances", similarity.transpose((2,0,1)))
        ltrb = geo_maps.transpose((2,0,1))*4
        return pss_maps,ltrb,training_mask#, text_polys, text_tags



if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)

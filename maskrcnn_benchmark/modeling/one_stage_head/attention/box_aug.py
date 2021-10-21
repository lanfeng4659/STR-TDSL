import numpy as np
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
# def boxlist_iou(boxlist1, boxlist2):
#     area1 = (boxlist1[:,3] - boxlist1[:,1])*(boxlist1[:,2] - boxlist1[:,0]).clamp(min=0)
#     area2 = (boxlist2[:,3] - boxlist2[:,1])*(boxlist2[:,2] - boxlist2[:,0]).clamp(min=0)


#     lt = torch.max(boxlist1[:, None, :2], boxlist2[:, :2])  # [N,M,2]
#     rb = torch.min(boxlist1[:, None, 2:], boxlist2[:, 2:])  # [N,M,2]

#     TO_REMOVE = 1

#     wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

#     iou = inter / (area1[:, None] + area2 - inter).clamp(min=1)
#     return iou.clamp(min=0,max=1)
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
class Shift(object):
    def __init__(self, max_ratio=0.3):
        print("Shift on X or Y")
        self.max_ratio = max_ratio
    def get_ratio(self,num):
        return np.random.randint(-100,100, num)/100*self.max_ratio
    def __call__(self,boxes):
        '''
        boxes: xyxy
        '''
        new_boxes = boxes.clone()
        r = self.get_ratio(boxes.size(0))
        r = torch.tensor(r).type_as(boxes).reshape([boxes.size(0), 1])
        if np.random.randint(0,2)==0:
            '''
            shift_x
            '''
            new_boxes[:,::2] = boxes[:,::2] + r * (boxes[:,2] - boxes[:,0]).view(-1,1)
        else:
            new_boxes[:,1::2] = boxes[:,1::2] + r * (boxes[:,3] - boxes[:,1]).view(-1,1)
        return new_boxes
class Scale(object):
    def __init__(self, max_ratio=0.3):
        print("Scale on X or Y")
        self.max_ratio = max_ratio
    def get_ratio(self,num):
        return 1+np.random.randint(-100,100, num)/100*self.max_ratio
    def __call__(self,boxes):
        '''
        boxes: xyxy
        '''
        new_boxes = boxes.clone()
        r = self.get_ratio(boxes.size(0))
        r = torch.tensor(r).type_as(boxes)
        if np.random.randint(0,2)==0:
            '''
            shift_x
            '''
            new_boxes[:,0] = 0.5 * (boxes[:,2] + boxes[:,0]) - 0.5 * r * (boxes[:,2] - boxes[:,0])
            new_boxes[:,2] = 0.5 * (boxes[:,2] + boxes[:,0]) + 0.5 * r * (boxes[:,2] - boxes[:,0])
        else:
            new_boxes[:,1] = 0.5 * (boxes[:,3] + boxes[:,1]) - 0.5 * r * (boxes[:,3] - boxes[:,1])
            new_boxes[:,3] = 0.5 * (boxes[:,3] + boxes[:,1]) + 0.5 * r * (boxes[:,3] - boxes[:,1])
        return new_boxes
class BoxAug(object):
    def __init__(self,augs):
        self.augs = augs
    def __call__(self, sample,repeats=3):
        boxes_ori = sample.bbox.clone()
        boxes = sample.bbox.clone()
        all_boxes = [boxes]
        # for t in self.augs:
        #     boxes = t(boxes)
        for i in range(repeats):
            t = np.random.choice(self.augs)
            all_boxes.append(t(boxes))
        all_boxes = torch.cat(all_boxes)
        # print(all_boxes.shape,boxes.shape)
        
        
        boxlist1 = BoxList(all_boxes, sample.size, mode=sample.mode)
        boxlist1 = boxlist1.clip_to_image(remove_empty=True)
        # iou_table = boxlist_iou(boxlist1, sample)
        # iou = iou_table.max(dim=1)[0]
        # boxlist1.add_field("iou",iou)
        
        return boxlist1
def make_box_aug():
    augs = BoxAug([Shift(),Scale()])
    return augs
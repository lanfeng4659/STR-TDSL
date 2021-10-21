import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from shapely.geometry import box, Polygon
import math
from .image_process import *
from scipy.misc import imread, imresize
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, boxes=None, tags=None):
        # print(type(boxes)=='list',type(boxes))
        poly_fixed_points = False if type(boxes).__name__=='list' else True
        for t in self.transforms:
            assert len(boxes)==len(tags),print(boxes,tags)
            img, boxes, tags = t(img, boxes, tags, poly_fixed_points)
        return img, boxes, tags

class Resize(object):
    def __init__(self, size=(640,640)):
        self.width = size[1]
        self.heigth = size[0]
    def __call__(self, image, boxes=None, tags=None,poly_fixed_points=True):
        ori_h, ori_w, _ = image.shape

        new_image = imresize(image.copy(), (self.heigth,self.width))
        if boxes is not None:
            boxes[:,:,0] *= self.width*1.0/ori_w
            boxes[:,:,1] *= self.heigth*1.0/ori_h
            boxes[:,:,0] = np.clip(boxes[:,:,0],0,self.width)
            boxes[:,:,1] = np.clip(boxes[:,:,1],0,self.heigth)
        return new_image, boxes, tags
class RandomResize(object):
    def __init__(self, longer_sides=np.arange(640,2592, 32)):
        self.longer_sides = longer_sides
    def __call__(self, image, boxes = None, tags = None,poly_fixed_points=True):
        return random_resize(image, boxes, tags,poly_fixed_points, self.longer_sides)
class RandomRotate(object):
    def __init__(self, rotate_angles=np.arange(-15,15,1)):
        self.rotate_angles = rotate_angles
    def __call__(self, image, boxes = None, tags = None,poly_fixed_points=True):
        return random_rotate(image, boxes, tags,poly_fixed_points, self.rotate_angles)
class RandomRatioScale(object):
    def __init__(self, random_ratios = np.arange(0.8,1.3,0.1)):
        self.random_ratios = random_ratios
    def __call__(self, image, boxes, tags,poly_fixed_points=True):
        return random_ratio_scale(image, boxes, tags,poly_fixed_points, self.random_ratios)
class RandomCrop(object):
    def __init__(self, crop_size=(640,640), max_tries=10):
        self.crop_size = crop_size
        self.max_tries = max_tries
    def __call__(self, image, boxes, tags,poly_fixed_points=True):
        return random_crop(image, boxes, tags,poly_fixed_points, self.crop_size, self.max_tries)
class RandomRotate90(object):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
    def __call__(self, image, boxes, tags,poly_fixed_points=True):
        if random.random() > self.ratio:
            return image, boxes, tags
        h, w, _ = image.shape
        image = np.rot90(image)
        new_boxes = np.zeros_like(boxes)
        for i, box in enumerate(boxes):
            new_boxes[i] = abs(box - [w,0])
        new_boxes = new_boxes[:,(1,2,3,0),:][:,:,(1,0)]
        return image, new_boxes, tags
# longer_sides=np.arange(640,2592, 32)
# longer_sides=np.arange(512,1536, 32)
class PSSAugmentation(object):
    def __init__(self,longer_side_arange=np.arange(640,1920, 32)):
        self.augment = Compose([
        RandomResize(longer_sides=np.arange(640,1920, 32)),
        RandomRotate(rotate_angles=np.arange(-15,15,1)),
        RandomRatioScale(),
        RandomCrop(crop_size=(640,640), max_tries=10)
        ])
        self.no_crop = Compose([
        RandomRotate(rotate_angles=np.arange(-15,15,1)),
        Resize((640,640))
        ])

    def __call__(self, img, boxes, tags, no_crop=False):
        if no_crop:
            return self.no_crop(img, boxes, tags)
        else:
            return self.augment(img, boxes, tags)
class CTWAugmentation(object):
    def __init__(self,longer_side_arange=np.arange(640,1920, 32)):
        self.augment = Compose([
        # RandomResize(longer_sides=np.arange(640,1920, 32)),
        RandomRotate(rotate_angles=np.arange(-10,10,1)),
        RandomRatioScale(),
        RandomCrop(crop_size=(640,640), max_tries=20)
        ])
        self.no_crop = Compose([
        RandomRotate(rotate_angles=np.arange(-10,10,1)),
        Resize((640,640))
        ])

    def __call__(self, img, boxes, tags, no_crop=False):
        if no_crop:
            return self.no_crop(img, boxes, tags)
        else:
            return self.augment(img, boxes, tags)
class SythAugmentation(object):
    def __init__(self):
        self.augment = Compose([
        Resize((640,640))
        ])

    def __call__(self, img, boxes, tags):
        return self.augment(img, boxes, tags)
class RetrievalAugmentation(object):
    def __init__(self):
        self.augment = Compose([
        RandomRotate(rotate_angles=np.arange(-15,15,1)),
        Resize((512,512))
        ])

    def __call__(self, img, boxes, tags):
        return self.augment(img, boxes, tags)
class TestAugmentation(object):
    def __init__(self,longer_side=1280):
        self.longer_side = longer_side
    def __call__(self, image, boxes=None, tags=None):
        ori_h, ori_w, _ = image.shape
        if ori_h > ori_w:
            h = self.longer_side
            w = int(ori_w*1.0/ori_h*h)
        else:
            w = self.longer_side
            h = int(ori_h*1.0/ori_w*w)
        pad_h = h if h%32==0 else (h//32+1)*32
        pad_w = w if w%32==0 else (w//32+1)*32
        # new_image = np.zeros([pad_h, pad_w,3])
        new_image = imresize(image.copy(), (pad_h,pad_w))
        if boxes is not None:
            boxes[:,:,0] *= pad_w*1.0/ori_w
            boxes[:,:,1] *= pad_h*1.0/ori_h
            boxes[:,:,0] = np.clip(boxes[:,:,0],0,pad_w)
            boxes[:,:,1] = np.clip(boxes[:,:,1],0,pad_h)
        return new_image,boxes,tags








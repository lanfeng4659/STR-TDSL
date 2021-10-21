import os
import functools
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import cv2
import sys
import json
import scipy.io as sio
import glob
import math
from coco_text_api import coco_text
from tqdm import tqdm
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
# from maskrcnn_benchmark.structures.segmentation_mask import Polygons

# from maskrcnn_benchmark.utils.rec_util import LabelMap
import string
class ClockwiseSortPoints(object):
    def __init__(self):
        self.reverse = False
    def __call__(self, pts):
        self.pts = np.array(pts).reshape([-1,2])
        self.gravity = self._gravity()
        outs = sorted(self.pts, key=functools.cmp_to_key(self.point_compare))
        outs = np.array(outs)
        # outs = sorted(outs, key=self.angle)
        if outs.shape[0]==4:
            outs = self.get_mini_boxes(outs)
        # print(outs)
        return outs
    def get_mini_boxes(self,points):
        #print(cnt)
        # bounding_box = cv2.minAreaRect(cnt)
        # points = cv2.boxPoints(bounding_box)
        points = list(points)
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
    def _gravity(self):
        pts = self.pts
        pts_num = pts.shape[0]
        area = 1.
        center_x = 0.
        center_y = 0.
        for i in range(pts_num):
            next_point_index = 0 if i == pts_num-1 else i+1
            area += (pts[i,0]*pts[next_point_index,1] - pts[next_point_index,0]*pts[i,1])/2
            center_x += (pts[i,0]*pts[next_point_index,1] - pts[next_point_index,0]*pts[i,1]) * (pts[i,0] + pts[next_point_index,0])
            center_y += (pts[i,0]*pts[next_point_index,1] - pts[next_point_index,0]*pts[i,1]) * (pts[i,1] + pts[next_point_index,1])
        center_x /= 6*area
        center_y /= 6*area
        return np.array([center_x, center_y])
    def angle(self, v1, v2=(-10,-10)):
        dx1 = v1[0] - self.gravity[0]
        dy1 = v1[1] - self.gravity[1]
        dx2 = v2[0]
        dy2 = v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180/math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180/math.pi)
        # print(angle2)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle
    def point_compare(self,a, b):
        center = self.gravity
        if a[0]>=0 and b[0]<0:
            return 1
        if a[0]==0 and b[0]==0:
            return a[1] > b[1]
        det = (a[0] - center[0]) * (b[1] - center[1]) - (b[0] - center[0]) * (a[1] - center[1])
        if det < 0:
            return 1
        if det > 0:
            return -1
        d1 = (a[0] - center[0]) * (a[0] - center[0]) + (a[1] - center[1]) * (a[1] - center[1])
        d2 = (b[0] - center[0]) * (b[0] - center[0]) + (b[1] - center[1]) * (b[1] - center[1])
        return -1
class COCOTextDateset(torch.utils.data.Dataset):
    def __init__(self, data_dir,use_difficult=False,transforms=None,is_train=True,character_set=None):
        self.transforms=transforms
        self.is_train=is_train
        self.max_lens=35
        # self.label_map = LabelMap(character_set=character_set)
        self._img_path = os.path.join(data_dir,"train2014")
        self.clock_wise_sort_points=ClockwiseSortPoints()
        # import ipdb; ipdb.set_trace()
        anno_file = os.path.join(data_dir,"cocotext.v2.json") #if is_train else os.path.join(data_dir,"COCO_Text.json")
        # anno_file = os.path.join(data_dir,"COCO_Text.json")
        self.ct = coco_text.COCO_Text(anno_file)
        # ct.info()
        # self.train_imgs = ct.getImgIds(imgIds=ct.train, 
        #                 catIds=[('legibility','legible'),('class','machine printed')])
        # self.validate_imgs = ct.getImgIds(imgIds=ct.val, 
        #                 catIds=[('legibility','legible'),('class','machine printed')])
        # self.train_anns = ct.getAnnIds(imgIds=ct.train, 
        #                     catIds=[('legibility','legible'),('class','machine printed')], 
        #                     areaRng=[0,200])
        # self.validate_anns = ct.getAnnIds(imgIds=ct.val, 
        #                     catIds=[('legibility','legible'),('class','machine printed')], 
        #                     areaRng=[0,200])
        # self.imgIds = ct.getImgIds(imgIds=ct.train, 
        #             catIds=[('legibility','legible')])
        # for img_id in self.imgIds:
        #     img_info = ct.loadImgs(img_id)
        #     annIds = ct.getAnnIds(imgIds=img_info[0]['id'])
        #     anns = ct.loadAnns(annIds)
        #     print(anns)
        # import ipdb; ipdb.set_trace()
        # if is_train:
        #     self.imgs = self.ct.getImgIds(imgIds=self.ct.train, 
        #                 catIds=[('legibility','legible'),('class','machine printed')])
        # else:
        #     self.imgs = self.ct.getImgIds(imgIds=self.ct.val, 
        #                 catIds=[('legibility','legible'),('class','machine printed')])
        if is_train:
            imgs = self.ct.getImgIds(imgIds=self.ct.train+self.ct.val,catIds=[])
            self.imgs = self.filter_text_images(imgs)
            print("original image number:{}, text image number:{}".format(len(imgs), len(self.imgs)))
        else:
            self.imgs = self.ct.getImgIds(imgIds=self.ct.val,catIds=[])
            print("original image number:{}".format(len(self.imgs)))
        
        

    def __getitem__(self, idx):
        img_id = self.imgs[idx]
        img_info = self.ct.loadImgs(img_id)[0]
        img_path = os.path.join(self._img_path,img_info['file_name'])
               
        # image = Image.open(img_path).convert("RGB")
        anno,pts=self.get_groundtruth(img_info)

        return anno,img_path

    def __len__(self):
        return len(self.imgs)
    def filter_text_images(self,imgs):
        new_imgs = []
        for idx in range(len(imgs)):
            img_id = imgs[idx]
            img_info = self.ct.loadImgs(img_id)[0]
            annIds = self.ct.getAnnIds(imgIds=img_info['id'])
            anns = self.ct.loadAnns(annIds)
            if len(anns) > 0:
                new_imgs.append(img_id)
        return new_imgs
    # def GetPts_total(self,polygt, idx):
    #     pts_lst = []
    #     if polygt[idx][5].shape[0] == 0:
    #         hard = 1
    #     else:
    #         hard = int(polygt[idx][5][0]=='#')
    #     for pts_num in range(polygt[idx][1].shape[1]):
    #         pts_lst.append([polygt[idx][1][0][pts_num],polygt[idx][3][0][pts_num]])
    #     pts = np.array(pts_lst, dtype=np.int32)
    #     return pts, hard
    def GetAnn_coco(self,ann):
        pts_lst = []
        pts = np.array(ann['mask'], dtype=np.float).reshape([-1,2])
        # print(ann)
        text = ann['utf8_string']
        detection_ignore = int(ann['legibility']=='illegible')
        recognition_ignore = int(ann['legibility']=='illegible' or ann['language']!='english' or text in ['',' '])
        # if recognition_ignore == 1:
        #     text=''
        # all recognition_ignore proposals will be filterred out owing to text_len is zero in roi_rec_feature_extractors.py
        return pts, text, detection_ignore, recognition_ignore

    def Gettext_total(self,polygt,idx):
        return polygt[idx][4]


    def get_groundtruth(self,img_info):
        annIds = self.ct.getAnnIds(imgIds=img_info['id'])
        anns = self.ct.loadAnns(annIds)
        # import ipdb; ipdb.set_trace()
        boxes=[]
        texts=[]
        polys=[]
        detection_ignores=[]
        recognition_ignores=[]
        
        text_labels = []
        text_lens = []
        pts_de=[]

        for ann in anns:
            pts,text,detection_ignore, recognition_ignore=self.GetAnn_coco(ann)
            pts = self.clock_wise_sort_points(pts)
            if recognition_ignore or detection_ignore:
                continue
            min_x,min_y,w,h = ann['bbox']
            box=np.array([min_x,min_y,min_x+w,min_y+h], dtype=np.int32)
            # print(box,pts)
            boxes.append(box)
            
            # polys.append(pts_expand.reshape(-1).tolist())
            detection_ignores.append(detection_ignore)
            recognition_ignores.append(recognition_ignore)

            texts.append(text)

        n_boxes = len(boxes)

        if len(boxes)>0:
            res={
                'boxes':boxes,
                'texts':texts,
                'detection_ignores':torch.tensor(detection_ignores),
                'recognition_ignores':torch.tensor(recognition_ignores)
            }
        else:
            res={
                'boxes':boxes,
                'texts':texts,
                'detection_ignores':torch.zeros([0], dtype=torch.int),
                'recognition_ignores':torch.zeros([0], dtype=torch.int),
            }
        return res,pts_de

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": 768, "width": 1280}
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
text_num = {}
from shutil import copyfile
if __name__ == '__main__':
    # print("hello")
    character_set=(string.digits + string.ascii_letters + string.punctuation)
    dataset = COCOTextDateset("/workspace/wanghao/datasets/coco_text", character_set=character_set)
    selected_images = os.listdir("datasets/cocotext_top500_retrieval/images")[:500]
    # label_map = LabelMap(character_set=character_set)
    imgname_words={}
    for i in range(len(dataset)):
        anno,img_path = dataset[i]
        texts = anno["texts"]
        boxes = anno["boxes"]
        imgname = os.path.basename(img_path)
        if imgname not in selected_images:
            continue
        copyfile(img_path, os.path.join("./datasets/cocotext_week_annotation_500/images",imgname))
        f = open(os.path.join("./datasets/cocotext_week_annotation_500/gts",imgname.replace('.jpg','.txt')),'w')
        for box,text in zip(boxes,texts):
            if len(text)<3:
                text='###'
            str_line = '{},{},{},{},{}\n'.format(box[0],box[1],box[2],box[3],text)
            f.write(str_line)



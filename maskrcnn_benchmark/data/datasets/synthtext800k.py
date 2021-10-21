import torch
import os
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
import json
import cv2
import torch
from scipy.special import comb as n_over_k
import scipy.io as sio
from tqdm import tqdm
# import unicode
from .augs import PSSAugmentation, SythAugmentation, TestAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.boxlist_ops import box_xyxy_to_xyxy
# from .utils import *
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, Polygons
# from maskrcnn_benchmark.utils.rec_util import LabelMap

def text_list_generate(text):
    word_list = []
    for part in text:
        part_word_list = part.strip().replace(' ', '\n').split('\n')
        for i in range(len(part_word_list)-1, -1, -1):
            if part_word_list[i] == '':
                part_word_list.remove('')
        word_list += part_word_list
    return word_list

def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)

def load_ann(img_path, filter_tag=False):
    txt_folder = "/home/wanghao/datasets/SynthText/Text_GT"
    gt = os.path.join(txt_folder, img_path.split('/')[-1].replace('.jpg', '.txt').replace('.png', '.txt').replace('.gif', '.txt'))
    # print(gt,img_path)
    # gt = unicode(gt, 'utf-8')#gt.decode('utf-8')
    item = {}
    item['polys'] = []
    item['tags'] = []
    item['texts'] = []
    item['gt_path'] = gt
    item['img_path'] = img_path
    # print(gt)
    reader = codecs.open(gt,encoding='utf-8').readlines()
    # reader = open(gt).readlines()
    for line in reader:
        parts = line.strip().split(',')
        if filter_tag:
            # label = 'fakelabel'
            label = parts[-1]
        else:
            label = parts[-1]
        label = filter_word(label)
        if len(label)<3:
            continue
        if label == '###':
            continue
        line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
        # if filter_tag:
        #     xmin, ymin, xmax, ymax = list(map(float, line[:4]))
        #     item['polys'].append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        # else:
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
        item['polys'].append(get_ordered_polys(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])))
        item['texts'].append(label.lower())
        if label == '###':
            item['tags'].append(True)
        else:
            item['tags'].append(False)
    # if filter_tag:
    #     if len(item['polys'])==0:
    #         return None, None, None
    item['polys'] = np.array(item['polys'], dtype=np.float32)
    item['tags'] = np.array(item['tags'], dtype=np.bool)
    item['texts'] = np.array(item['texts'], dtype=np.str)
    
    return item['img_path'], item['polys'], item['texts']
        
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

class SynthText(object):
    def __init__(self, img_folder):
        self.generate_information(img_folder)
        self.difficult_label = "###"
    def generate_information(self, img_folder):
        gt_mat = os.path.join(img_folder, 'gt.mat')
        s_data = sio.loadmat(gt_mat)
        names = s_data['imnames']
        name = names[0]
        image_path_list = [os.path.join(img_folder, name_i[0]) for name_i in name]
        gt_folder = "/home/wanghao/datasets/SynthText/Text_GT"
        gt_path_list = [os.path.join(gt_folder, gt) for gt in os.listdir(gt_folder)]
        self.image_path_list = sorted(image_path_list)
        self.gt_path_list = sorted(gt_path_list)
        self.filter_tag = True
        self.sample_num = len(self.image_path_list)
        # import pdb
        # pdb.set_trace()
    def len(self):
        return self.sample_num
    def getitem(self,index):
        img_path, polys, texts = load_ann(self.image_path_list[index], self.filter_tag)
        # if img_path == None:
        #     index = 0
        # img_path, polys, texts = load_ann(self.image_path_list[index], self.filter_tag)
        return img_path, polys, texts

class SynthTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None, is_train=True, augment=None):
        super().__init__()
        self.dataset = SynthText(data_dir)
        # print(self.dataset.len())
        # print(is_train)
        self.is_train = is_train
        self.transforms = transforms
        self.augment = eval(augment)()

    def __len__(self):
        return self.dataset.len()

    def __getitem__(self, index):
        if self.is_train:
            polys = []
            while len(polys) ==0:
                img_path, polys, texts = self.dataset.getitem(index)
                index = np.random.randint(0,len(self))
            img = imread(img_path, mode="RGB")
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags = self.augment(img, polys, texts)
            boxes = []#[[np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])] for poly in polys]
            # # boxes = np.array(boxes).reshape([-1,4])
            # order_polys = []
            # boundarys = []
            for poly in polys:
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
                # boundarys.append(pts_expand)
                # order_polys.append(get_ordered_polys(poly))
                # cv2.drawContours(aug_img, pts_expand.reshape([1,-1,2]).astype(np.int32),-1,color=(255,0,0),thickness=1)
            # cv2.imwrite(os.path.join('vis',os.path.basename(path)), aug_img[:,:,(2,1,0)])
            boxes = np.array(boxes).reshape([-1,4])
            # order_polys = np.array(order_polys).reshape([-1,8])
            # boundarys = np.array(boundarys).reshape([-1,NUM_POINT*4])
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            # boxlist.add_field('polys',torch.tensor(order_polys))
            # boxlist.add_field('boundarys',torch.tensor(boundarys))
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            boxlist.add_field('texts',tags)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, index
        else:
            img_path, polys, texts = self.dataset.getitem(index)
            img = imread(img_path, mode="RGB")
            aug_img, _, _ = self.augment(img)
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist=None
            if self.transforms:
                image,_ = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, None, index

    def get_img_info(self, index):
        if self.is_train:
            return {"path":"none", "height": 768, "width": 1280}
        path, _, _ = self.dataset.getitem(index)
        size = Image.open(path).size
        # size = [1280,768]
        return {"path":path, "height": size[1], "width": size[0]}
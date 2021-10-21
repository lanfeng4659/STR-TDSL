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
from tqdm import tqdm
# import unicode
from .augs import PSSAugmentation, SythAugmentation, TestAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, Polygons
# from maskrcnn_benchmark.utils.rec_util import LabelMap
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
def bezier_to_poly(bez, sample_n):
    Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * n_over_k(n,k)
    BezierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]
    assert(len(bez) == 16)
    s1_bezier = bez[:8].reshape((4,2))
    s2_bezier = bez[8:].reshape((4,2))
    t_plot = np.linspace(0, 1, sample_n)
    Bezier_top = np.array(BezierCoeff(t_plot)).dot(s1_bezier)
    Bezier_bottom = np.array(BezierCoeff(t_plot)).dot(s2_bezier)
    poly = Bezier_top.tolist()
    bottom = Bezier_bottom.tolist()
    # bottom.reverse()
    poly.extend(bottom)
    return poly

def get_ordered_polys(cnt):
    #print(cnt)
    bounding_box = cv2.minAreaRect(cnt.astype(np.int32))
    points = cv2.boxPoints(bounding_box)
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

# def spilt_point(pts):
#     if pts.shape[0]==4:
#         return pts[:2,:],pts[2:,:]

# def get_center_line(pts, num_exp=7):
#     sh=pts.shape
#     if sh[0]<4:
#         return None

#     up_point,down_point=spilt_point(pts)
#     if up_point is None:
#         return None

#     def expand(point_base):
#         #  all point must in clock or unclockwise
#         dis=np.linalg.norm(point_base[1:,:]-point_base[:-1,:],axis=1)
#         up_len=dis.sum()
#         dis_per=max(up_len/(num_exp-1),1)

#         line_seg=[0]
#         for i in range(dis.shape[0]):
#             line_seg.append(line_seg[i]+dis[i])
        
#         line_seg=np.array(line_seg)

#         exp_list=[]
#         for i in range(num_exp):
#             cur_pos=dis_per*i
            
#             dis=line_seg-cur_pos
#             index=np.argsort(np.abs(line_seg-cur_pos))
#             if dis[index[0]]*dis[index[1]]<0:
#                 a_idx,b_idx=index[0],index[1]
#             elif len(dis)>2:
#                 a_idx,b_idx=index[0],index[2]
#             else:
#                 a_idx,b_idx=index[0],index[1]

#             point_exp=(point_base[a_idx,:]-point_base[b_idx,:])*(cur_pos-line_seg[b_idx])/(line_seg[a_idx]-line_seg[b_idx]+1e-6)+point_base[b_idx,:]

#             exp_list.append(point_exp)
        
#         return exp_list
    
#     up_expand=expand(up_point)
#     down_expand=expand(down_point)
    
#     point_expand=np.array(up_expand+down_expand)
#     return point_expand

def add_line(cnt):
    # print(cnt.shape)  [14,2]
    cnt_new = np.zeros((7,2))
    for index in range(7):
        cnt_new[index] = (cnt[index] + cnt[-1-index])/2
    return cnt_new

class Synth150k(object):
    def __init__(self, path, is_training=True):
        self.is_training = is_training
        self.generate_information(path)
        self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        # self.convert_to_texts()
        # print(self.len())
    def id_dict(self, annotations):
        dict_ = {}
        for idx,anno in enumerate(annotations):
            k = str(anno['image_id'])
            if k not in dict_:
                dict_[k] = []
            dict_[k].append(idx)
        return dict_
    def generate_information_jason(self, path):
        with open(os.path.join(path, 'annotations/ecms_v1_maxlen25.json'), 'r', encoding='utf-8') as f:
            all_info1 = json.load(f)
        with open(os.path.join(path, 'annotations/syntext_word_eng.json'), 'r', encoding='utf-8') as f:
            all_info2 = json.load(f)
        self.annotations = [all_info1['annotations'], all_info2['annotations']]
        self.id_dicts = [self.id_dict(annos) for annos in self.annotations]
        self.img_folders = [os.path.join(path, 'images/emcs_imgs/'), os.path.join(path, 'images/syntext_word_eng/')]
        self.lens = [self.annotations[0][-1]['image_id']+1,self.annotations[1][-1]['image_id']]
    def generate_information(self, path):
        self.img_paths = []
        for folder in [os.path.join(path, 'images/emcs_imgs/'), os.path.join(path, 'images/syntext_word_eng/')]:
            for v in os.listdir(folder):
                self.img_paths.append(os.path.join(folder, v))
    def getitem(self, idx):
        img_path = self.img_paths[idx]
        f = open(img_path.replace("images","texts")+'.txt', mode='r')
        lines = f.readlines()
        beziers = []
        texts = []
        for line in lines:
            parts = line.split(',')
            text = filter_word("".join(parts[16:]))
            if len(text)<3:
                continue
            beziers.append([float(v) for v in parts[:16]])
            texts.append(text)
        f.close()
        return img_path, beziers, texts

    def len(self):
        return len(self.img_paths)
    def getitem_jason(self, idx):
        if idx < self.lens[0]:
            id_data, index = 0, idx
        else:
            id_data, index = 1, idx - self.lens[0] + 1 # offset from 1
        
        name_id = index
        image_name = ('%07d.jpg' % name_id) if id_data == 0 else ('%08d.jpg' % name_id)
        img_path = os.path.join(self.img_folders[id_data], image_name)

        beziers = []
        texts = []
        id = str(index)
        for annotation_id in self.id_dicts[id_data][id]:
            ann = self.annotations[id_data][annotation_id]
            text_label = "".join([self.CTLABELS[v] for v in ann['rec'] if v < len(self.CTLABELS)])
            # word = filter_word(text_label)
            word = text_label
            if len(word)==0:
                continue
            beziers.append(ann['bezier_pts']) #: bezier_pts:[x0,y0,...,x8,y8]
            texts.append(word)
        return img_path, beziers, texts
    def convert_to_texts(self):
        for idx in tqdm(range(self.len())):
            img_path, beziers, texts = self.getitem_jason(idx)
            folder = img_path.replace(os.path.basename(img_path),"").replace("images","texts")
            # print(folder)
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(os.path.join(folder, os.path.basename(img_path)+'.txt'),mode='w') as f:
                for ber, text in zip(beziers, texts):
                    str_ = ",".join([str(v) for v in ber]) + ',' + text + '\n'
                    f.writelines(str_)


class SynthText150kDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        self.dataset = Synth150k(data_dir, is_train)
        # print("hello")
        self.annotation_id = 0
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        # print("init end", self.dataset.len())
    def __len__(self):
        return self.dataset.len()
    def __getitem__(self, index):
        while True:
            img_path, beziers, texts = self.dataset.getitem(index)
            if len(texts)>0:
                break
            index = np.random.randint(0,self.dataset.len())
        # print("textsdddd",texts)
        img = imread(img_path)
        if self.is_train:
            texts = np.array(texts, dtype=np.str)
            beziers = np.array(beziers)
            polys_before = []
            # sample_n decides the shape of polys
            sample_n = 7
            for bez in beziers:
                poly_before = bezier_to_poly(bez, sample_n)
                polys_before.append(poly_before)
            beziers = beziers.reshape([-1,8,2])
            polys_before = np.array(polys_before).reshape([-1,2*sample_n,2])
            aug_img, polys_after, tags = self.augment(img, polys_before, texts)
            # print(tags)
            
            if len(tags)==0:
                print(tags,texts,img_path)
            for t in tags:
                if len(t)==0:
                    print(tags,texts,img_path)
            # print(tags)
            # lines = []
            # NUM_POINT = 7
            # polys = []
            # for poly in polys_after:
            #     lines.append(add_line(poly))
            #     poly = get_ordered_polys(poly)
            #     polys.append(poly)
            polys = np.array(polys_after)
            # print(polys.shape)
            # for p in polys[:,:7].reshape([-1,2]).astype(np.int32):
            #     cv2.circle(aug_img, (p[0], p[1]), 2, (255,0,0), 1)
            # for p in polys[:,7:].reshape([-1,2]).astype(np.int32):
            #     cv2.circle(aug_img, (p[0], p[1]), 2, (0,255,0), 1)
            # cv2.imwrite(os.path.join("curves", os.path.basename(img_path)), aug_img)
            # print(texts)
            # exit()
            boxes = []
            for poly in polys:
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])

            boxes = np.array(boxes).reshape([-1,4])
            polys = np.array(polys).reshape([-1,sample_n*4])
            # lines = np.array(lines).reshape([-1,14])
            # for line in lines:
            # cv2.circle(aug_img, lines[:,:7], radius, color, thickness)
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist = BoxList(boxes, image.size, mode="xyxy")
            boxlist.add_field('polys',torch.tensor(polys))
            # boxlist.add_field('lines',torch.tensor(lines))
            boxlist.add_field('labels',torch.tensor([1 for text in tags]))
            boxlist.add_field('texts',tags)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, index
        else:
            img = imread(img_path)
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
        name_id = index
        image_name = ('%07d.jpg' % name_id)
        img_path = self.dataset.img_folder + image_name
        size = Image.open(img_path).size
        return {"path":img_path, "height": size[1], "width": size[0]}
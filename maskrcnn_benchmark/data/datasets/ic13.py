import torch
import os
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
import cv2
# import unicode
from .augs import PSSAugmentation,TestAugmentation,RetrievalAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, Polygons
# from maskrcnn_benchmark.utils.rec_util import LabelMap
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
def get_ordered_polys(cnt):
    #print(cnt)
    # bounding_box = cv2.minAreaRect(cnt.astype(np.int32))
    # points = cv2.boxPoints(bounding_box)
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
def load_ann(gt_paths,img_paths,split_char=' '):
    res = []
    for gt,img_path in zip(gt_paths,img_paths):
        # gt = unicode(gt, 'utf-8')#gt.decode('utf-8')
        item = {}
        item['polys'] = []
        item['boxes'] = []
        item['tags'] = []
        item['texts'] = []
        item['gt_path'] = gt
        item['img_path'] = img_path
        # print(gt)
        reader = codecs.open(gt,encoding='utf-8').readlines()
        # reader = open(gt).readlines()
        for line in reader:
            # print(line)
            parts = line.strip().split(split_char)
            label = parts[-1]
            label = filter_word(label)
            if len(label)<3:
                continue
            if label == '###':
                continue
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            xmin, ymin, xmax, ymax = list(map(float, line[:4]))
            item['polys'].append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            item['texts'].append(label.lower())
            if label == '###':
                item['tags'].append(True)
            else:
                item['tags'].append(False)
        # if len(item['polys'])==0:
        #     continue
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['tags'] = np.array(item['tags'], dtype=np.bool)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    return res
class ICDAR2013(object):
    def __init__(self, path, is_training = True):
        self.is_training = is_training
        self.difficult_label = '###'
        self.generate_information(path)
    def generate_information(self, path):
        if self.is_training:
            image_floder = os.path.join(path, 'train_images')
            gt_floder = os.path.join(path, 'train_gts')
            self.image_path_list = [os.path.join(image_floder, image) for image in os.listdir(image_floder)]
            gt_path_list    = [os.path.join(gt_floder, gt) for gt in os.listdir(gt_floder)]
            self.image_path_list = sorted(self.image_path_list)
            gt_path_list = sorted(gt_path_list)
            self.targets = load_ann(gt_path_list,self.image_path_list)
            self.sample_num = len(self.targets)
        else:
            image_floder = os.path.join(path, 'test_images')
            gt_floder = os.path.join(path, 'test_gts')
            self.image_path_list = [os.path.join(image_floder, image) for image in os.listdir(image_floder)]
            gt_path_list    = [os.path.join(gt_floder, gt) for gt in os.listdir(gt_floder)]
            self.image_path_list = sorted(self.image_path_list)
            gt_path_list = sorted(gt_path_list)
            print(len(self.image_path_list),len(gt_path_list))
            self.targets = load_ann(gt_path_list,self.image_path_list,split_char=',')
            self.sample_num = len(self.image_path_list)
            # print(self.sample_num,len(self.targets))
    def len(self):
        return self.sample_num
    def getitem(self,index):
        if self.is_training:
            return self.targets[index]['img_path'], self.targets[index]['polys'].copy(), self.targets[index]['texts'].copy()
        else:
            # print(index)
            return self.targets[index]['img_path'], self.targets[index]['polys'].copy(), self.targets[index]['texts'].copy()
NUM_POINT=7
class Icdar13Dateset(torch.utils.data.Dataset):

    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = ICDAR2013(data_dir, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            path, polys, texts = self.dataset.getitem(idx)
            img = imread(path, mode="RGB")
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            # print(texts)
            aug_img, polys, tags = self.augment(img, polys, texts)
            if len(tags) == 0:
                aug_img, polys, tags = self.augment(img, polys, texts, no_crop=True)

            boxes = []
            for poly in polys:
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
                # boundarys.append(pts_expand)
                # order_polys.append(get_ordered_polys(poly))
            #     cv2.drawContours(aug_img, poly.reshape([1,-1,2]).astype(np.int32),-1,color=(255,0,0),thickness=1)
            # cv2.imwrite(os.path.join('vis',os.path.basename(path)), aug_img[:,:,(2,1,0)])
            # print(tags)
            boxes = np.array(boxes).reshape([-1,4])
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            # boxlist.add_field('polys',torch.tensor(order_polys))
            boxlist.add_field('texts',tags)
            # boxlist.add_field('polys',torch.tensor(order_polys))
            # boxlist.add_field('boundarys',torch.tensor(boundarys))
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
        else:
            path, polys, queries = self.dataset.getitem(idx)
            # print(polys)
            img = imread(path,mode="RGB")
            ori_h, ori_w, _ = img.shape
            aug_img, polys, tags = self.augment(img, None, None)
            test_h, test_w, _ = aug_img.shape
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            trues = [1]*len(queries)
            boxlist = BoxList([[0,0,0,0]], image.size, mode="xyxy")
            boxlist.add_field('retrieval_trues',trues)
            boxlist.add_field('texts',np.array(queries))
            boxlist.add_field('scale',np.array([ori_w/test_w, ori_h/test_h]))
            boxlist.add_field('path',np.array(path))
            boxlist.add_field("y_trues",trues)
            # boxlist.add_field('test_texts',self.dataset.all_texts)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
    # def __getitem__(self, idx):
    #     if self.is_train:
    #         path, polys, texts = self.dataset.getitem(idx)
    #         boxes = []
    #         order_polys = []
    #         for poly in polys:
    #             boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
    #             # order_polys.append(get_ordered_polys(poly))
    #         boxes = np.array(boxes).reshape([-1,4])
    #         image = Image.open(path).convert('RGB')

    #         boxlist = BoxList(boxes, image.size, mode="xyxy")
    #         # boxlist.add_field('polys',torch.tensor(order_polys))
    #         boxlist.add_field('texts',texts)
    #         # boxlist.add_field('polys',torch.tensor(order_polys))
    #         # boxlist.add_field('boundarys',torch.tensor(boundarys))
    #         boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in texts]))
    #         if self.transforms:
    #             image, boxlist = self.transforms(image, boxlist)
    #         # return the image, the boxlist and the idx in your dataset
    #         return image, boxlist, idx
    #     else:
    #         path, _, _ = self.dataset.getitem(idx)
    #         img = imread(path)
    #         aug_img, _, _ = self.augment(img)
    #         image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
    #         boxlist=None
    #         if self.transforms:
    #             image,_ = self.transforms(image, boxlist)
    #         # return the image, the boxlist and the idx in your dataset
    #         return image, None, idx

    def __len__(self):
        return self.dataset.len()

    def expand_point(self, poly):
        poly = np.array(poly).reshape(-1, 2)
        up_x = np.linspace(poly[0, 0], poly[1, 0], NUM_POINT)
        up_y = np.linspace(poly[0, 1], poly[1, 1], NUM_POINT)
        up = np.stack((up_x, up_y), axis=1)
        do_x = np.linspace(poly[2, 0], poly[3, 0], NUM_POINT)
        do_y = np.linspace(poly[2, 1], poly[3, 1], NUM_POINT)
        do = np.stack((do_x, do_y), axis=1)
        poly_expand = np.concatenate((up, do), axis=0)
        return poly_expand.reshape(-1).tolist()

    def get_img_info(self, idx):
        path, _, _ = self.dataset.getitem(idx)
        # size = Image.open(path).size
        size = [720,1280]
        return {"path":path, "height": size[1], "width": size[0]}



import torch
import os
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
import cv2
import json
from maskrcnn_benchmark.data.datasets.augs import PSSAugmentation,TestAugmentation,RetrievalAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, Polygons
# from maskrcnn_benchmark.utils.rec_util import LabelMap
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)

class ArT(object):
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
            self.targets = load_ann(gt_path_list, self.image_path_list)
            self.sample_num = len(self.targets)
        else:
            # image_floder = os.path.join(path, 'test_images')
            # gt_floder = os.path.join(path, 'test_gts')
            # self.image_path_list = [os.path.join(image_floder, image) for image in os.listdir(image_floder)]
            # gt_path_list    = [os.path.join(gt_floder, gt) for gt in os.listdir(gt_floder)]
            # self.image_path_list = sorted(self.image_path_list)
            # gt_path_list = sorted(gt_path_list)
            # self.targets = load_ann(gt_path_list, self.image_path_list)
            # self.sample_num = len(self.targets)

            self.parse_data(path)
            # self.sample_num = len(self.image_path_list)
    def parse_data(self,path):
        # path = "datasets/ArT/train_labels.json"
        with open(os.path.join(path, 'train_labels.json'), 'r') as load_f:
            dict_ = json.load(load_f)
        img_lists = list(dict_.keys())
        self.img_lists = [os.path.join(path, 'train_images', v+'.jpg') for v in img_lists]
        queries = {}
        for name in img_lists:
            datas = dict_[name]
            for data in datas:
                text, script, ignore = data['transcription'], data['language'], data['illegibility']
                text = filter_word(text).lower()
                if len(text) < 3: continue
                if text in queries.keys():
                    queries[text] += 1
                else:
                    queries[text] = 1
        # queries = sorted(queries.items(), key= lambda x:x[1], reverse=True)
        self.str_queries = [k for k,v in queries.items() if v >= 5]
        y_trues = np.zeros([len(self.str_queries), len(self.img_lists)])

        for idx, name in enumerate(img_lists):
            datas = dict_[name]
            for data in datas:
                text, script, ignore = data['transcription'], data['language'], data['illegibility']
                text = filter_word(text).lower()
                if text in self.str_queries:
                    y_trues[self.str_queries.index(text), idx] = 1
        self.y_trues = y_trues
        self.sample_num = len(self.img_lists)
    def len(self):
        return self.sample_num
    def getitem(self,index):
        if self.is_training:
            return self.targets[index]['img_path'], self.targets[index]['polys'].copy(), self.targets[index]['texts'].copy()
        else:
            return self.img_lists[index], None, self.str_queries
NUM_POINT=7
class ArTDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            if augment == "PSSAugmentation":
                self.augment = eval(augment)(longer_side_arange=np.arange(1280,2560, 32))
            else:
                self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1600)
        self.transforms=transforms
        self.is_train=is_train
        data_dir = '/workspace/wanghao/projects/RetrievalTPAMI/datasets/ArT'
        self.dataset = ArT(data_dir, is_train)

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
            boxes = np.array(boxes).reshape([-1,4])
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist = BoxList(boxes, image.size, mode="xyxy")
            boxlist.add_field('texts',tags)
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            if self.transforms:
                # print("before",boxlist,(boxlist.get_field("texts")))
                image, boxlist = self.transforms(image, boxlist)
                # print("after",boxlist,(boxlist.get_field("texts")))
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
            boxlist.add_field("det_thred", 0.2)
            # boxlist.add_field('test_texts',self.dataset.all_texts)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx

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


if __name__ == "__main__":
    path = "datasets/ArT/train_labels.json"
    with open(path, 'r') as load_f:
        dict_ = json.load(load_f)
    img_lists = list(dict_.keys())
    queries = {}
    for name in img_lists:
        datas = dict_[name]
        for data in datas:
            text, script, ignore = data['transcription'], data['language'], data['illegibility']
            text = filter_word(text).lower()
            if len(text) < 3: continue
            if text in queries.keys():
                queries[text] += 1
            else:
                queries[text] = 1
    # queries = sorted(queries.items(), key= lambda x:x[1], reverse=True)
    queries = [k for k,v in queries.items() if v >= 5]
    # print(queries)
    # print(len(queries))



    # queries = {}
    # folder = '/workspace/wanghao/projects/RetrievalTPAMI/datasets/icdar2015/test_gts'
    # for path in [os.path.join(folder, v) for v in os.listdir(folder)]:
    #     # print(path)
    #     reader = open(path).readlines()
    #     for line in reader:
    #         parts = line.strip().split(',')
    #         label = parts[-1]
    #         label = filter_word(label).lower()
    #         if label in queries.keys():
    #             queries[label] += 1
    #         else:
    #             queries[label] = 1
    # queries = sorted(queries.items(), key= lambda x:x[1], reverse=True)
    # print(queries)




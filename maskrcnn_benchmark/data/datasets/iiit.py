import torch
import os
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
# import unicode
from .augs import PSSAugmentation,TestAugmentation,SythAugmentation,RetrievalAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
import numpy as np
import cv2
import scipy.io as scio
class IIIT(object):
    def __init__(self, path, is_training = True):
        # assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        self.all_texts = []
        self.parse_data(path)

    def parse_data(self,gt_path):
        dataFile = os.path.join(gt_path,"data.mat")
        imgPath = os.path.join(gt_path,"imgDatabase")
        data = scio.loadmat(dataFile)
        images = [img for img in os.listdir(imgPath)]
        str_queries = []
        for i in range(data['data'].shape[1]):
            str_queries.append(str(data['data'][0,i][0][0][0][0]).lower())
        y_trues = np.zeros([len(str_queries),len(images)])
        for i in range(len(str_queries)):
            for j in range(len(data['data'][0,i][1])):
                imgName = data['data'][0,i][1][j][0][0]
                y_trues[i,images.index(imgName)] = 1
        self.img_lists = [os.path.join(gt_path,"imgDatabase",imgName) for imgName in images]
        self.str_queries = str_queries
        self.y_trues = y_trues


    def len(self):
        return len(self.img_lists)
    def getitem(self,index):
        return self.img_lists[index], self.str_queries, self.y_trues
NUM_POINT=7
class IIITDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True, augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = IIIT(data_dir, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            # print("get item")
            path, queries, trues = self.dataset.getitem(idx)
            img = imread(path)
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags = self.augment(img, None, None)
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist = BoxList([], image.size, mode="xyxy")
            boxlist.add_field('retrieval_trues',trues)
            boxlist.add_field('texts',queries)
            # boxlist.add_field("y_trues",trues)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
        else:
            path, queries, trues = self.dataset.getitem(idx)
            # print(polys)
            img = imread(path,mode="RGB")
            ori_h, ori_w, _ = img.shape
            aug_img, polys, tags = self.augment(img, None, None)
            test_h, test_w, _ = aug_img.shape
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

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
        # print("get info")
        if self.is_train:
            return {"path":"none", "height": 768, "width": 1280}
        path, _, _ = self.dataset.getitem(idx)
        size = Image.open(path).size
        # size = [1280,768]
        return {"path":path, "height": size[1], "width": size[0]}


if __name__ == "__main__":
    data_dir = "/root/datasets/ic15_end2end"
    ic15_dataset = IC15(data_dir)
    image, boxlist, idx = ic15_dataset[0]
    import ipdb; ipdb.set_trace()


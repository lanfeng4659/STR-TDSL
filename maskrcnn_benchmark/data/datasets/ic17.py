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
def load_ann(gt_paths, img_paths):
    res = []
    for gt,img_path in zip(gt_paths,img_paths):
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
            label = parts[-1]
            script = parts[-2]
            label = '###' if script.lower() != "latin" else label
            label = filter_word(label)
            if len(label)<3:
                continue
            if label == '###':
                continue
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label)
            if label == '###':
                item['tags'].append(True)
            else:
                item['tags'].append(False)
        if len([label for label in item['texts'] if label!="###"])==0:
            continue
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['tags'] = np.array(item['tags'], dtype=np.bool)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    #     print('read',item['polys'])
    # exit()
    return res
class ICDAR2017(object):
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
            # for img_path, gt_path in zip(self.image_path_list, gt_path_list):
            #     print(img_path, gt_path)
            # self.targets = load_ann(gt_path_list,self.image_path_list)[:1500]
            self.targets = load_ann(gt_path_list,self.image_path_list)
            print(len(self.targets))
    def len(self):
        return len(self.targets)
    def getitem(self,index):
        if self.is_training:
            return self.targets[index]['img_path'], self.targets[index]['polys'].copy(), self.targets[index]['texts'].copy()
NUM_POINT=7
class Icdar17Dateset(torch.utils.data.Dataset):

    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            # print(augment)
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = ICDAR2017(data_dir, is_train)

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
            boxes = []#[[np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])] for poly in polys]
            # boxes = np.array(boxes).reshape([-1,4])
            order_polys = []
            # boundarys = []
            for poly in polys:
                # print("before:",poly)
                # pts_expand=self.expand_point(poly)
                # # print("after:",poly)
                # if pts_expand is None:
                #     continue
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
                # boundarys.append(pts_expand)
                # order_polys.append(get_ordered_polys(poly))
            #     cv2.drawContours(aug_img, poly.reshape([1,-1,2]).astype(np.int32),-1,color=(255,0,0),thickness=1)
            # cv2.imwrite(os.path.join('vis',os.path.basename(path)), aug_img[:,:,(2,1,0)])
            boxes = np.array(boxes).reshape([-1,4])
            order_polys = np.array(order_polys).reshape([-1,8])
            # boundarys = np.array(boundarys).reshape([-1,NUM_POINT*4])
            # print(boxes.shape, len(boundarys))
            # h, w, _ = aug_img.shape
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            # boxlist.add_field('polys',torch.tensor(order_polys))
            boxlist.add_field('texts',tags)
            # boxlist.add_field('boundarys',torch.tensor(boundarys))
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
        else:
            path, _, _ = self.dataset.getitem(idx)
            img = imread(path)
            aug_img, _, _ = self.augment(img)
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')
            boxlist=None
            if self.transforms:
                image,_ = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, None, idx


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
        size = Image.open(path).size
        return {"path":path, "height": size[1], "width": size[0]}


if __name__ == "__main__":
    data_dir = "/root/datasets/ic15_end2end"
    ic15_dataset = IC15(data_dir)
    image, boxlist, idx = ic15_dataset[0]
    import ipdb; ipdb.set_trace()


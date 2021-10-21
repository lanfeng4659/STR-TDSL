import torch
import os
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
# import unicode
from .augs import PSSAugmentation,TestAugmentation,SythAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
import os
import numpy as np
import cv2
from xml.etree.ElementTree import ElementTree
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    # print(chars)
    # exit()
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
def load_ann(gt_paths,chars):
    res = []
    idxs = []
    for gt in gt_paths:
        # gt = unicode(gt, 'utf-8')#gt.decode('utf-8')
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['texts'] = []
        item['gt_path'] = gt
        # print(gt)
        reader = codecs.open(gt,encoding='utf-8').readlines()
        # reader = open(gt).readlines()
        for line in reader:
            parts = line.strip().split(',')
            label = parts[-1]
            # print(label)
            # print("before:",label)
            label = filter_word(label,chars=chars)
            # print("after:",label)
            if len(label)<2:
                continue
            if label == '###':
                continue
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label)
        if len(item['polys'])==0:
            continue
        # print(len(res))
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)

    return res
class SynthtextChinese(object):
    def __init__(self, path, is_training = True):
        assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        # self.chars = np.load(os.path.join(path, 'chars.npy')).tolist()
        self.chars = np.load("/workspace/wanghao/projects/Pytorch-yolo-phoc/selected_chars.npy").tolist()
        self.generate_information(path)
        
    def generate_information(self, path):
        if self.is_training:
            # self.image_floder = os.path.join(path, 'images')
            self.gt_floder = os.path.join(path, 'gts')
            self.gt_list    =[os.path.join(self.gt_floder, name) for name in os.listdir(self.gt_floder)]
            self.gt_floder = os.path.join(path.replace("SynthText_Chinese","SynthText_Chinese_RCTW"), 'gts')
            self.gt_list.extend([os.path.join(self.gt_floder, name) for name in os.listdir(self.gt_floder)])
            self.samples = load_ann(self.gt_list,self.chars)
            print(len(self.samples))


    def len(self):
        return len(self.samples)
    def getitem(self,index):
        # print(self.len())
        sample = self.samples[index]
        gt_path = sample["gt_path"]
        img_path = gt_path.replace("gts","images").replace(".txt",".jpg")
        return img_path, sample['polys'].copy(), sample['texts'].copy()
NUM_POINT=7
class SynthtextChineseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = SynthtextChinese(data_dir, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            # print("get item")
            path, polys, texts = self.dataset.getitem(idx)
            img = imread(path)
            # print(polys.shape, polys)
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


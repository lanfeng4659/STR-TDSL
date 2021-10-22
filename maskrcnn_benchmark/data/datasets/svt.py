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
from xml.etree.ElementTree import ElementTree

class SVT(object):
    def __init__(self, path, is_training = True):
        # assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        self.all_texts = []
        self.generate_information(path)
        
    def generate_information(self, path):
        self.path = path
        if self.is_training:
            self.datas = self.parse_xml_file(os.path.join(path,'train.xml'))
        else:
            self.datas = self.parse_xml_file(os.path.join(path,'test.xml'))
            # self.write_to_txts(self.datas, "./svts")
            # self.gt_list    = os.listdir(self.gt_floder)
    def write_to_txts(self, datas,folder):
        for data in datas:
            filename = os.path.join(folder, os.path.basename(data['path'])).replace(".jpg",'.txt')
            print(filename)
            f = open(filename, 'w')
            for box, text in zip(data["xyxys"], data["texts"]):
                line = "{},{},{},{},{}\r\n".format(box[0],box[1],box[2],box[3],text)
                f.write(line)
    def parse_xml_file(self,gt_path):
        datas = []
        tree = ElementTree()
        tree.parse(gt_path)
        for object_ in tree.findall("image"):
            image_name = object_.find("imageName").text
            dict_ = {}
            dict_['path'] = os.path.join(self.path, image_name)
            boxes = []
            texts = []
            for text_object in object_.findall("taggedRectangles/taggedRectangle"):
                text = text_object.find("tag").text
                # print(text)
                rec = text_object.attrib
                # {'height': '38', 'width': '55', 'x': '645', 'y': '294'}
                x,y,h,w = int(rec['x']),int(rec['y']),int(rec['height']),int(rec['width'])
                xmin = int(x)
                ymin = int(y)
                xmax = int(x + w)
                ymax = int(y + h)
                boxes.append([xmin,ymin,xmax,ymax])
                texts.append(text.lower())
                self.all_texts.append(text.lower())
            dict_['xyxys'] = np.array(boxes, dtype=np.float32)
            dict_['boxes'] = np.array(boxes, dtype=np.float32)[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2])
            dict_['texts'] = np.array(texts, dtype=np.str)
            datas.append(dict_)
        self.str_queries = []
        for text in self.all_texts:
            if text not in self.str_queries:
                self.str_queries.append(text)

        y_trues = np.zeros([len(self.str_queries),len(datas)])
        # print(y_trues.shape)
        for idx,data in enumerate(datas):
            for text in data["texts"]:
                y_trues[self.str_queries.index(text),idx]=1
        self.y_trues = y_trues
        return datas


    def len(self):
        return len(self.datas)
    def getitem(self,index):
        # print(self.len())
        data = self.datas[index]
        # print(data)
        return data['path'], data['boxes'].copy(), data['texts'].copy(),self.str_queries
NUM_POINT=7
class SVTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True, augment=None):
        super().__init__()
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
            # self.augment = TestAugmentation(longer_side=1024)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = SVT(data_dir, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            # print("get item")
            path, polys, texts,queries = self.dataset.getitem(idx)
            img = imread(path)
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags = self.augment(img, polys, texts)
            boxes = []
            for poly in polys:
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
            boxes = np.array(boxes).reshape([-1,4])
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            boxlist.add_field('texts',tags)
            if self.transforms:
                image, boxlist = self.transforms(image, boxlist)
            # return the image, the boxlist and the idx in your dataset
            return image, boxlist, idx
        else:
            path, polys, texts,queries = self.dataset.getitem(idx)
            # print(polys)
            img = imread(path)
            ori_h, ori_w, _ = img.shape
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags = self.augment(img, polys, texts)
            test_h, test_w, _ = aug_img.shape
            boxes = []
            for poly in polys:
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
            boxes = np.array(boxes).reshape([-1,4])
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            boxlist.add_field('texts',np.array(queries))
            boxlist.add_field('scale',np.array([ori_w/test_w, ori_h/test_h]))
            boxlist.add_field('path',np.array(path))
            boxlist.add_field("y_trues",self.dataset.y_trues)
            boxlist.add_field("det_thred", 0.05)
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
        path, _, _,_ = self.dataset.getitem(idx)
        size = Image.open(path).size
        # size = [1280,768]
        return {"path":path, "height": size[1], "width": size[0]}


if __name__ == "__main__":
    data_dir = "/root/datasets/ic15_end2end"
    ic15_dataset = IC15(data_dir)
    image, boxlist, idx = ic15_dataset[0]
    import ipdb; ipdb.set_trace()


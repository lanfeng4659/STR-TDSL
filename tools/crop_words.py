import torch
import os
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import codecs
from tqdm import tqdm
import os
import numpy as np
import cv2
from xml.etree.ElementTree import ElementTree

class Synthtext90k(object):
    def __init__(self, path, is_training = True):
        assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        self.generate_information(path)
    def generate_information(self, path):
        if self.is_training:
            self.image_floder = os.path.join(path, 'images')
            self.gt_floder = os.path.join(path, 'annotations')
            # self.image_list = os.listdir(self.image_floder)
            self.gt_list    = os.listdir(self.gt_floder)
    def parse_xml_file(self,gt_path):
        texts = []
        polys = []
        tree = ElementTree()
        tree.parse(gt_path)
        for object_ in tree.findall("object"):
        # print(objects)
            text = object_.find("name").text
            xmin = int(object_.find("bndbox/xmin").text)
            ymin = int(object_.find("bndbox/ymin").text)
            xmax = int(object_.find("bndbox/xmax").text)
            ymax = int(object_.find("bndbox/ymax").text)
            texts.append(text)
            polys.append([xmin,ymin,xmax,ymax])
        return polys,np.array(texts, dtype=np.str)


    def len(self):
        return len(self.gt_list)
    def getitem(self,index):
        # print(self.len())
        gt_name = self.gt_list[index]
        gt_path = os.path.join(self.gt_floder, gt_name)
        img_path = os.path.join(self.image_floder, gt_name.replace('.xml','.jpg'))
        polys, texts = self.parse_xml_file(gt_path)
        # print(texts)
        # img = cv2.imread(img_path)
        # print(self.image_path_list[index])
        return img_path, polys, texts
save_path = './datasets/SynthText_90KDict/crops'
dataset = Synthtext90k('./datasets/SynthText_90KDict/')
# if os.path.exists(save_path):
#     os.makedirs(save_path)
for i in tqdm(range(dataset.len())):
    path, polys, texts = dataset.getitem(i)
    image = Image.open(path)
    width, height = image.size

    for poly, text in zip(polys, texts):
        xmin,ymin,xmax,ymax = poly
        img_save_path = os.path.join(save_path, os.path.basename(path).replace('.jpg','__{}.jpg'.format(text)))
        # if os.path.exist(img_save_path):
        #     continue
        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            continue
        patch = image.crop((np.clip(xmin-4,0,width-1), np.clip(ymin-4,0,height-1), np.clip(xmax+4,0,width-1), np.clip(ymax+4,0,height-1)))
        
        patch.save(img_save_path)
# num = 0
# for i in tqdm(range(dataset.len())):
#     path, polys, texts = dataset.getitem(i)
#     num += len(texts)
# print(num)

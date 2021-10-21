import os

import torch
import torch.utils.data
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import cv2
import sys
import json
import scipy.io as sio
import glob

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from maskrcnn_benchmark.data.datasets.augs import PSSAugmentation, TestAugmentation
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
# from maskrcnn_benchmark.structures.segmentation_mask import Polygons

# from maskrcnn_benchmark.utils.rec_util import LabelMap
def filter_word(text,chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    char_list = [c for c in text if c in chars]
    return "".join(char_list)
class TotalText(object):
    def __init__(self, path, is_training = True):
        self.is_training = is_training
        self.difficult_label = '#'
        if is_training:
            gt_folder = os.path.join(path, 'Groundtruth2/Polygon/Train/')
            # print(gt_folder)
            self.gt_file_list=glob.glob(os.path.join(gt_folder,'*mat'))
            self.image_folder=os.path.join(path, 'Dataset/Images/Train')
            self.gt_folder=gt_folder
        else:
            gt_folder = os.path.join(path, 'Groundtruth2/Polygon/Test/')
            self.gt_file_list=glob.glob(os.path.join(gt_folder,'*mat'))
            self.image_folder=os.path.join(path, 'Dataset/Images/Test')

            self.gt_folder=gt_folder
            self.parse_data()
    def len(self):
        return len(self.gt_file_list)
    def parse_data(self, instance_num=4):
        image_str_dict = {}
        for idx in range(len(self.gt_file_list)):
            gt_path=self.gt_file_list[idx]
            image_name = os.path.basename(gt_path).replace('poly_gt_','').replace('mat','jpg')
            polygt = sio.loadmat(gt_path)['polygt']
            for p in polygt:
                image_str_dict[image_name] = [filter_word(str(p[-2][0])).lower() for p in polygt if len(filter_word(str(p[-2][0])))>=3]
        query_nums_dict = {}
        for key, value in image_str_dict.items():
            for v in value:
                if v in query_nums_dict.keys():
                    query_nums_dict[v] += 1
                else:
                    query_nums_dict[v] = 1
        query_nums_list = sorted(query_nums_dict.items(), key=lambda x: x[1], reverse=True)
        query_nums_dict = {k:v for k,v in query_nums_list if v >=instance_num}
        # import ipdb; ipdb.set_trace()
        images = list(image_str_dict.keys())
        str_queries = list(query_nums_dict.keys())

        y_trues = np.zeros([len(str_queries),len(images)])
        for i in range(len(str_queries)):
            for j in range(len(images)):
                y_trues[i,j] = 1 if str_queries[i] in image_str_dict[images[j]] else 0
            # y_trues = correct_labels(i,images,y_trues, str_queries[i])
        # print(y_trues.sum(axis=0), len(str_queries))
        self.img_lists = [os.path.join(self.image_folder,imgName) for imgName in images]
        self.str_queries = str_queries
        self.y_trues = y_trues
        print(len(self.str_queries))
    def GetPts_total(self,polygt, idx):
        pts_lst = []
        if polygt[idx][5].shape[0] == 0:
            hard = 1
        else:
            hard = int(polygt[idx][5][0]=='#')
        for pts_num in range(polygt[idx][1].shape[1]):
            pts_lst.append([polygt[idx][1][0][pts_num],polygt[idx][3][0][pts_num]])
        pts = np.array(pts_lst, dtype=np.int32)
        return pts, hard

    def Gettext_total(self,polygt,idx):
        return polygt[idx][4]
    def get_groundtruth(self,polygt):
        polys=[]
        ignores=[]
        texts = []
        num=len(polygt)

        for ii in range(num):
            pts,hard=self.GetPts_total(polygt,ii)
            text = self.Gettext_total(polygt,ii)
            if len(text)==0:
                continue
            text = text[0]
            text = filter_word(text)
            if len(text)<3:
                continue
            polys.append(pts.reshape(-1,2))
            ignores.append(hard)
            # text = text if len(text)>0 else self.difficult_label
            # print(pts, text)
            texts.append(text)
        assert len(polys)==len(texts), print(polys,texts)
        return polys, texts, ignores
    def getitem(self,idx):
        if self.is_training:
            gt_path=self.gt_file_list[idx]
            polygt = sio.loadmat(gt_path)['polygt']
            img_path=os.path.join(self.image_folder,os.path.basename(gt_path).split('_')[-1].replace('mat','jpg'))
            if not os.path.isfile(img_path):
                img_path=img_path.replace('jpg','JPG')        
            polys, texts, ignores=self.get_groundtruth(polygt)
            return img_path, polys, texts
        else:
            return self.img_lists[idx], self.str_queries.copy(), self.y_trues.copy()

        
NUM_POINT=7
class TotalTextDateset(torch.utils.data.Dataset):
    def __init__(self, data_dir, use_difficult=False, transforms=None, is_train=True,augment=None):
        if is_train:
            self.augment = eval(augment)()
        else:
            self.augment = TestAugmentation(longer_side=1280)
        self.transforms=transforms
        self.is_train=is_train
        self.dataset = TotalText(data_dir, is_train)
        # self.max_lens=35
        # self.label_map = LabelMap()

        # if is_train:
        #     gt_folder = os.path.join(data_dir, 'Groundtruth2/Polygon/Train/')
        #     self.gt_file_list=glob.glob(os.path.join(gt_folder,'*mat'))
        #     self.image_folder=os.path.join(data_dir, 'Dataset/Images/Train')
        #     self.gt_folder=gt_folder
        # else:
        #     gt_folder = os.path.join(data_dir, 'Groundtruth2/Polygon/Test/')
        #     self.gt_file_list=glob.glob(os.path.join(gt_folder,'*mat'))
        #     self.image_folder=os.path.join(data_dir, 'Dataset/Images/Test')

        #     self.gt_folder=gt_folder

    def __getitem__(self, idx):
        if self.is_train:
            while True:
                path, polys, texts = self.dataset.getitem(idx)
                # print(texts)
                if len(texts)>0:
                    break
                idx = np.random.randint(0,self.dataset.len())

            img = imread(path)
            # print(polys.shape, polys)
            assert len(polys)==len(texts),print(polys,texts)
            aug_img, polys, tags_ = self.augment(img, polys, texts)
            boxes = []#[[np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])] for poly in polys]
            # boxes = np.array(boxes).reshape([-1,4])
            expand_polys = []
            tags = []
            for poly,tag in zip(polys,tags_):
                # print(poly.shape)
                pts_expand=self.expand_point(poly,NUM_POINT)
                if pts_expand is None:
                    continue
                boxes.append([np.min(poly[:,0]), np.min(poly[:,1]), np.max(poly[:,0]), np.max(poly[:,1])])
                expand_polys.append(pts_expand)
                tags.append(tag)
            if len(tags)==0:
                return self.__getitem__(np.random.randint(0,self.dataset.len()))
            #     cv2.drawContours(aug_img, pts_expand.reshape([1,-1,2]).astype(np.int32),-1,color=(255,0,0),thickness=1)
            # cv2.imwrite(os.path.join('tt',os.path.basename(path)), aug_img[:,:,(2,1,0)])
            boxes = np.array(boxes).reshape([-1,4])
            expand_polys = np.array(expand_polys).reshape([-1,NUM_POINT*4])
            # for p in expand_polys.reshape([-1,14,2])[:,:7].reshape([-1,2]).astype(np.int32):
            #     cv2.circle(aug_img, (p[0], p[1]), 2, (255,0,0), 1)
            # for p in expand_polys.reshape([-1,14,2])[:,7:].reshape([-1,2]).astype(np.int32):
            #     cv2.circle(aug_img, (p[0], p[1]), 2, (0,255,0), 1)
            # cv2.imwrite(os.path.join("tt", os.path.basename(path)), aug_img)
            # print(boxes.shape, len(expand_polys))
            # h, w, _ = aug_img.shape
            image = Image.fromarray(aug_img.astype(np.uint8)).convert('RGB')

            boxlist = BoxList(boxes, image.size, mode="xyxy")
            boxlist.add_field('polys',torch.tensor(expand_polys))
            boxlist.add_field('labels',torch.tensor([-1 if text==self.dataset.difficult_label else 1 for text in tags]))
            boxlist.add_field('texts',np.array(tags, dtype=np.str))
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
        if self.transforms:
            # FIXME may ssome box out of the region??
            
            image, boxlist = self.transforms(image, boxlist)
        return image, boxlist, idx

    def __len__(self):
        return self.dataset.len()

    def GetPts_total(self,polygt, idx):
        pts_lst = []
        if polygt[idx][5].shape[0] == 0:
            hard = 1
        else:
            hard = int(polygt[idx][5][0]=='#')
        for pts_num in range(polygt[idx][1].shape[1]):
            pts_lst.append([polygt[idx][1][0][pts_num],polygt[idx][3][0][pts_num]])
        pts = np.array(pts_lst, dtype=np.int32)
        return pts, hard

    def Gettext_total(self,polygt,idx):
        return polygt[idx][4]

    # FIXME still have bug, in img719.jpg, a norm text be ignore
    def spilt_point(self,pts):

        if pts.shape[0]==4:
            return pts[:2,:],pts[2:,:]
        # calculate the angle between each point

        cos_values=[]
        num=pts.shape[0]
        for idx in range(num):
            ii=idx
            vec1=(pts[(ii+1)%num,:]-pts[ii,:])
            vec2=(pts[(ii+num-1)%num]-pts[ii,:])
            cos_value=(vec1*vec2).sum()/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-6)
            cos_value_0=np.abs(cos_value)
            
            ii=(idx+1)%num
            vec1=(pts[(ii+1)%num,:]-pts[ii,:])
            vec2=(pts[(ii+num-1)%num]-pts[ii,:])
            cos_value=(vec1*vec2).sum()/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-6)
            cos_value=np.abs(cos_value)

            cos_value=(cos_value+cos_value_0)/2
            cos_values.append(cos_value)
        # find the 3 min value
        cos_val=np.array(cos_values)

        index_chooses=[(0,1),(0,2),(0,3),(1,2),(1,3)]
        for idx,idx_choos in enumerate(index_chooses):
            min_index=np.argsort(cos_val)[idx_choos,]
            min_index_clock=np.sort(min_index)
            if not (abs(min_index_clock[0] - min_index_clock[1]) == 1 or abs((min_index_clock[0] - min_index_clock[1] + num) % num) == 1):
                break
            if idx==len(index_chooses)-1:
                return None,None

        min_index_clock=np.array((min_index_clock[0],(min_index_clock[0]+1)%num,min_index_clock[1],(min_index_clock[1]+1)%num),dtype=np.int32)

        min_index_final=min_index_clock[(1,2,3,0),]
        # corner_point=pts[min_index_clock,:]

        # if np.linalg.norm(corner_point[0]-corner_point[1])>np.linalg.norm(corner_point[2]-corner_point[3]):
        #     min_index_final=min_index_clock
        # else:
        #     min_index_final=min_index_clock[(1,2,3,0),]

        split_0,split_1=[],[]
        iidx,iidx_e=min_index_final[0],min_index_final[1]
        while iidx!=iidx_e:
            split_0.append(pts[iidx,:])
            iidx=(iidx+1)%num
        split_0.append(pts[iidx_e,:])
        
        iidx,iidx_e=min_index_final[2],min_index_final[3]
        while iidx!=iidx_e:
            split_1.append(pts[iidx,:])
            iidx=(iidx+1)%num
        split_1.append(pts[iidx_e,:])
        
        # the clockwise vec x coord is greater to 0
        if (split_0[-1]-split_0[0])[0]>0:
            left_up=split_0
            right_down=split_1
        else:
            left_up=split_1
            right_down=split_0
            
        return np.array(left_up),np.array(right_down)

    def expand_point(self,pts,num_exp=7):
        sh=pts.shape
        if sh[0]<4:
            return None

        up_point,down_point=self.spilt_point(pts)
        if up_point is None:
            return None

        def expand(point_base):
            #  all point must in clock or unclockwise
            dis=np.linalg.norm(point_base[1:,:]-point_base[:-1,:],axis=1)
            up_len=dis.sum()
            dis_per=max(up_len/(num_exp-1),1)

            line_seg=[0]
            for i in range(dis.shape[0]):
                line_seg.append(line_seg[i]+dis[i])
            
            line_seg=np.array(line_seg)

            exp_list=[]
            for i in range(num_exp):
                cur_pos=dis_per*i
                
                dis=line_seg-cur_pos
                index=np.argsort(np.abs(line_seg-cur_pos))
                if dis[index[0]]*dis[index[1]]<0:
                    a_idx,b_idx=index[0],index[1]
                elif len(dis)>2:
                    a_idx,b_idx=index[0],index[2]
                else:
                    a_idx,b_idx=index[0],index[1]

                point_exp=(point_base[a_idx,:]-point_base[b_idx,:])*(cur_pos-line_seg[b_idx])/(line_seg[a_idx]-line_seg[b_idx]+1e-6)+point_base[b_idx,:]

                exp_list.append(point_exp)
            
            return exp_list
        
        up_expand=expand(up_point)
        down_expand=expand(down_point)
        
        point_expand=np.array(up_expand+down_expand)
        return point_expand

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": 768, "width": 1280}

if __name__ == "__main__":
    TotalText('./datasets/TotalText',False)
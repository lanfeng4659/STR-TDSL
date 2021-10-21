import logging
import tempfile
import os
import torch
from collections import OrderedDict
import itertools
from tqdm import tqdm
import json
# from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

# from maskrcnn_benchmark.data.datasets.bezier import BEZIER
from torch import nn
from maskrcnn_benchmark.config import cfg
from shapely.geometry import *
import cv2
import numpy as np
from scipy.special import comb as n_over_k
from sklearn.metrics import average_precision_score
class GlobalLocalSimilarity(nn.Module):
    
    def __init__(self,divided_nums = [1,3,5]):
        super(GlobalLocalSimilarity, self).__init__()
        self.divided_nums = divided_nums
        self.normalize = nn.functional.normalize
    def compute_similarity(self,x,y,divided_num=1):
        x = x.view(x.size(0),divided_num,-1)
        y = y.view(y.size(0),divided_num,-1)
        sims = torch.stack([self.normalize(x[:,i,:]).mm(self.normalize(y[:,i,:]).t()) for i in range(divided_num)],dim=-1)
        return sims.mean(dim=-1)
        
    def forward(self, x,y):
        x_tanh = x.tanh()
        y_tanh = y.tanh()
        sims = torch.stack([self.compute_similarity(x_tanh, y_tanh, divided_num) for divided_num in self.divided_nums],dim=-1)
        return sims.mean(dim=-1)
class DynamicMaxSimilarity(nn.Module):
    
    def __init__(self,frame_num):
        super(DynamicMaxSimilarity, self).__init__()
        self.frame_num = frame_num
    # def sim(self,x,y):
    #     # print(x.shape)
    #     x_nor = torch.nn.functional.normalize(x.view(x.size(0),-1).tanh())
    #     y_nor = torch.nn.functional.normalize(y.view(y.size(0),-1).tanh())
    #     return x_nor.mm(y_nor.t())
    def sim(self,x,y):
        x_nor = torch.nn.functional.normalize(x.view(-1,x.size(-1)).tanh()) # x_bw,c
        y_nor = torch.nn.functional.normalize(y.view(-1,y.size(-1)).tanh()) # y_bw,c
        similarity = x_nor.mm(y_nor.t()) # (x_bw,y_bw)
        similarity = similarity.reshape([x.size(0),x.size(1),y.size(0),y.size(1)])
        return similarity.permute(0,2,1,3)
    def push_similarity(self,global_sim, local_sim, steps):
        return (global_sim*(steps-1)+local_sim)/steps
    def forward(self,a,b):
        si = torch.zeros([a.size(0),b.size(0),self.frame_num+1, self.frame_num+1]).type_as(a)
        local_similarity = self.sim(a,b)
        for i in range(1, self.frame_num+1):
            for j in range(1, self.frame_num+1):
                local_sim = local_similarity[:,:,i-1,j-1]
                all_sim = torch.stack([self.push_similarity(si[:,:,i-1,j], local_sim, max(i,j)), 
                                       self.push_similarity(si[:,:,i,j-1], local_sim, max(i,j)), 
                                       self.push_similarity(si[:,:,i-1,j-1], local_sim, max(i,j))]
                                       ,dim=-1)
                si[:,:,i,j] = torch.max(all_sim,dim=-1)[0]
        return si[:,:,-1,-1]
def iiit_retrieval_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    rec_type,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("Evaluating bbox proposals")
    mAP = evaluate_box_proposals(predictions, dataset,output_folder)
    return mAP
def filter_inner_box(polys):
    '''
    filter thoese containing box.
    First find box with overlap, the decide which to leave
    '''
    if polys.size==0:
        return polys
    polys=polys.reshape(-1,4,2)
    centers=polys.mean(1)

    remove_flag=np.zeros((centers.shape[0]))
    for idx,cp in enumerate(centers):
        contain_status=[cv2.pointPolygonTest(poly.astype(np.int32),(cp[0],cp[1]),False) for poly in polys]
        idx_conts=np.where(np.array(contain_status)==True)[0]
        for id_c in idx_conts:
            if id_c!=idx:
                poly_m=polys[id_c]
                poly_c=polys[idx]

                # if one side of current poly is close to big poly
                # then remve this poly

                m_short=min(np.linalg.norm(poly_m[0]-poly_m[1]),np.linalg.norm(poly_m[1]-poly_m[2]))
                c_short=min(np.linalg.norm(poly_c[0]-poly_c[1]),np.linalg.norm(poly_c[1]-poly_c[2]))

                if m_short/c_short>0.9 and m_short/c_short<1/0.9:
                    remove_flag[idx]=1
    keep_idx=np.where(remove_flag==False)
    polys=polys[keep_idx,:,:]
    return polys.reshape(-1,4,2)
def write_to_file(bboxes, filename):
    ''' the socres is the average score of boundingbox region
    '''
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        line = "{},{},{},{},{}\r\n".format(bbox[0],bbox[1],bbox[2],bbox[3],"None")

        lines.append(line)
    
    with open(filename,'w') as f:
        for line in lines:
            f.write(line)
    # print(filename)
def show_detection(boxes, polys, path, output_folder):
    import cv2
    # boxes = bbox.data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2])
    # polys = polys.data.cpu().numpy()
    img_save_path = os.path.join(output_folder, os.path.basename(path))
    image = cv2.imread(path)
    cv2.drawContours(image, boxes.astype(np.int32), -1, color=(255,0,0), thickness=2)
    # cv2.drawContours(image, polys.astype(np.int32), -1, color=(0,0,255), thickness=2)
    cv2.imwrite(img_save_path, image)
    # print(img_save_path)
# inspired from Detectron
def meanAP(preds, trues):
    APs = []
    for y_scores, y_trues in zip(preds, trues):
        AP = average_precision_score(y_trues, y_scores)
        APs.append(AP)
    return APs
def re_ranking(retrieval_texts_embedding,predictions,y_trues):
    y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):
        for image_id, prediction in enumerate(predictions):
            img_embedding = prediction.get_field("imgs_embedding_nor")
            if img_embedding.size(0)==0:
                score = 0
                box = [0,0,0,0]
            else:
                similarity = embedding.mm(img_embedding.t())
                score,box_idx =  similarity.max(dim=1)
                # print(similarity.min(dim=1))
                score = score.data.cpu().numpy()[0]
                # k = min(img_embedding.size(0),2)
                # print(similarity.shape)
                # score = torch.topk(similarity,k,dim=1)[0].mean().data.cpu().numpy()
                box_idx = box_idx.data.cpu().numpy()[0]
                box = prediction.get_field("boxes")[box_idx].data.cpu().numpy()

            y_scores[idx,image_id] = score
    APs = meanAP(y_scores, y_trues)
    # show_aps(APs)
    # print(sum(APs)/len(APs))
    # y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)

    APs = meanAP(y_scores+y_scores2, y_trues)
    # print(sum(APs)/len(APs))
    return y_scores
def evaluate_box_proposals(
    predictions, dataset,output_folder, thresholds=0.23, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # img_floder = os.path.join(output_folder,'images')
    # txt_floder = os.path.join(output_folder,'texts')
    # for folder in [img_floder, txt_floder]:
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    y_trues = dataset.dataset.y_trues
    words_embedding_nor = None
    y_scores = []
    retrieval_image_embedding={}
    dms = DynamicMaxSimilarity(15)
    gls = GlobalLocalSimilarity()
    for image_id, prediction in enumerate(predictions):
        # print(prediction.fields())
        if "words_embedding_nor" in prediction.fields():
            words_embedding_nor = prediction.get_field("words_embedding_nor")
        boxes = prediction.bbox.data.cpu().numpy()
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        img_embedding = prediction.get_field("imgs_embedding_nor")
        if img_embedding.size(0)==0:
            y_scores.append(torch.zeros([words_embedding_nor.size(0)]).to(img_embedding.device))
            continue
        if img_embedding.dim()==3:
            similarity = dms(words_embedding_nor,img_embedding)
        else:
            similarity =  words_embedding_nor.mm(img_embedding.t())
            # similarity =  gls(words_embedding_nor, img_embedding)
        # score,box_idx =  similarity.max(dim=1)
        # # print(similarity.min(dim=1))
        # score = score.data.cpu().numpy()[0]
        # # k = min(img_embedding.size(0),2)
        # # print(similarity.shape)
        # # score = torch.topk(similarity,k,dim=1)[0].mean().data.cpu().numpy()
        # box_idx = box_idx.data.cpu().numpy()[0]
        # box = prediction.get_field("boxes")[box_idx].data.cpu().numpy()
        # if score>max_score:
        #     max_score = score
        #     retrieval_image_embedding[text]=img_embedding[box_idx].view(1,-1)
        y_scores.append(similarity.max(dim=1)[0])
    # import ipdb;ipdb.set_trace()
    y_scores = torch.stack(y_scores,dim=1).data.cpu().numpy()
    APs = meanAP(y_scores, y_trues)
    mAP = sum(APs)/len(APs)

    # y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)
    # print(mAP)
    # np.save(os.path.join(output_folder,"retrieval_results.npy"), retrieval_results)
    return mAP

def evaluate_box_proposals1(
    predictions, dataset,output_folder, thresholds=0.23, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    gt_overlaps = []
    num_pos = 0
    img_floder = os.path.join(output_folder,'images')
    txt_floder = os.path.join(output_folder,'texts')
    for folder in [img_floder, txt_floder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    retrieval_texts_embedding = {}
    for image_id, prediction in enumerate(predictions):
        words_embedding = prediction.get_field("words_embedding_nor")
        image_name = os.path.basename(str(prediction.get_field("path")))
        filename = os.path.join(txt_floder, image_name).replace(".jpg",'.txt')
        # print(prediction.data.cpu().numpy().shape)
        boxes = prediction.bbox.data.cpu().numpy()
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        write_to_file(boxes, filename)
        if words_embedding.size(0) == 0:
            continue
        for idx,text in enumerate(prediction.get_field("texts")):
            text = text.lower()
            # print(retrieval_texts_embedding.keys())
            if text in retrieval_texts_embedding.keys():
                continue
            retrieval_texts_embedding[text] = words_embedding[idx,:].reshape(1,-1)
    y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    y_trues = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    print(y_scores.shape)
    for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):

        for image_id, prediction in enumerate(predictions):
            y_trues[idx,image_id] = 1 if text in [t.lower() for t in prediction.get_field("texts")] else 0
            img_embedding = prediction.get_field("images_embedding_nor")
            if img_embedding.size(1)==0:
                score = 0
            else:
                # score =  embedding.mm(img_embedding.t()).max(dim=1)[0].data.cpu().numpy()[0]
                score =  embedding.mm(img_embedding).max(dim=1)[0].data.cpu().numpy()[0]
            y_scores[idx,image_id] = score
    APs = meanAP(y_scores, y_trues)
    print(sum(APs)/len(APs))
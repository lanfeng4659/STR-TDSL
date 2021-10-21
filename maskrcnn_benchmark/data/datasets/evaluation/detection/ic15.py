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

from maskrcnn_benchmark.config import cfg
from shapely.geometry import *
import cv2
import numpy as np
from scipy.special import comb as n_over_k
from .ic15_scripts.script import eval_s,eval_rects
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']

def ic15_detection_evaluation(
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
    evaluate_box_proposals(predictions, dataset,output_folder)
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
def write_to_file(bboxes, path, output_dir):
    ''' the socres is the average score of boundingbox region
    '''

    filename = os.path.join(output_dir, 'res_%s.txt' % (os.path.basename(path).replace(".jpg","")))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox.reshape([-1])]
        # if words is not None:
        #     line = "%d, %d, %d, %d, %d, %d, %d, %d" % tuple(values)
        #     line = line+','+words[b_idx]+'\r\n'
        # elif scores is None:
        line = "%d, %d, %d, %d, %d, %d, %d, %d\r\n" % tuple(values)
        # else:
        #     values.append(scores[b_idx])
        #     line = "%d, %d, %d, %d, %d, %d, %d, %d,%f\r\n" % tuple(values)
        lines.append(line)
    
    with open(filename,'w') as f:
        for line in lines:
            f.write(line)
    print(filename)
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
def evaluate_box_proposals(
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

    for image_id, prediction in enumerate(predictions):
        info = dataset.get_img_info(image_id)

        # TODO replace with get_img_info?
        path = info["path"]
        image_width = info["width"]
        image_height = info["height"]
        # print(path, image_width, image_height)
        # continue
        prediction = prediction.resize((image_width, image_height))
        ratio_width, ratio_height = prediction.get_field("ratios")
        # objectness = prediction.get_field("labels")
        scores = prediction.get_field("scores")
        polys = prediction.get_field("polys").view(-1,4,2)
        polys[:,:,0] *= ratio_width
        polys[:,:,1] *= ratio_height
        # print(polys.shape)
        # centerness = objectness*scores
        pos_idxs = torch.nonzero(scores>thresholds).view(-1)
        bbox = prediction.bbox[pos_idxs,:].data.cpu().numpy()[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2])
        polys = polys[pos_idxs,:].data.cpu().numpy()
        # polys = filter_inner_box(polys)
        # print(bbox.shape)
        show_detection(bbox, polys, path, img_floder)
        write_to_file(polys, path, txt_floder)
    cmd='cd %s;zip -j detect.zip texts/*'%output_folder
    import subprocess
    subprocess.check_output(cmd, shell = True)

    abs_path=os.path.abspath(output_folder)
    para = {'g': './maskrcnn_benchmark/data/datasets/evaluation/detection/ic15_scripts/gt.zip',
            's': os.path.join(abs_path, 'detect.zip'),
            'o': abs_path}
    func_name = 'eval_s(para)'
    try:
        res = eval(func_name)
    except:
        print('eval error!')
        # os.chdir('../')
    with open(os.path.join(abs_path, 'result.json'), 'w') as f:
        json.dump(res, f, indent=2)


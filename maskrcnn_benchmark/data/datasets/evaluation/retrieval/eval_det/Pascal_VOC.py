import os
from os import listdir
from scipy import io
import numpy as np
import re
from tqdm import tqdm
from .polygon_fast import iou
from .polygon_fast import iod
import cv2

"""
Input format: y0,x0, ..... yn,xn. Each detection is separated by the end of line token ('\n')'
"""

def transcription_match(transGt,transDet,specialCharacters=str('!?.:,*"()·[]/\''),onlyRemoveFirstLastCharacterGT=False):

    if onlyRemoveFirstLastCharacterGT:
        #special characters in GT are allowed only at initial or final position
        if (transGt==transDet):
            return True        

        if specialCharacters.find(transGt[0])>-1:
            if transGt[1:]==transDet:
                return True

        if specialCharacters.find(transGt[-1])>-1:
            if transGt[0:len(transGt)-1]==transDet:
                return True

        if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1:
            if transGt[1:len(transGt)-1]==transDet:
                return True
        return False
    else:
        transGt=re.sub('[!?.:,*"()·[]/\']','',transGt).lower()
        transDet=re.sub('[!?.:,*"()·[]/\']','',transDet).lower().strip()
            
        return transGt == transDet

def eval_result(input_dir,allInputs,gt_floder,ignore_difficult=True):
    # allInputs = listdir(input_dir)
    # f = open(gt_list)
    # lines = f.readlines()
    # allInputs = [line.strip().split(',')[1] for line in lines]

    def input_reading_mod(path):
        """This helper convert input"""
        if os.path.exists(path)==False:
            return []
        with open(path, 'r') as input_fid:
            lines = input_fid.readlines()
        dts = []
        for line in lines:
            parts = line.strip().split(',')[:-1]
            pts = list(map(float, parts))
            # print(np.array(pts).shape)
            pts = list([pts[0],pts[1],pts[2],pts[1],pts[2],pts[3],pts[0],pts[3]])
            dts.append(pts)
        return dts
    # def input_reading_mod(path):
    #     """This helper convert input"""
    #     if os.path.exists(path)==False:
    #         return []
    #     with open(path, 'r') as input_fid:
    #         lines = input_fid.readlines()
    #     dts = []
    #     for line in lines:
    #         parts = line.strip().split(' ')
    #         pts = list(map(float, parts[:-1]))
    #         dts.append(pts)
    #     return dts

    def gt_reading_mod(path):
        with open(path, 'r') as input_fid:
            lines = input_fid.readlines()
        gts = []
        for line in lines:
            parts = line.strip().split(',')
            label = parts[-1]
            # print(parts)
            # pts = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts[:-1]]
            pts = list(map(float, parts[:-1]))
            pts = list([pts[0],pts[1],pts[2],pts[1],pts[2],pts[3],pts[0],pts[3]])
            # pts.append()
            gts.append([pts,label])
        return gts
    def gt_reading_mod_ic15(path):
        with open(path, 'r') as input_fid:
            lines = input_fid.readlines()
        gts = []
        for line in lines:
            # parts = line.strip().split(' ')
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            label = parts[-1]
            # print(parts)
            # pts = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts[:-1]]
            pts = list(map(float, parts[:8]))
            # pts.append()
            gts.append([pts,label])
        return gts

    def detection_filtering(detections, groundtruths, threshold=0.5):
        dcGTs = [gt for gt in groundtruths if gt[1] == '###']
        for gt_id, gt in enumerate(dcGTs):
            coords = gt[0]
            gt_x = list(map(int, coords[::2]))
            gt_y = list(map(int, coords[1::2]))
            delete_flag = [False]*len(detections)
            for det_id, detection in enumerate(detections):
                detection = list(map(int, detection))
                det_x = detection[0::2]
                det_y = detection[1::2]
                delete_flag[det_id] = iod(det_x, det_y, gt_x, gt_y) > threshold
            detections = [det for flag, det in zip(delete_flag, detections) if flag==False]
        return detections

    global_tp = 0
    global_tp_rec = 0
    global_fp = 0
    global_fn = 0
    global_num_of_gt = 0
    global_num_of_det = 0

    for input_id in tqdm(allInputs):
        if (input_id != '.DS_Store'):
            detections = input_reading_mod(os.path.join(input_dir, input_id))
            groundtruths = gt_reading_mod(os.path.join(gt_floder, input_id))
            # groundtruths = gt_reading_mod(os.path.join(gt_floder, input_id.replace("res_","gt_")))
            if ignore_difficult:
                detections = detection_filtering(detections, groundtruths) #filtering detections overlaps with DC area
                groundtruths = [gt for gt in groundtruths if gt[1]!="###"]
            global_num_of_gt = global_num_of_gt + len(groundtruths)
            global_num_of_det = global_num_of_det + len(detections)
            iou_table = np.zeros((len(detections), len(groundtruths)))
            det_flag = np.zeros((len(detections), 1))
            gt_flag = np.zeros((len(groundtruths), 1))
            tp = 0
            fp = 0
            fn = 0

            if len(detections) > 0:
                for det_id, detection in enumerate(detections):
                    det_x = detection[0::2]
                    det_y = detection[1::2]
                    # print(detection)
                    # det_x = detection[(0,2,2,0)]
                    # det_y = detection[(1,1,3,3)]
                    if len(groundtruths) > 0:
                        for gt_id, gt in enumerate(groundtruths):
                            gt_x = gt[0][::2]
                            gt_y = gt[0][1::2]
                            # gt_x = gt[0][(0,2,2,0)]
                            # gt_y = gt[0][(1,1,3,3)]
                            iou_table[det_id, gt_id] = iou(det_x, det_y, gt_x, gt_y)
                        best_matched_gt_id = np.argmax(iou_table[det_id, :]) #identified the best matched detection candidates with current groundtruth
                        if (iou_table[det_id, best_matched_gt_id] > 0.6):
                            if gt_flag[best_matched_gt_id] == 0: ### if the gt is already matched previously, it should be a false positive
                                tp = tp + 1.0
                                global_tp = global_tp + 1.0
                                gt_flag[best_matched_gt_id] = 1
                            else:
                                fp = fp + 1.0
                                global_fp = global_fp + 1.0
                        else:
                            fp = fp + 1.0
                            global_fp = global_fp + 1.0

            try:
                local_precision = tp / (tp + fp)
            except ZeroDivisionError:
                local_precision = 0

            try:
                local_recall = tp / len(groundtruths)
            except ZeroDivisionError:
                local_recall = 0

            # fid = open(fid_path, 'a')
            temp = ('%s______/Precision:_%s_______/Recall:_%s\n' %(input_id, str(local_precision), str(local_recall)))

    global_precision = global_tp / global_num_of_det
    global_recall = global_tp / global_num_of_gt
    f_score = 2*global_precision*global_recall/(global_precision+global_recall)

    print(('P: %s /R:%s' %(str(global_precision), str(global_recall))),'F: ',f_score)
    
    return  {'precison':global_precision,'recall':global_recall,'hmean':f_score,\
             'global_tp':global_tp,'global_num_of_det':global_num_of_det,\
             'global_num_of_gt':global_num_of_gt}  

# def eval()
# if __name__ =='__main__':
    # eval_category('/data/wanghao/projects/scene_text/EAST/Log',
    #     '/data/wanghao/datasets/scene_text/lists/new_test_list.txt',
    #     '/data/wanghao/datasets/scene_text/v1/',
    #     ignore_difficult=True)
    # eval_category('/data/wanghao/projects/scene_text/EAST/Log',
    #     '/data/wanghao/datasets/scene_text/lists/new_test_list.txt',
    #     '/data/wanghao/datasets/scene_text/v1/',
    #     ignore_difficult=False)
    # eval_result('/data/wanghao/projects/scene_text/maskrcnn/Log/use_gn_n/inference/icdar15_test/txt_result',
    #     '/data/wanghao/datasets/scene_text/lists/new_test_list.txt',
    #     '/data/wanghao/projects/scene_text/maskrcnn/maskrcnn_benchmark/data/datasets/evaluation/icdar/gts',
    #     True)
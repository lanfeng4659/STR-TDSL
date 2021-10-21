import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
import json

# from .script import eval_s
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
# from .Pascal_VOC import eval_category
def select_top_predictions( predictions,confidence_threshold=0.7):
    """
    Select only predictions which have a `score` > confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def rect_to_xys(rect, image_shape,ratio):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]

    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x

    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y

    # rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])

    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x*ratio[0])
        y = get_valid_y(y*ratio[1])
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

def mask_to_box(mask, image_size, min_height=5, min_area=200):
    h,w,_=image_size
    try:
        _, contours, _ = cv2.findContours(
            mask.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(
            mask.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    # choose biggest contours as the text region, recall increase
    area=0
    result=None
    for bbox_contours in contours:
        rect = cv2.minAreaRect(bbox_contours)
        rect = list(rect)
        sw, sh = rect[1][:]
        # if min(sw, sh) < min_height:
        #     xys=None
        # if sw*sh < min_area:
        #     xys=None
        # if max(sw, sh) * 1.0 / min(sw, sh) < 2:
        #    return None
        rect = tuple(rect)
        xys = rect_to_xys(rect, [h, w],(1,1))
        if sw*sh>area:
            area=sw*sh
            result=xys
    return result
def mask_to_contours(mask, image_size, min_height=5, min_area=200):
    h,w,_=image_size
    try:
        _, contours, _ = cv2.findContours(
            mask.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(
            mask.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    # new_contours = []
    # for contour in contours:
    #     coefficient = .01
    #     epsilon = coefficient * cv2.arcLength(contour, True)
    #     poly_approx = cv2.approxPolyDP(contour, epsilon, True)
    #     new_contours.append(poly_approx)
        # pts = poly_approx.reshape([-1]).tolist()
    return contours
def contours_to_box(contours):
    # choose biggest contours as the text region, recall increase
    area=0
    result=None
    for bbox_contours in contours:
        rect = cv2.minAreaRect(bbox_contours)
        rect = list(rect)
        sw, sh = rect[1][:]
        # if min(sw, sh) < min_height:
        #     xys=None
        # if sw*sh < min_area:
        #     xys=None
        # if max(sw, sh) * 1.0 / min(sw, sh) < 2:
        #    return None
        rect = tuple(rect)
        xys = rect_to_xys(rect, [h, w],(1,1))
        if sw*sh>area:
            area=sw*sh
            result=xys
    return result

def write_to_file(bboxes, gt_name, output_dir,scores=None):
    ''' the socres is the average score of boundingbox region
    '''
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
    # bboxes = bboxes.tolist()
    # save to file
    filename = os.path.join(output_dir, gt_name)
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        # print(bbox)
        values = [int(v) for v in bbox.reshape([-1])]
        str_write = ""
        for v in values[:-1]:
            str_write += str(v)
            str_write += ","
        str_write += str(values[-1])
        str_write += "\r\n"
        # if scores is None:
        #     line = "%d, %d, %d, %d, %d, %d, %d, %d\r\n" % tuple(values)
        # else:
        #     values.append(scores[b_idx])
        #     line = "%d, %d, %d, %d, %d, %d, %d, %d,%f\r\n" % tuple(values)
        lines.append(str_write)
    
    with open(filename,'w') as f:
        for line in lines:
            # print(line)
            f.write(line)

def wirte_file(image_shape,image_name,predictions,output_dir):
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    polys=np.empty((0,8))
    # NOTE filter hard example
    for mask_label in zip(masks,labels):
        mask=mask_label[0]
        label=mask_label[1]
        if label==1:
            poly=mask_to_box(mask[0,:,:,None],image_shape)
            if poly is not None:
                polys = np.concatenate((polys, poly[np.newaxis, :]), axis=0)
    
    image_name=os.path.basename(image_name).split('.')[0]

    infer_path=os.path.join(output_dir,'txt_result')
    if os.path.isdir(infer_path)==False:
        os.mkdir(infer_path)
    # txt_path=os.path.join(infer_path,'txt_result')
    # if os.path.isdir(txt_path)==False:
    #     os.mkdir(txt_path)
    
    write_to_file(polys,image_name,infer_path)
def get_contours(image_shape,predictions):
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    contours = []
    # import ipdb; ipdb.set_trace()
    for mask_label in zip(masks,labels):
        mask=mask_label[0]
        label=mask_label[1]
        if label==1:
            contours_same_proposal=mask_to_contours(mask[0,:,:,None],image_shape)
            if len(contours_same_proposal)==0:
                continue
            areas = [cv2.contourArea(contour) for contour in contours_same_proposal]            
            contours.append(contours_same_proposal[np.argmax(areas)])
    return contours
            

def scenetext_evaluation(dataset,predictions,output_folder,step,**kwargs):
    '''
        predictions are dict {'image_id':prediction, ....}
    '''
    masker = Masker(threshold=0.5, padding=1)
    for img_id, prediction in enumerate(tqdm(predictions)):
        prediction=select_top_predictions(prediction)
        # predictions[img_id]=pred
        _,target,idx=dataset[img_id]
        img_path, gt_path=dataset._img_txt_list[idx]
        gt_name = os.path.basename(gt_path)
        gt_floder = gt_path.replace('datasets/scene_text/v1/',os.path.join(output_folder,"txt_results/")).replace(gt_name,"")
        
        image = cv2.imread(img_path)
        height, width = target.get_field('image_size').numpy()[::-1]
        prediction = prediction.resize((width, height))
        contours = []
        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
            contours = get_contours((height,width,3),prediction)
            contours = [contour.reshape([-1]) for contour in contours]
        write_to_file(contours, gt_name, gt_floder,scores=None)

        #     for contour in contours:
        #         contour = np.array(contour).reshape([1,-1,2]).astype(np.int32)
        #         image = cv2.drawContours(image, contour, -1, color=(255,0,0), thickness=3)
        # save_path = os.path.join("/data/visual", os.path.basename(img_path))
        # cv2.imwrite(save_path, image)
        # print(save_path)
            # print(contours)

        # wirte_file((height,width,3),dataset._imgpath_list[img_id],prediction,output_folder)
    return None
    # cmd='cd %s;zip -j detect.zip txt_result/*'%output_folder
    # import subprocess
    # subprocess.check_output(cmd, shell = True)

    # abs_path=os.path.abspath(output_folder)
    # para = {'g': './maskrcnn_benchmark/data/datasets/evaluation/ic15/gt.zip',
    #         's': os.path.join(abs_path, 'detect.zip'),
    #         'o': abs_path}
    # # os.chdir('./maskrcnn_benchmark/data/datasets/evaluation/ic15/')
    
    
    # func_name = 'eval_s(para)'
    # try:
    #     res = eval(func_name)
    # except:
    #     print('eval error!')
    #     # os.chdir('../')
    # with open(os.path.join(abs_path, 'result_%s.json'%step), 'w') as f:
    #     json.dump(res, f, indent=2)

    # return res
    

    

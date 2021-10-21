import numpy as np
from numpy import random
import cv2
def random_crop(image, boxes = None, tags = None,poly_fixed_points=True, crop_size=(640, 640), max_tries = 50):
    crop_h, crop_w = crop_size
    ori_h, ori_w, c = image.shape
    start_x_max = np.clip(ori_w - crop_w, 0, ori_w)
    start_y_max = np.clip(ori_h - crop_h, 0, ori_h)
    new_w = crop_w if start_x_max == 0 else ori_w
    new_h = crop_h if start_y_max == 0 else ori_h
    new_image = np.zeros((new_h, new_w, c))
    new_image[0:ori_h, 0:ori_w, :] = image
    if start_x_max==0 and start_y_max==0:
        return new_image, boxes, tags
    #ensure the croped area is not in box
    h_array = np.zeros((new_h), dtype=np.int32)
    w_array = np.zeros((new_w), dtype=np.int32)
    for box in boxes:
        box = np.round(box, decimals=0).astype(np.int32)
        minx = np.min(box[:, 0])
        maxx = np.max(box[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(box[:, 1])
        maxy = np.max(box[:, 1])
        h_array[miny:maxy] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        h , w, c = new_image.shape
        new_image = cv2.resize(new_image, (crop_w, crop_h))
        if poly_fixed_points==False:
            boxes = [box*[crop_w*1.0/w,crop_h*1.0/h] for box in boxes]
        else:
            boxes[:,:,0] *= crop_w*1.0/w
            boxes[:,:,1] *= crop_h*1.0/h
        return new_image, boxes, tags
    for i in range(max_tries):
        rand_start_x = random.randint(0, start_x_max+1)
        rand_end_x = rand_start_x + crop_w
        rand_start_y = random.randint(0, start_y_max+1)
        rand_end_y = rand_start_y + crop_h
        x_valid = rand_start_x in w_axis and rand_end_x in w_axis
        y_valid = rand_start_y in h_axis and rand_end_y in h_axis
        if not (x_valid and y_valid):
            continue
        if len(boxes) != 0:
            if poly_fixed_points==False:
                selected_boxes = []
                for idx, box in enumerate(boxes):
                    box_axis_in_area = (box[:, 0] >= rand_start_x) & (box[:, 0] <= rand_end_x) \
                                & (box[:, 1] >= rand_start_y) & (box[:, 1] <= rand_end_y)
                    if np.sum(box_axis_in_area) == box.shape[0]:
                        selected_boxes.append(idx)
            else:
                box_axis_in_area = (boxes[:, :, 0] >= rand_start_x) & (boxes[:, :, 0] <= rand_end_x) \
                                & (boxes[:, :, 1] >= rand_start_y) & (boxes[:, :, 1] <= rand_end_y)
                selected_boxes = np.where(np.sum(box_axis_in_area, axis=1) == boxes.shape[1])[0]
            if len(selected_boxes) > 0:
                break
        else:
            selected_boxes = []
            continue
    if i == max_tries-1:
        h , w, c = new_image.shape
        new_image = cv2.resize(new_image, (crop_w, crop_h))        
        if poly_fixed_points==False:
            boxes = [box*[crop_w*1.0/w,crop_h*1.0/h] for box in boxes]
        else:
            boxes[:,:,0] *= crop_w*1.0/w
            boxes[:,:,1] *= crop_h*1.0/h
        return new_image, boxes, tags
    cropped_image = new_image[rand_start_y:rand_end_y, rand_start_x:rand_end_x,:]
    # boxes = boxes[selected_boxes]
    # tags = tags[selected_boxes]
    
    if poly_fixed_points==False:
        boxes = [boxes[idx] for idx in selected_boxes]
        tags = [tags[idx] for idx in selected_boxes]
        boxes = [box-[rand_start_x,rand_start_y] for box in boxes]
    else:
        boxes = boxes[selected_boxes]
        tags = tags[selected_boxes]
        boxes[:,:,0] -= rand_start_x
        boxes[:,:,1] -= rand_start_y
    return cropped_image, boxes, tags

def random_ratio_scale(image, boxes = None, tags = None,poly_fixed_points=True, ratios = np.arange(0.8,1.3,0.1)):
    rand_index = random.randint(0, len(ratios)-1)
    ratio = ratios[rand_index]
    ori_h, ori_w, c = image.shape
    new_h = int(ori_w * ratio)
    ratio_h = new_h*1.0 / ori_h
    new_image = cv2.resize(image, (ori_w, new_h))
    if poly_fixed_points==False:
        boxes = [box*[1,ratio_h] for box in boxes]
    else:
        boxes[:,:,1]*= ratio_h
    return new_image, boxes, tags
def random_resize(image, boxes = None, tags = None,poly_fixed_points=True, longer_sides=np.arange(640,2592, 32)):
    rand_index = random.randint(0, len(longer_sides)-1)
    longer_side = longer_sides[rand_index]
    ori_h, ori_w, c = image.shape
    if ori_h > ori_w:
        ratio_h = longer_side*1.0/ori_h
        # boxes[:,:,1]*= ratio_h
        if poly_fixed_points==False:
            boxes = [box*[1,ratio_h] for box in boxes]
        else:
            boxes[:,:,1]*= ratio_h
        new_image = cv2.resize(image, (ori_w, longer_side))
    else:
        ratio_w = longer_side*1.0/ori_w
        if poly_fixed_points==False:
            boxes = [box*[ratio_w,1] for box in boxes]
        else:
            # print(boxes)
            boxes[:,:,0]*= ratio_w
        # boxes[:,:,0]*= ratio_w
        new_image = cv2.resize(image, (longer_side, ori_h))
    return new_image, boxes, tags
def random_rotate(image, boxes = None, tags = None,poly_fixed_points=True,rotate_angles = np.arange(-10,10,1)):
    rand_index = random.randint(0, len(rotate_angles)-1)
    angle = rotate_angles[rand_index]
    ori_h, ori_w, _ = image.shape
    cX, cY = ori_w//2, ori_h//2
    matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = matrix[0,0], -matrix[0,1]
    abs_cos, abs_sin = np.abs(cos), np.abs(sin)
    nW = int((ori_h * abs_sin) + (ori_w * abs_cos))
    nH = int((ori_h * abs_cos) + (ori_w * abs_sin))
    matrix[0, 2] += (nW / 2) - cX
    matrix[1, 2] += (nH / 2) - cY
    new_image = cv2.warpAffine(image, matrix, (nW, nH))
    
    if poly_fixed_points==False:
        boxes = [box-[cX,cY] for box in boxes]
        new_boxes = []
        for box in boxes:
            temp_box = np.zeros_like(box)
            temp_box[:,0] = (box[:,0]*cos - box[:,1]*sin) + nW/2
            temp_box[:,1] = (box[:,1]*cos + box[:,0]*sin) + nH/2
            new_boxes.append(temp_box)
    else:
        boxes -= np.array([cX, cY])
        new_boxes = np.zeros_like(boxes)
        new_boxes[:,:,0] = (boxes[:,:,0]*cos - boxes[:,:,1]*sin) + nW/2
        new_boxes[:,:,1] = (boxes[:,:,1]*cos + boxes[:,:,0]*sin) + nH/2
    return new_image, new_boxes, tags
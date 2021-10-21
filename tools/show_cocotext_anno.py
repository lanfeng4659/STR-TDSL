import os
import torch
import numpy as np
import cv2
from scipy.misc import imread, imresize
import shutil
from PIL import Image,ImageDraw
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from evaluation.Pascal_VOC import eval_result
def meanAP(preds, trues):
    APs = []
    for y_scores, y_trues in zip(preds, trues):
        AP = average_precision_score(y_trues, y_scores)
        APs.append(AP)
    return APs
def vis_multi_image(image_list, shape=[1,-1]):
    image_num = len(image_list)
    h, w,_ = np.array(image_list[0]).shape
    #print h,w
    num_w = int(image_num/shape[0])
    num_h = shape[0]
    new_im = Image.new('RGB', (num_w*w,num_h*h))
    for idx, image in enumerate(image_list):
        idx_w = idx%num_w
        idx_h = int(idx/num_w)
        new_im.paste(image, (int(idx_w*w),int(idx_h*h)))
    return new_im
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
    show_aps(APs)
    print(sum(APs)/len(APs))
    return y_scores
def write_to_txt(boxes,filename):
    f = open(filename,mode='w')
    for box in boxes:
        line = "{},{},{},{}\n".format(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
        f.write(line)
# def find_repeat_queries(texts):

def evaluate_box_proposals(predictions, visul_path, text_path):
    retrieval_texts_embedding = {}
    y_trues = None
    for image_id, prediction in tqdm(enumerate(predictions)):
        
        # print(prediction)
        if "words_embedding_nor" in prediction.fields():
            words_embedding = prediction.get_field("words_embedding_nor")
        # if "y_trues" in prediction.fields():
        #     y_trues = prediction.get_field("y_trues")
            # import ipdb;ipdb.set_trace()
        boxes = prediction.bbox
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        prediction.add_field("boxes", boxes)
        img_embedding = prediction.get_field("imgs_embedding_nor")
        assert words_embedding.size(0) == len([text for text in prediction.get_field('texts')])
        if words_embedding.size(0) == 0:
            continue
        similarity = words_embedding.mm(img_embedding.t())
        texts = prediction.get_field('texts')
        score,box_idx =  similarity.max(dim=1)
        # print(box_idx.shape)
        score = score.data.cpu().numpy()[0]
        box_idx = box_idx.data.cpu().numpy()
        boxes = prediction.get_field("boxes")[box_idx].data.cpu().numpy()
        # print(words_embedding.shape,boxes.shape)
        path = str(prediction.get_field("path"))
        # print(path)
        image = imread(path,mode="RGB")
        for box in boxes:
            minx,miny,maxx,maxy = box
            box = np.array([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]]).reshape([-1,4,2]).astype(np.int32)
            cv2.drawContours(image, box, -1, color=(0,255,0),thickness=2)
        cv2.imwrite(os.path.join(visul_path,os.path.basename(path)),image[:,:,(2,1,0)])
        write_to_txt(boxes,os.path.join(text_path,os.path.basename(path.replace('.jpg','.txt'))))
        # print(prediction.fields())


def show_aps(aps):
    import matplotlib.pyplot as plt
    aps = sorted(aps)
    plt.plot(aps)
    plt.plot([sum(aps)/len(aps)]*len(aps))
    plt.savefig("aps.png")
def draw_boxes(path,score,box,y):
    # print(box)
    # image = cv2.imread(path,cv2.IMREAD_COLOR)
    image = imread(path,mode="RGB")
    # print(path)
    # print(image.shape)
    minx,miny,maxx,maxy = box
    # print(box)
    box = np.array([[minx,miny],[maxx,miny],[maxx,maxy],[minx,maxy]]).reshape([-1,4,2]).astype(np.int32)
    cv2.drawContours(image, box, -1, color=(0,255,0),thickness=2)
    cv2.putText(image,str(score),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,str(y),(100,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    cv2.putText(image,os.path.basename(path),(100,300),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),1)
    
    image = Image.fromarray(image).convert('RGB').resize((768, 640))
    return image
# save_path = "show_retrievals"
# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
# os.makedirs(save_path)
# predictions = torch.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/predictions.pth")
def packing(save_dir, pack_dir, pack_name):
    files = os.listdir(save_dir)
    if not os.path.exists(pack_dir):
        os.mkdir(pack_dir)
    os.system('zip -r -j -q '+os.path.join(pack_dir, pack_name+'.zip')+' '+save_dir+'/*')
if __name__ =='__main__':
    folder = "./Log/cocotext_anno/inference/cocotext_anno"
    visul_path = os.path.join(folder,'visual')
    text_path = os.path.join(folder,'texts')
    for path in [visul_path, text_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    predictions = torch.load(os.path.join(folder,'predictions.pth'))

    retrieval_results = evaluate_box_proposals(predictions, visul_path, text_path)
    # gt_floder = './datasets/cocotext_week_annotation_500/gts/'
    # for thred in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #     print(thred)
    #     eval_result(text_path,os.listdir(gt_floder),gt_floder,True,thred)
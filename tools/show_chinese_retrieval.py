import os
import torch
import numpy as np
import cv2
from scipy.misc import imread, imresize
import shutil
from PIL import Image,ImageDraw
from tqdm import tqdm
from sklearn.metrics import average_precision_score
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
def evaluate_box_proposals(predictions):
    retrieval_texts_embedding = {}
    y_trues = None
    for image_id, prediction in tqdm(enumerate(predictions)):
        
        # print(prediction)
        if "words_embedding_nor" in prediction.fields():
            words_embedding = prediction.get_field("words_embedding_nor")
        if "y_trues" in prediction.fields():
            y_trues = prediction.get_field("y_trues")
            # import ipdb;ipdb.set_trace()
        boxes = prediction.bbox
        scale = prediction.get_field("scale")
        boxes[:,::2] *= scale[0]
        boxes[:,1::2] *= scale[1]
        prediction.add_field("boxes", boxes)
        if words_embedding.size(0) == 0:
            continue
        for idx,text in enumerate(prediction.get_field("texts")):
            text = text.lower()
            retrieval_texts_embedding[text] = words_embedding[idx,:].reshape(1,-1)
    y_scores = np.zeros([len(retrieval_texts_embedding.keys()), len(predictions)])
    # y_trues = prediction.get_field("y_trues")
    retrieval_results = {}
    retrieval_image_embedding={}
    for idx,(text, embedding) in enumerate(retrieval_texts_embedding.items()):
        retrieval_results[text] = {}
        retrieval_image_embedding[text] = embedding
        max_score = 0
        for image_id, prediction in enumerate(predictions):
            img_embedding = prediction.get_field("imgs_embedding_nor")
            if img_embedding.size(0)==0:
                score = 0
                box = [0,0,0,0]
            else:
                similarity = embedding.mm(img_embedding.t())
                if "char_awareness" in prediction.fields():
                    char_awareness = prediction.get_field("char_awareness")
                    similarity = char_awareness[:,idx].view(-1,1).t()+similarity
                score,box_idx =  similarity.max(dim=1)
                # print(similarity.min(dim=1))
                score = score.data.cpu().numpy()[0]
                # k = min(img_embedding.size(0),2)
                # print(similarity.shape)
                # score = torch.topk(similarity,k,dim=1)[0].mean().data.cpu().numpy()
                box_idx = box_idx.data.cpu().numpy()[0]
                box = prediction.get_field("boxes")[box_idx].data.cpu().numpy()
            if score>max_score:
                max_score = score
                retrieval_image_embedding[text]=img_embedding[box_idx].view(1,-1)
            y_scores[idx,image_id] = score
            retrieval_results[text][str(prediction.get_field("path"))] = [score,box,y_trues[idx,image_id]]
        retrieval_results[text]["y_trues"] = sum(y_trues[idx])
        retrieval_results[text]["ap"] = meanAP(y_scores[idx].reshape([1,-1]), y_trues[idx].reshape([1,-1]))
    APs = meanAP(y_scores, y_trues)
    show_aps(APs)
    print(sum(APs)/len(APs))

    y_scores2 = re_ranking(retrieval_image_embedding,predictions,y_trues)

    APs = meanAP(y_scores+y_scores2, y_trues)
    print(sum(APs)/len(APs))
    return retrieval_results
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
save_path = "show_chinese_retrievals"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
# predictions = torch.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/predictions.pth")
if __name__ =='__main__':
    # predictions = torch.load("./Log/finetune_on_ic17_based_on_6853/inference/iiit_test/predictions.pth")
    predictions = torch.load("./Log/retrival_synthtext_chinese2/inference/chinese_collect/predictions.pth")
    retrieval_results = evaluate_box_proposals(predictions)
    np.save("temp.npy",retrieval_results)
    # retrieval_results = np.load("temp.npy",allow_pickle=True).item()
    # results = np.load("/root/data/projects/FCOSText/Log/retrival_e2e_add_retrieval_loss_10_no_tanh/inference/svt_test/retrieval_results.npy",allow_pickle=True).item()
    for text, datas in retrieval_results.items():
        # dict_ = {data.keys():data.values() for data in datas}
        y_trues = datas["y_trues"]
        ap = datas["ap"]
        del datas["y_trues"]
        del datas["ap"]
        sorted_paths = sorted(datas.items(), key=lambda datas:datas[1][0], reverse=True)
        image_list = [draw_boxes(x[0], x[1][0], x[1][1],x[1][2]) for x in sorted_paths[:20]]
        image = vis_multi_image(image_list, shape=[4,-1])
        image.save(os.path.join(save_path,"{}_{}_{}.jpg".format(text,y_trues, ap[0])))
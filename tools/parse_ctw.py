import jsonlines
import numpy as np
from tqdm import tqdm
import cv2
import os
import codecs
def show_box(image_path,boxes):
    image = cv2.imread(image_path)
    polys = boxes[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
    cv2.drawContours(image,polys,-1,color=(255,0,0),thickness=1)
    cv2.imwrite("show.jpg",image)
def is_chinese(uchar): #7185
    return '\u4e00' <= uchar <= '\u9fa5'
def is_alphabet(uchar): #7185
    return ('\u0041' <= uchar <= '\u005a') or ('\u0061' <= uchar <= '\u007a')
def is_number(uchar): #7185
    return ('\u0030' <= uchar <= '\u0039')
def filter_word(text):
    char_list = [c for c in text if (is_chinese(c) or is_number(c))]
    return "".join(char_list)
def load_ann(gt_paths,img_paths):
    res = []
    idxs = []
    chars = []
    for gt,img_path in zip(gt_paths,img_paths):
        # gt = unicode(gt, 'utf-8')#gt.decode('utf-8')
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['texts'] = []
        item['gt_path'] = gt
        item['img_path'] = img_path
        # print(gt)
        reader = codecs.open(gt,encoding='utf-8').readlines()
        # reader = open(gt).readlines()
        for line in reader:
            parts = line.strip().split(',')
            # print(parts)
            # assert len(parts)==10,(parts,"".join(parts[9:]))
            label = "".join(parts[9:])
            label = filter_word(label)
            if len(label)<3:
                continue
            if label == '###':
                continue
            for char in label:
                if char not in chars:
                    chars.append(char)
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label.lower())
        if len(item['polys'])==0:
            continue
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    # print(len(chars))
    # print(chars)
    #     print('read',item['polys'])
    # exit()
    return res,chars
class RCTW(object):
    def __init__(self, path, is_training = True):
        self.is_training = is_training
        self.difficult_label = '###'
        self.generate_information(path)
    def generate_information(self, path):
        if self.is_training:
            image_floder = os.path.join(path, 'train_images')
            gt_floder = os.path.join(path, 'train_gts')
            self.image_path_list = [os.path.join(image_floder, image) for image in os.listdir(image_floder)]
            gt_path_list    = [os.path.join(gt_floder, gt) for gt in os.listdir(gt_floder)]
            self.image_path_list = sorted(self.image_path_list)
            gt_path_list = sorted(gt_path_list)
            self.targets,self.chars = load_ann(gt_path_list, self.image_path_list)
            self.sample_num = len(self.targets)
    def len(self):
        return self.sample_num
    def getitem(self,index):
        if self.is_training:
            return self.targets[index]['img_path'], self.targets[index]['polys'].copy(), self.targets[index]['texts'].copy()
        else:
            return self.image_path_list[index], None, None
    def write_word_to_files(self,f):
        for target in self.targets:
            for word in target["texts"].copy():
                str_line = "{}\n".format(word)
                f.write(str_line)
class CTW(object):
    def __init__(self, path, is_training = True):
        # assert is_training==True
        self.is_training = is_training
        self.difficult_label = '###'
        self.all_texts = []
        self.parse_data(path)

    def parse_data(self,path):
        imageInstances = []
        image_path = os.path.join(path,"train_images")
        for anno_path in [os.path.join(path,"train.jsonl"),os.path.join(path,"val.jsonl")]:
            with open(anno_path, "r+", encoding="utf8") as f:
                
                for item in tqdm(jsonlines.Reader(f)):
                    instance_ = {}
                    # print(item["ignore"])
                    file_name,h,w = item["file_name"], item["height"],item["width"]
                    boxes=[]
                    texts=[]
                    # if not self.is_training:
                    #     print(item.keys())
                    for sentence in item["annotations"]:
                        text = "".join([char["text"] for char in sentence if char["is_chinese"]])
                        poly = np.array([char["polygon"] for char in sentence]).reshape([-1,2])
                        minx,miny,maxx,maxy = np.min(poly[:,0]),np.min(poly[:,1]),np.max(poly[:,0]),np.max(poly[:,1])
                        text = filter_word(text)
                        if len(text)<2:
                            continue
                        boxes.append([minx,miny,maxx,maxy])
                        texts.append(text)
                    if len(boxes)==0:
                        continue
                    instance_["path"] = os.path.join(image_path,file_name)
                    instance_["polys"] = np.array(boxes)[:,(0,1,2,1,2,3,0,3)].reshape([-1,4,2]).astype(np.int32)
                    instance_["texts"] = np.array(texts, dtype=np.str)
                    imageInstances.append(instance_)
        self.samples = imageInstances
    def count_word(self,):
        count_num = {}
        for sample in self.samples:
            for text in sample["texts"]:
                if text not in count_num.keys():
                    count_num[text] = 0
                count_num[text]+=1
        count_num = sorted(count_num.items(), key=lambda d: d[1],reverse=True)
        print(count_num[:100])
    def get_char_list(self,excepts=[]):
        chars = []
        for sample in self.samples:
            path,polys,texts = sample['path'],sample['polys'],sample['texts']
            if path in excepts:
                continue
            for text in texts:
                for char in text:
                    if char not in chars:
                        chars.append(char)
        print(len(chars))
        return chars
    def write_to_txt(self,excepts=[]):
        for sample in self.samples:
            path,polys,texts = sample['path'],sample['polys'],sample['texts']
            if path in excepts:
                continue
            f = open("ctw_top100_retrieval/train_gts/{}".format(os.path.basename(path).replace(".jpg",".txt")),'w')
            for poly,text in zip(polys,texts):
                pts = poly.reshape([-1]).tolist()
                str_line = "{},{},{},{},{},{},{},{},{}\n".format(pts[0],pts[1],pts[2],pts[3],pts[4],pts[5],pts[6],pts[7],text)
                f.write(str_line)
            f.close()
    def write_word_to_files(self,f):
        for target in self.samples:
            for word in target["texts"].copy():
                str_line = "{}\n".format(word)
                f.write(str_line)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        return self.samples[index]["path"], self.samples[index]["polys"], self.samples[index]["texts"]
def merge_chars(char1,char2):
    chars = char1.copy()
    for char in char2:
        if char not in char1:
            chars.append(char)
    return chars
from shutil import copyfile
def extract_retrieval_samples(dataset):
    imgname_words={}
    text_num = {}
    for i in range(len(dataset)):
        img_path,polys,texts = dataset[i]
        if len(texts)==0:
            continue
        imgname_words[os.path.basename(img_path)] = []
        for word in texts:
            if word not in text_num.keys():
                text_num[word] = []
            text_num[word].append(img_path)
            imgname_words[os.path.basename(img_path)].append(word)
    text_num = sorted(text_num.items(), key=lambda d: len(d[1]),reverse=True)
    all_images = []
    f = open("ctw_top100_retrieval/queries.txt",'w')
    for k,v in text_num[:100]:
        print("{}:{}".format(k,len(v)))
        f.write(k+'\n')
        for img in v:
            if img not in all_images:
                all_images.append(img)
    print(len(all_images))
    dataset.write_to_txt(all_images)
    for path in all_images:
        imgname = os.path.basename(path)
        copyfile(path, os.path.join("ctw_top100_retrieval/images",imgname))
        f = open("ctw_top100_retrieval/gts/{}".format(imgname.replace(".jpg",".txt")),'w')
        str_line = ""
        for word in imgname_words[imgname]:
            # print(imgname,len(imgname_words[imgname]))
            f.write(word+'\n')
f = open("words.txt",'w')
dataset = CTW("/workspace/wanghao/datasets/CTW")
# extract_retrieval_samples(dataset)
# chars = dataset.get_char_list()
# print(chars)
dataset.write_word_to_files(f)

dataset = RCTW("/workspace/wanghao/datasets/RCTW")
dataset.write_word_to_files(f)
# all_chars = merge_chars(chars,dataset.chars)
# print(len(all_chars))
# np.save("ctw_top100_retrieval/rctw_ctw_chars.npy",all_chars)
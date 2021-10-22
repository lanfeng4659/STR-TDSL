import editdistance
import random
import numpy as np
import torch
# from geo_map_cython_lib import gen_geo_map
# from editdistance3 import editdistance
from multiprocessing.dummy import Pool as ThreadPool
class TextGenerator(object):
    def __init__(self,ratios=[1,1,1,5],chars='abcdefghijklmnopqrstuvwxyz0123456789'):
        self.func = []
        for ratio, func in zip(ratios,[self._insert,self._delete,self._change,self._keep]):
            self.func.extend([func]*ratio)
        self.chars = chars
        self.char_to_label_map = {}
        self.is_chinese = len(self.chars)>1000
        for i, c in enumerate(self.chars):
            self.char_to_label_map[c] = i
    def __call__(self,word):
        idx = 0
        new_word = word
        while idx < len(new_word):
            new_word = random.choice(self.func)(new_word,idx)
            idx +=1
        return new_word
    def label_map(self, word):
        if self.is_chinese:
            result = [self.char_to_label_map[char] for char in word if char in self.chars]
            # print(word,result)
            return result
        else:
            return [self.char_to_label_map[char] for char in word if char.lower() in self.chars]
    def label_map_with_padding(self, word, max_len=15, padding=0):
        labels = [padding]*max_len
        if self.is_chinese:
            for i,char in enumerate(word):
                if char in self.chars:
                    labels[i] = self.char_to_label_map[char]
                # result = [self.char_to_label_map[char] for char in word if char in self.chars]
            # print(word,result)
            return labels
        else:
            for i,char in enumerate(word):
                char = char.lower()
                if char in self.chars:
                    labels[i] = self.char_to_label_map[char]
            return labels
    def _insert(self,word, idx):
        assert idx < len(word)
        return word[:idx]+random.choice(self.chars)+word[idx:]
    def _delete(self,word, idx):
        assert idx < len(word)
        return word[:idx]+word[idx+1:]
    def _change(self,word, idx):
        assert idx < len(word)
        return word[:idx]+random.choice(self.chars)+word[idx+1:]
    def _keep(self,word, idx):
        return word
    def similarity_on_pair(self,a,b,dis=0):
        a_len = len(a)
        b_len = len(b)
        op = [[0]*(b_len+1) for i in range(a_len+1)]
        si = [[0]*(b_len+1) for i in range(a_len+1)]
        for i in range(b_len+1):
            op[0][i] = i
            si[0][i] = 0
        for i in range(a_len+1):
            op[i][0] = i
            si[i][0] = 0
        for i in range(1, a_len+1):
            for j in range(1, b_len+1):
                if a[i-1]==b[j-1]:
                    op[i][j] = op[i-1][j-1]
                else:
                    op[i][j] = min([op[i-1][j], op[i][j-1], op[i-1][j-1]])+1
                si[i][j] = 1-op[i][j]/max([i,j])
        # print(si)
        return si
    def editdistance(self, word1, word2):
        return 1-editdistance.eval(word1,word2)/max(len(word1), len(word2))
    def calculate_similarity_matric(self, words1, words2):
        similarity = np.zeros([len(words1), len(words2)])
        for i,word1 in enumerate(words1):
            for j,word2 in enumerate(words2):
                similarity[i,j] = self.editdistance(word1, word2)
        return similarity
    def phoc_level_1(self, words):
        phoc1 = np.array([[1 if c in word else 0 for c in self.chars] for word in words]).reshape([len(words),len(self.chars)])
        return phoc1
    def filter_words(self, texts):
        idxs,new_texts = [], []
        for idx, text in enumerate(texts):
            text = text.lower()
            char_list = [c for c in text if c in self.chars]
            if len(char_list)<3:
                continue
            idxs.append(idx)
            new_texts.append("".join(char_list))
        return idxs, new_texts
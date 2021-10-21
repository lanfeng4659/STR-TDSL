import editdistance
import zipfile
import re

def char2num(char):
    if char in '0123456789':
        num = ord(char) - ord('0') + 1
    elif char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
        num = ord(char.lower()) - ord('a') + 11
    else:
        print('error symbol', char)
        exit()
    return num - 1

def ed_delect_cost(j, i, word1, word2, scores):
    ## delect a[i]
    c = char2num(word1[j])
    return scores[c][j]


def ed_insert_cost(i, j, word1, word2, scores):
    ## insert b[j]
    if i < len(word1) - 1:
        c1 = char2num(word1[i])
        c2 = char2num(word1[i + 1])
        return (scores[c1][i] + scores[c2][i+1])/2
    else:
        c1 = char2num(word1[i])
        return scores[c1][i]


def ed_replace_cost(i, j, word1, word2, scores):
    ## replace a[i] with b[j]
    c1 = char2num(word1[i])
    c2 = char2num(word2[j])
    # if word1 == "eeatpisaababarait".upper():
    #     print(scores[c2][i]/scores[c1][i])
    return max(1 - scores[c2][i]/scores[c1][i]*5, 0)

def weighted_edit_distance(word1, word2, scores):
    m = len(word1)
    n = len(word2)
    dp = [[0 for __ in range(m + 1)] for __ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):  ## word2
        for j in range(1, m + 1): ## word1
            delect_cost = ed_delect_cost(j-1, i-1, word1, word2, scores)  ## delect a[i]
            insert_cost = ed_insert_cost(j-1, i-1, word1, word2, scores)  ## insert b[j]
            if word1[j - 1] != word2[i - 1]:
                replace_cost = ed_replace_cost(j-1, i-1, word1, word2, scores) ## replace a[i] with b[j]
            else:
                replace_cost = 0
            dp[i][j] = min(dp[i-1][j] + insert_cost, dp[i][j-1] + delect_cost, dp[i-1][j-1] + replace_cost)

    return dp[n][m]

def find_match_word(rec_str, lexicon, scores_numpy=None, use_ed = True, weighted_ed = False):
    if not use_ed:
        return rec_str
    rec_str = rec_str.upper()
    dist_min = 100
    dist_min_pre = 100
    match_word = ''
    match_dist = 100
    if not weighted_ed:
        if rec_str in lexicon:
            return rec_str,0

        for word in lexicon:
            word = word.upper()
            ed = editdistance.eval(rec_str, word)
            length_dist = abs(len(word) - len(rec_str))
            # dist = ed + length_dist
            dist = ed
            if dist<dist_min:
                dist_min = dist
                match_word = word
                # match_word = pairs[word]
                match_dist = dist
        # print(rec_str,'-->>',match_word)
        return match_word, match_dist
    else:
        small_lexicon_dict = dict()
        for word in lexicon:
            word = word.upper()
            ed = editdistance.eval(rec_str, word)
            small_lexicon_dict[word] = ed
            dist = ed
            if dist<dist_min_pre:
                dist_min_pre = dist
        small_lexicon = []
        for word in small_lexicon_dict:
            if small_lexicon_dict[word]<=dist_min_pre+2:
                small_lexicon.append(word)

        for word in small_lexicon:
            word = word.upper()
            try:
                ed = weighted_edit_distance(rec_str, word_remove, scores_numpy)
            except:
                ed = editdistance.eval(rec_str, word)
            dist = ed
            if dist<dist_min:
                dist_min = dist
                match_word = word
                # match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist

class make_match_to_dict(object):
    def __init__(self,lexicon_type,weighted_ed):
        self.lexicon_type=lexicon_type
        self.weighted_ed=weighted_ed

        if lexicon_type==1:
            lexicon_path='maskrcnn_benchmark/data/datasets/evaluation/totaltext/full_dict.txt'
            lexicon_fid=open(lexicon_path, 'r')
            lexicon=[]
            for line in lexicon_fid.readlines():
                line=line.strip()
                lexicon.append(line)
        elif lexicon_type==3:
            pass
                        
        self.lexicon=lexicon

    def __call__(self,word,scores=None,use_lexicon=True,ids=0):
        if self.lexicon_type==1:
            lexicon=self.lexicon
        elif self.lexicon_type==3:
            lexicon=self.lexicon[ids]
        match_word, match_dist = find_match_word(word, lexicon, scores, use_lexicon, self.weighted_ed)
        if match_dist<3:
            return match_word
        else:
            return ''
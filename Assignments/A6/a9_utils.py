import re
import math
import copy
import random
import time
import json

class InvertedIndex:
    def __init__(self):
        self.index = {}

    def insert(self, word, docid):
        index = self.index
        if word in index:
            index[word].add(docid)
        else:
            index[word] = set([docid])

    def get(self, *words):
        docids = set()
        for word in words:
            docids =  docids.union(self.index.get(word, set()))
        return docids


def jaccard(s, t):
    intersect = len(set(s) & set(t))
    union = len(s) + len(t) - intersect
    if union == 0:
        return 0
    else:
        return intersect *1.0/union

def jaccard_w(s, t, lower_case = True, alphanum_only = True):
    return jaccard(wordset(s, lower_case, alphanum_only), \
                            wordset(t, lower_case, alphanum_only))


def jaccard_g(s, t, gram_size, lower_case = True, alphanum_only = True):
    return jaccard(gramset(s, gram_size, lower_case, alphanum_only), \
                            gramset(t, gram_size, lower_case, alphanum_only))

def editsim(s, t):
    n = len(s)
    m = len(t)

    dist = [[0 for j in range(m+1)] for i in range(n+1)]

    for i in range(1,n+1):
        dist[i][0] = dist[i-1][0] + 1

    for j in range(1,m+1):
        dist[0][j] = dist[0][j-1] + 1

    for i in range(1,n+1):
        for j in range(1,m+1):
           dist[i][j] = min(dist[i-1][j]+1, \
                                          dist[i][j-1]+1,
                                dist[i-1][j-1] if s[i-1] == t[j-1] else dist[i-1][j-1]+1)

    return 1- dist[n][m]*1.0/max(m, n)


def alphnum(s):
    sep = re.compile(r"[\W]")
    items = sep.split(s)
    return " ".join([item for item in items if item.strip()!=""])


def wordset(s, lower_case = True, alphanum_only = True):
    if lower_case:
       s = s.lower()
    if alphanum_only:
        s = alphnum(s)
    return s.split()


def gramset(s, gram_size, lower_case = True, alphanum_only = True):
    if gram_size <= 0 or type(gram_size) is not int:
        raise Exception("'gram_size="+str(gram_size) + "' is not a non-negative integer.")
    if lower_case:
       s = s.lower()
    if alphanum_only:
        s = alphnum(s)
    ns = len(s)+gram_size-1
    s = "^"*(gram_size-1)+s+"$"*(gram_size-1)
    return [s[i:i+gram_size] for i in range(ns)]


class SimJoin:

    def __init__(self, k_o_list):
        self.k_o_list = k_o_list


    def _idf(self, docs):
        word_to_idf = {}
        word_to_count = {}
        for doc in docs:
            flags = {}
            for w in doc:
                if w in flags:
                    continue
                flags[w] = True
                word_to_count[w] = word_to_count.get(w, 0) + 1
        for w, c in word_to_count.items():
            word_to_idf[w] = math.log(len(docs)*1.0/c)
        return word_to_idf


    def _get_idf(self, word):
        return self.word_to_idf.get(word, self.max_idf)


    def _sum_weight(self, words):
        sum_weight = 0
        for w in words:
            sum_weight += self._get_idf(w)
        return sum_weight


    def _prefix(self, key, threshold, weight_on=False):
        if len(key) == 0:
            return []

        last_pos = len(key)
        if weight_on:
            overlap_weight = self._sum_weight(key)*threshold
            for i, w in enumerate(reversed(key)):
                if overlap_weight - self._get_idf(w) <= 0:
                    last_pos = len(key)-i
                    break
                else:
                    overlap_weight -= self._get_idf(w)
        else:
            last_pos = len(key)-int(math.ceil(len(key)*threshold))+1

        return key[:last_pos]


    # if Jaccard(s, t) >= threshold, the function will return the real similarity; Otherwise, it will return 0.
    def _jaccard(self, s, t, threshold, weight_on):
        if weight_on:
            sum1 = self._sum_weight(s)
            sum2 = self._sum_weight(t)
            if sum1 < sum2:
                if sum1 < sum2*threshold:
                    return 0
            else:
                if sum2 < sum1*threshold:
                    return 0
            intersect = self._sum_weight(set(s) & set(t))
            union = sum1+sum2-intersect
        else:
            if len(s) < len(t):
                if len(s) < len(t)*threshold:
                    return 0
            else:
                if len(t) < len(s)*threshold:
                    return 0
            intersect = len(set(s) & set(t))
            union = len(t) + len(s) - intersect

        if union != 0 and intersect*1.0/union+1e-6 >= threshold:
            return intersect*1.0/union
        else:
            return 0


    def selfjoin(self, threshold, weight_on = False):
        if threshold < 0 or threshold > 1:
            raise Exception("threshold is not in the range of [0, 1]")
        # Compute IDF for each word
        k_list = [k for k, o in self.k_o_list]
        self.word_to_idf = self._idf(k_list)
        self.max_idf = math.log(len(k_list)*2.0) # For the words that are not in docs, their idf will be set to self.max_idf

        k_o_list = self.k_o_list

        # Sort the elements in each joinkey in decreasing order of IDF
        sk_list = []
        for k, o in k_o_list:
            sk = sorted(k, key=lambda x: (self._get_idf(x), x), reverse=True)
            sk_list.append(sk)

        # (1) Generate candidate pairs whose prefixes share elements;
        # (2) Compute the similarity of the candidate pairs and return the ones whose similarity is above  the threshold
        idx = InvertedIndex()

        joined = []
        for i, sk in enumerate(sk_list):
            prefix = self._prefix(sk, threshold, weight_on)
            ids = idx.get(*prefix)
            for j in ids:
                sim = self._jaccard(sk, sk_list[j], threshold, weight_on)
                if sim != 0:
                    joined.append((k_o_list[i], k_o_list[j], sim))

            for w in prefix:
                idx.insert(w, i)

        return joined


def simjoin(data):
    def joinkey_func(row):
        # concatenate 'name', 'address', 'city' and 'type', and
        # tokensize the concatenated string into a set of words
        return wordset(' '.join(row[1:]))

    key_row_list = [(joinkey_func(row) , row) for row in data]
    sj = SimJoin(key_row_list)
    result = sj.selfjoin(0.4)
    result.sort(key=lambda x: -x[2])
    simpairs = [(row1, row2) for (key1, row1), (key2, row2), sim in result]
    return simpairs

def featurize(pair):
    row1, row2 = pair

    #cleaning
    row1 = [alphnum(str(x).lower()) for x in row1]
    row2 = [alphnum(str(x).lower()) for x in row2]

    # features
    f1 = editsim(row1[1], row2[1])
    f2 = jaccard_w(row1[1], row2[1])
    f3 = editsim(row1[2], row2[2])
    f4 = jaccard_w(row1[2], row2[2])
    f5 = editsim(row1[3], row2[3])
    f6 = editsim(row1[4], row2[4])

    return (f1, f2, f3, f4, f5, f6)


def crowdsourcing(pair):
    is_match = False
    with open('true_matches.json') as f:
        true_matches = json.load(f)
        id1, id2 = str(pair[0][0]), str(pair[1][0])
        print('\x1b[1;31m'+'Are they matching?'+'\x1b[0m')
        print pair[0]
        print pair[1]
        if [id1, id2] in true_matches or [id2, id1] in true_matches:
            is_match = True
        time.sleep(1)
    print '\x1b[1;31m'+'Answer: %s' %("Yes" if is_match else "No") +'\x1b[0m'
    print
    return is_match

def crowdsourcing_fast(pair):
    is_match = False
    with open('true_matches.json') as f:
        true_matches = json.load(f)
        id1, id2 = str(pair[0][0]), str(pair[1][0])
        if [id1, id2] in true_matches or [id2, id1] in true_matches:
            is_match = True
    return is_match


def total_true_matches():
    total = 0
    with open('true_matches.json') as f:
        true_matches = json.load(f)
        total = len(true_matches)
    return total


def evaluate(identified_matches):
    n = 0
    for pair in identified_matches:
        if crowdsourcing_fast(pair):
            n += 1
    precision = n*1.0/len(identified_matches)
    recall = n*1.0/total_true_matches()
    fscore = 2*precision*recall/(precision+recall)
    return (precision, recall, fscore)









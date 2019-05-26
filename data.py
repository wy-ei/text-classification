import os
from collections import Counter
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'

# 文档最大长度限制
DOCUMENT_MAX_LENGTH = 500

CATEGIRY_LIST = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
CATEGIRY_MAP = { c: i for i,c in enumerate(CATEGIRY_LIST) }


def build_dict(files, num_words=5000):
    counter = Counter()
    
    for file in files:
        fin = open(file, encoding='utf-8', mode='r')
        for line in fin:
            counter.update(line)
        fin.close()
    
    words = [w for w, c in counter.most_common(num_words - 2)]
    words =  [PAD_WORD, UNK_WORD] + words
    
    dct = {word: i for i, word in enumerate(words)}

    return dct


class NewsDataSet(Dataset):
    def __init__(self, file, dictionary):
        self.dct = dictionary
        self.docs, self.labels = self.process_file(file)
        
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, i):
        return self.docs[i], self.labels[i]
    
    def process_line(self, line):
        label, document = line.strip().split('\t')
        UNK = self.dct[UNK_WORD]
        PAD = self.dct[PAD_WORD]

        if len(document) > DOCUMENT_MAX_LENGTH:
            document = document[:DOCUMENT_MAX_LENGTH]

        idx = [self.dct.get(w, UNK) for w in document]

        if len(idx) < DOCUMENT_MAX_LENGTH:
            idx += [PAD] * (DOCUMENT_MAX_LENGTH - len(idx))
        
        idx = torch.tensor(idx, dtype=torch.long)
        label = CATEGIRY_MAP[label]

        return idx, label

    def process_file(self, file):
        docs = []
        labels = []

        with open(file, encoding='utf-8', mode='r') as fin:
            for line in fin:
                document, label = self.process_line(line)
                docs.append(document)
                labels.append(label)

        return docs, labels
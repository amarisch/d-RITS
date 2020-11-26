import os
import time

import ujson as json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self,filename, indices):
        super(MySet, self).__init__()
        self.content = open('./{}'.format(filename)).readlines()

        if len(indices) == 0:
            self.indices = np.arange(len(self.content))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rec = json.loads(self.content[self.indices[idx]])
        return rec

def collate_fn(recs):
    forward = map(lambda x: x['forward'], recs)

    def to_tensor_dict(recs):
        values = []
        masks = []
        deltas = []
        evals = []
        eval_masks = []
        forwards = []
        for r in recs:
            values.append(r['values'])
            masks.append(r['masks'])
            deltas.append(r['deltas'])
            evals.append(r['evals'])
            eval_masks.append(r['eval_masks'])
            forwards.append(r['forwards'])
            
        values = torch.FloatTensor(values)
        masks = torch.FloatTensor(masks)
        deltas = torch.FloatTensor(deltas)
        evals = torch.FloatTensor(evals)
        eval_masks = torch.FloatTensor(eval_masks)
        forwards = torch.FloatTensor(forwards)

        return {'values': values, 'forwards': forwards, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward)}
    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))

    return ret_dict

def get_loader(filename, indices, batch_size = 64, shuffle = False, get_labels=False):
    data_set = MySet(filename, indices)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )
    if get_labels:
        rec = data_set[0]
        num_classes = len(rec['label'])
        labels = np.zeros(num_classes)
        for rec in data_set:
            labels += rec['label']
        return data_iter, labels
    else:
        return data_iter

import torch
import dgl
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score

def split_data(hg, etype_name, train_ratio, eval_ratio, test_ratio):
    src, dst = hg.edges(etype=etype_name)
    etype_src = src.numpy().tolist()
    etype_dst = dst.numpy().tolist()
    
    num_link = len(etype_src)

    pos_label=[1]*num_link
    pos_data=list(zip(etype_src, etype_dst, pos_label))

    etype_adj = np.array(hg.adj(etype=etype_name).to_dense())
    full_idx = np.where(etype_adj==0)

    sample = random.sample(range(0, len(full_idx[0])), num_link)
    neg_label = [0]*num_link
    neg_data = list(zip(full_idx[0][sample],full_idx[1][sample], neg_label))
    
    full_data = pos_data + neg_data
    random.shuffle(full_data)

    train_size = int(len(full_data) * train_ratio)
    eval_size = int(len(full_data) * eval_ratio)
    test_size = len(full_data) - train_size - eval_size
    
    train_data = full_data[:train_size]
    eval_data = full_data[train_size : train_size+eval_size]
    test_data = full_data[train_size+eval_size : train_size+eval_size+test_size]
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    test_data = np.array(test_data)
    
    return train_data, eval_data, test_data

def evaluate_auc(pred, label):
    res=roc_auc_score(y_score=pred, y_true=label)
    return res

def evaluate_acc(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return accuracy_score(y_pred=res, y_true=label)
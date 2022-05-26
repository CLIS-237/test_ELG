import torch
import dgl
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score
'''
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
'''
def split_data(hg, etype_name, train_ratio, eval_ratio, test_ratio, usage):
    src, dst = hg.edges(etype=etype_name)
    etype_src = src.numpy().tolist()
    etype_dst = dst.numpy().tolist()
    
    num_link = len(etype_src)

    pos_label=[1]*num_link
    pos_data=list(zip(etype_src, etype_dst, pos_label))
    
    random.shuffle(pos_data)

    pos_train_size = int(len(pos_data) * usage * train_ratio)
    pos_eval_size = int(len(pos_data) * usage * eval_ratio)
    pos_test_size = int(len(pos_data) * usage) - pos_train_size - pos_eval_size

    pos_train_data = pos_data[:pos_train_size]
    pos_eval_data = pos_data[pos_train_size : pos_train_size + pos_eval_size]
    pos_test_data = pos_data[pos_train_size + pos_eval_size : pos_train_size + pos_eval_size + pos_test_size]

    pos_train_row_idx = get_row_idx(pos_train_data)
    pos_eval_row_idx  = get_row_idx(pos_eval_data)
    pos_test_row_idx  = get_row_idx(pos_test_data)
    
    etype_adj = np.array(hg.adj(etype=etype_name).to_dense())
    train_xy_iszero = get_zero_xy(pos_train_row_idx, etype_adj)
    eval_xy_iszero = get_zero_xy(pos_eval_row_idx, etype_adj)
    test_xy_iszero = get_zero_xy(pos_test_row_idx, etype_adj)

    sample_train = random.sample(range(0, len(train_xy_iszero[0])), pos_train_size)
    sample_eval  = random.sample(range(0, len(eval_xy_iszero[0])), pos_eval_size)
    sample_test  = random.sample(range(0, len(test_xy_iszero[0])), pos_test_size)
    neg_train_data = list(zip(train_xy_iszero[0][sample_train],train_xy_iszero[1][sample_train], [0]*pos_train_size))
    neg_eval_data  = list(zip(eval_xy_iszero[0][sample_eval],eval_xy_iszero[1][sample_eval], [0]*pos_eval_size))
    neg_test_data  = list(zip(test_xy_iszero[0][sample_test],test_xy_iszero[1][sample_test], [0]*pos_test_size))
    
    train_data = pos_train_data + neg_train_data
    eval_data = pos_eval_data + neg_eval_data
    test_data = pos_test_data + neg_test_data
    
    random.shuffle(train_data)
    random.shuffle(eval_data)
    random.shuffle(test_data)
    used_data = train_data + eval_data + test_data

    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    test_data = np.array(test_data)
    used_data = np.array(used_data)
    
    return train_data, eval_data, test_data, used_data

def get_row_idx(data):
    row_idx_list = []
    for i in range(0, len(data)):
        tmp_tuple = data[i]
        row_idx_list.append(tmp_tuple[0])
    return row_idx_list

def get_zero_xy(row_idx, adj):
    x = []
    y = []
    for i in row_idx:
        x.append(i)
        j = random.randint(0,adj.shape[1]-1)
        while adj[i][j] != 0:
            j = random.randint(0,adj.shape[1]-1)
        y.append(j)
    ret = (np.array(x), np.array(y))
    return ret

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

def search_first_catch(tmp_val_acc, eval_acc_list):
    ret = -1
    for i in range(0, len(eval_acc_list)):
        if eval_acc_list[i] >= tmp_val_acc:
            ret = i
            break
    return ret

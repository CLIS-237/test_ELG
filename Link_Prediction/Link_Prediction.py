import scipy.io
import urllib.request
import dgl
import math
import random
import time
import joblib
import torch as th
import numpy as np
import dgl.function as fn
import matplotlib.pyplot as plt
from HGTDGL.model import *
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from utils import split_data, evaluate_auc, evaluate_acc

# load data
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = './HGTDGL/tmp/ACM.mat'
urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
device = th.device("cuda:1")

# Build the original graph G
original_G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        ('paper','publish-on','conference') : data['PvsC'].nonzero(),
        ('conference','publish','paper') : data['PvsC'].transpose().nonzero(),
    })

# Split
train_ratio = 0.8
eval_ratio = 0.1
test_ratio = 1 - train_ratio - eval_ratio
train_data, eval_data, test_data = split_data(original_G, 'publish-on', train_ratio, eval_ratio, test_ratio)

# Delete the positive edges in eval/test data in the original graph G
train_pos = np.nonzero(train_data[:,2])
train_pos_idx = train_pos[0]
paper_conf_src_processed = train_data[train_pos_idx, 0]
paper_conf_dst_processed = train_data[train_pos_idx, 1]
G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        ('paper','publish-on','conference') : (paper_conf_src_processed, paper_conf_dst_processed),
        ('conference','publish','paper') : data['PvsC'].transpose().nonzero(),
    })

G = G.to(device)
train_data = torch.Tensor(train_data).long()
eval_data = torch.Tensor(eval_data).long()
test_data = torch.Tensor(test_data).long()

# Initialization
G.node_dict = {}
G.edge_dict = {}
for ntype in G.ntypes:
    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = th.ones(G.number_of_edges(etype), dtype=th.long).to(device) * G.edge_dict[etype] 
for ntype in G.ntypes:
    emb = nn.Parameter(th.Tensor(G.number_of_nodes(ntype), 400), requires_grad = False).to(device)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb

# Using HGT model to obtain node embeddings 
model = HGT(G, n_inp=400, n_hid=200, n_out=30, n_layers=2, n_heads=4, use_norm = True, dropout = 0.1).to(device)
optimizer = th.optim.AdamW(model.parameters())
scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr = 1e-3, pct_start=0.05)

# Train
train_step = 0
for epoch in range(200):
    model.train()
    paper_idx, conf_idx, label = train_data[:,0], train_data[:,1], train_data[:,2]
    label = label.to(torch.float32).to(device)
    
    logits = model.forward(G, 'paper', 'conference', paper_idx, conf_idx)
    logits = logits.to(device)
    # Loss
    train_loss = F.binary_cross_entropy_with_logits(logits, label)
    
    # backward
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        paper_idx, conf_idx, label = eval_data[:,0], eval_data[:,1], eval_data[:,2]
        label = label.to(torch.float32).to(device)

        logits = model.forward(G, 'paper', 'conference', paper_idx, conf_idx)
        logits = logits.to(device)

        eval_loss = F.binary_cross_entropy_with_logits(logits, label)
        eval_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
    
    model.eval()
    with torch.no_grad():
        paper_idx, conf_idx, label = test_data[:,0], test_data[:,1], test_data[:,2]
        label = label.to(torch.float32).to(device)

        logits = model.forward(G, 'paper', 'conference', paper_idx, conf_idx)
        logits = logits.to(device)

        test_loss = F.binary_cross_entropy_with_logits(logits, label)
        test_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())

    if epoch % 40 == 0:
        print("Train Loss: {:.5}, Eval Loss: {:.5}, Eval Acc: {:.5}, Test Loss: {:.5}, Test Acc: {:.5}".format(train_loss, eval_loss, eval_acc, test_loss, test_acc))
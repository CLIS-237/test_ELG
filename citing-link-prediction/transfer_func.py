import scipy.io
import urllib.request
import dgl
import math
import random
import time
import copy
import joblib
import torch as th
import numpy as np
import dgl.function as fn
import matplotlib.pyplot as plt
from HGTDGL.model import *
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from utils import split_data, evaluate_auc, evaluate_acc

def no_transfer(target):
    # load data
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = './HGTDGL/tmp/ACM.mat'
    urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)
    device = th.device("cuda:2")

    # Build the original graph G
    original_G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            # ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
            ('paper','publish-on','conference') : data['PvsC'].nonzero(),
            ('conference','publish','paper') : data['PvsC'].transpose().nonzero(),
        })
    original_G = original_G.to(device)

    # Split subgraph based on subjects
    subjectCodes = []
    for a in data['L']:
        subjectCodes.append(a[0][0])

    subject = subjectCodes.index(target)
    subjects = [subject]
    
    subject_papers = torch.tensor([]).to(device)
    subject_papers = subject_papers.type(torch.int64)
    for s in subjects:
        subject_papers = torch.concat([subject_papers, (original_G.successors(s, etype='has'))])
    subject_papers = torch.unique(subject_papers)

    conferences = torch.tensor([]).to(device)
    conferences = conferences.type(torch.int64)
    for p in subject_papers:
        conferences = torch.concat([conferences, (original_G.successors(p, etype='publish-on'))])
    conferences = torch.unique(conferences)

    authors = torch.tensor([]).to(device)
    authors = authors.type(torch.int64)
    for p in subject_papers:
        authors = torch.concat([authors, (original_G.successors(p, etype='written-by'))])
    authors = torch.unique(authors)

    citing_papers = torch.tensor([]).to(device)
    citing_papers = citing_papers.type(torch.int64)
    for p in subject_papers:
        citing_papers = torch.concat([citing_papers, (original_G.successors(p, etype='citing'))])
    citing_papers = torch.unique(citing_papers)
    papers = torch.unique(torch.concat([citing_papers, subject_papers]))
    
    # G0    the graph with the 'citing' edges both from subject papers and citing_papers, 
    #       the 'citing' edges from citing papers need to be deleted (if a citing paper 
    #        belongs to subject papers, corresponding edges should be kept) 
    G0 = dgl.node_subgraph(original_G, {'subject':subjects,'author':authors, 'paper':papers, 'conference':conferences})
    
    # Delete the redundant edges in G0
    subjects_in_G0 = [0]     # only one subject
    subject_papers_in_G0 = torch.tensor([]).to(device)
    subject_papers_in_G0 = subject_papers_in_G0.type(torch.int64)
    for s in subjects_in_G0:
        subject_papers_in_G0 = torch.concat([subject_papers_in_G0,(G0.successors(s, etype='has'))])
    subject_papers_in_G0 = torch.unique(subject_papers_in_G0)

    etype = ('paper','citing','paper')
    src_citing_all, dst_citing_all, eid_citing_all = G0.edges(etype=etype,form='all')

    subject_papers_in_G0_eid_list = []          # get all edge ids with subject_papers as the source nodes
    for idx in range(0, len(src_citing_all)):
        if src_citing_all[idx] in subject_papers_in_G0:
            subject_papers_in_G0_eid_list.append(eid_citing_all[idx].item())

    eid_citing_delete = list(set(eid_citing_all.tolist())^set(subject_papers_in_G0_eid_list))
    eid_citing_delete = th.tensor(eid_citing_delete, dtype=th.long).to(device)
    G0 = dgl.remove_edges(G0, eid_citing_delete, etype).to(device)
    G0 = G0.to('cpu')
    
    # Split train/eval/test
    train_ratio = 0.6
    eval_ratio = 0.2
    test_ratio = 1 - train_ratio - eval_ratio
    usage = 0.8
    train_data, eval_data, test_data, pos_unused_data = split_data(G0, 'citing', train_ratio, eval_ratio, test_ratio, usage)

    train_data_output = train_data
    eval_data_output = eval_data
    test_data_output = test_data
    pos_unused_data_output = pos_unused_data
    '''
    # Delete the positive edges in eval/test data in the graph G0
    train_pos = np.nonzero(train_data[:,2])
    train_pos_idx = train_pos[0]
    paper_paper_src_processed = train_data[train_pos_idx, 0]
    paper_paper_dst_processed = train_data[train_pos_idx, 1]
    '''
    # Delete the positive edges in train/eval/test data in the graph G0
    unused_pos = np.nonzero(pos_unused_data[:,2])
    unused_pos_idx = unused_pos[0]
    paper_paper_src_processed = pos_unused_data[unused_pos_idx, 0]
    paper_paper_dst_processed = pos_unused_data[unused_pos_idx, 1]

    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : (paper_paper_src_processed, paper_paper_dst_processed),
            # ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
            ('paper','publish-on','conference') : data['PvsC'].nonzero(),
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
    model = HGT(G, n_inp=400, n_hid=200, n_out=30, n_layers=2, n_heads=4, use_norm = True, MLP_dropout = 0).to(device)
    optimizer = th.optim.AdamW(model.parameters())
    scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr = 1e-3, pct_start=0.05)

    train_step = 0
    best_eval_acc = 0
    best_test_acc = 0
    start = time.time()
    best_time = time.time()
    best_model = copy.deepcopy(model)

    train_acc_list = []
    eval_acc_list = []
    test_acc_list = []
    time_list = []

    for epoch in range(200):
        model.train()
        paper_idx0, paper_idx1, label = train_data[:,0], train_data[:,1], train_data[:,2]
        label = label.to(torch.float32).to(device)
    
        logits = model.forward(G, 'paper', 'paper', paper_idx0, paper_idx1)
        logits = logits.to(device)
        # Loss
        train_loss = F.binary_cross_entropy_with_logits(logits, label)
        train_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            paper_idx0, paper_idx1, label = eval_data[:,0], eval_data[:,1], eval_data[:,2]
            label = label.to(torch.float32).to(device)

            logits = model.forward(G, 'paper', 'paper', paper_idx0, paper_idx1)
            logits = logits.to(device)

            eval_loss = F.binary_cross_entropy_with_logits(logits, label)
            eval_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
    
        model.eval()
        with torch.no_grad():
            paper_idx0, paper_idx1, label = test_data[:,0], test_data[:,1], test_data[:,2]
            label = label.to(torch.float32).to(device)

            logits = model.forward(G, 'paper', 'paper', paper_idx0, paper_idx1)
            logits = logits.to(device)

            test_loss = F.binary_cross_entropy_with_logits(logits, label)
            test_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())

        train_acc_list.append(train_acc.item())
        eval_acc_list.append(eval_acc.item())
        test_acc_list.append(test_acc.item())
        time_list.append(time.time()-start)

        if best_eval_acc < eval_acc:
            best_eval_acc = eval_acc
            best_test_acc = test_acc
            best_time = time.time()
            best_model = copy.deepcopy(model)
        
        if epoch % 40 == 0:
            print('LR: %.5f, Train-Loss: %.4f, Eval-Loss: %.4f, Test-Loss: %.4f, Train-ACC: %.4f, Eval-ACC: %.4f, Test-ACC: %.4f, Best-Eval-ACC: %.4f, Best-Test-ACC: %.4f' % (
                optimizer.param_groups[0]['lr'],
                train_loss.item(),
                eval_loss.item(),
                test_loss.item(),
                train_acc.item(),
                eval_acc.item(),
                test_acc.item(),
                best_eval_acc.item(),
                best_test_acc.item(),
            ))

    end = time.time()
    best_model_training_time = best_time - start
    total_training_time = end - start


    return best_model, best_model_training_time, total_training_time, best_eval_acc, best_test_acc, time_list, train_acc_list, eval_acc_list, test_acc_list, train_data_output, eval_data_output, test_data_output, pos_unused_data_output


def transfer(target, trained_model, train_data, eval_data, test_data, pos_unused_data):
    # load data
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = './HGTDGL/tmp/ACM.mat'
    urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)
    device = th.device("cuda:2")

    # Build the original graph G
    original_G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            # ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
            ('paper','publish-on','conference') : data['PvsC'].nonzero(),
            ('conference','publish','paper') : data['PvsC'].transpose().nonzero(),
        })
    original_G = original_G.to(device)

    # Split subgraph based on subjects
    subjectCodes = []
    for a in data['L']:
        subjectCodes.append(a[0][0])

    subject = subjectCodes.index(target)
    subjects = [subject]
    
    subject_papers = torch.tensor([]).to(device)
    subject_papers = subject_papers.type(torch.int64)
    for s in subjects:
        subject_papers = torch.concat([subject_papers, (original_G.successors(s, etype='has'))])
    subject_papers = torch.unique(subject_papers)

    conferences = torch.tensor([]).to(device)
    conferences = conferences.type(torch.int64)
    for p in subject_papers:
        conferences = torch.concat([conferences, (original_G.successors(p, etype='publish-on'))])
    conferences = torch.unique(conferences)

    authors = torch.tensor([]).to(device)
    authors = authors.type(torch.int64)
    for p in subject_papers:
        authors = torch.concat([authors, (original_G.successors(p, etype='written-by'))])
    authors = torch.unique(authors)

    citing_papers = torch.tensor([]).to(device)
    citing_papers = citing_papers.type(torch.int64)
    for p in subject_papers:
        citing_papers = torch.concat([citing_papers, (original_G.successors(p, etype='citing'))])
    citing_papers = torch.unique(citing_papers)
    papers = torch.unique(torch.concat([citing_papers, subject_papers]))
    
    # G0    the graph with the 'citing' edges both from subject papers and citing_papers, 
    #       the 'citing' edges from citing papers need to be deleted (if a citing paper 
    #       belongs to subject papers, corresponding edges should be kept) 
    G0 = dgl.node_subgraph(original_G, {'subject':subjects,'author':authors, 'paper':papers, 'conference':conferences})
    
    # Delete the redundant edges in G0
    subjects_in_G0 = [0]     # only one subject
    subject_papers_in_G0 = torch.tensor([]).to(device)
    subject_papers_in_G0 = subject_papers_in_G0.type(torch.int64)
    for s in subjects_in_G0:
        subject_papers_in_G0 = torch.concat([subject_papers_in_G0,(G0.successors(s, etype='has'))])
    subject_papers_in_G0 = torch.unique(subject_papers_in_G0)

    etype = ('paper','citing','paper')
    src_citing_all, dst_citing_all, eid_citing_all = G0.edges(etype=etype,form='all')

    subject_papers_in_G0_eid_list = []          # get all edge ids with subject_papers as the source nodes
    for idx in range(0, len(src_citing_all)):
        if src_citing_all[idx] in subject_papers_in_G0:
            subject_papers_in_G0_eid_list.append(eid_citing_all[idx].item())

    eid_citing_delete = list(set(eid_citing_all.tolist())^set(subject_papers_in_G0_eid_list))
    eid_citing_delete = th.tensor(eid_citing_delete, dtype=th.long).to(device)
    G0 = dgl.remove_edges(G0, eid_citing_delete, etype).to(device)
    G0 = G0.to('cpu')
    
    # Split train/eval/test
    '''
    train_ratio = 0.6
    eval_ratio = 0.2
    test_ratio = 1 - train_ratio - eval_ratio
    usage = 0.8
    train_data, eval_data, test_data, pos_unused_data = split_data(G0, 'citing', train_ratio, eval_ratio, test_ratio, usage)
    '''
    '''
    # Delete the positive edges in eval/test data in the graph G0
    train_pos = np.nonzero(train_data[:,2])
    train_pos_idx = train_pos[0]
    paper_paper_src_processed = train_data[train_pos_idx, 0]
    paper_paper_dst_processed = train_data[train_pos_idx, 1]
    '''
    # Delete the positive edges in train/eval/test data in the graph G0
    unused_pos = np.nonzero(pos_unused_data[:,2])
    unused_pos_idx = unused_pos[0]
    paper_paper_src_processed = pos_unused_data[unused_pos_idx, 0]
    paper_paper_dst_processed = pos_unused_data[unused_pos_idx, 1]

    G = dgl.heterograph({
            ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
            ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : (paper_paper_src_processed, paper_paper_dst_processed),
            # ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
            ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
            ('paper','publish-on','conference') : data['PvsC'].nonzero(),
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
    model = trained_model
    optimizer = th.optim.AdamW(model.parameters())
    scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr = 1e-3, pct_start=0.05)

    train_step = 0
    best_eval_acc = 0
    best_test_acc = 0
    start = time.time()
    best_time = time.time()
    best_model = copy.deepcopy(model)

    train_acc_list = []
    eval_acc_list = []
    test_acc_list = []
    time_list = []

    for epoch in range(200):
        model.train()
        paper_idx0, paper_idx1, label = train_data[:,0], train_data[:,1], train_data[:,2]
        label = label.to(torch.float32).to(device)
    
        logits = model.forward(G, 'paper', 'paper', paper_idx0, paper_idx1)
        logits = logits.to(device)
        # Loss
        train_loss = F.binary_cross_entropy_with_logits(logits, label)
        train_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            paper_idx0, paper_idx1, label = eval_data[:,0], eval_data[:,1], eval_data[:,2]
            label = label.to(torch.float32).to(device)

            logits = model.forward(G, 'paper', 'paper', paper_idx0, paper_idx1)
            logits = logits.to(device)

            eval_loss = F.binary_cross_entropy_with_logits(logits, label)
            eval_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
    
        model.eval()
        with torch.no_grad():
            paper_idx0, paper_idx1, label = test_data[:,0], test_data[:,1], test_data[:,2]
            label = label.to(torch.float32).to(device)

            logits = model.forward(G, 'paper', 'paper', paper_idx0, paper_idx1)
            logits = logits.to(device)

            test_loss = F.binary_cross_entropy_with_logits(logits, label)
            test_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())

        train_acc_list.append(train_acc.item())
        eval_acc_list.append(eval_acc.item())
        test_acc_list.append(test_acc.item())
        time_list.append(time.time()-start)

        if best_eval_acc < eval_acc:
            best_eval_acc = eval_acc
            best_test_acc = test_acc
            best_time = time.time()
            best_model = copy.deepcopy(model)
        
        if epoch % 40 == 0:
            print('LR: %.5f, Train-Loss: %.4f, Eval-Loss: %.4f, Test-Loss: %.4f, Train-ACC: %.4f, Eval-ACC: %.4f, Test-ACC: %.4f, Best-Eval-ACC: %.4f, Best-Test-ACC: %.4f' % (
                optimizer.param_groups[0]['lr'],
                train_loss.item(),
                eval_loss.item(),
                test_loss.item(),
                train_acc.item(),
                eval_acc.item(),
                test_acc.item(),
                best_eval_acc.item(),
                best_test_acc.item(),
            ))

    end = time.time()
    best_model_training_time = best_time - start
    total_training_time = end - start

    return best_model, best_model_training_time, total_training_time, best_eval_acc, best_test_acc, time_list, train_acc_list, eval_acc_list, test_acc_list    

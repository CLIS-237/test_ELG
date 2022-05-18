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
    data_file_path = './HGTDGL/tmp/MAG.mat'
    data = scipy.io.loadmat(data_file_path)
    device = th.device("cuda:1")
    
    # Build the original graph G
    original_G = dgl.heterograph({
            ('author', 'writing', 'paper') : data['AvsP'].nonzero(),
            ('paper', 'written-by', 'author') : data['AvsP'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            # ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'field') : data['PvsF'].nonzero(),
            ('field', 'has', 'paper') : data['PvsF'].transpose().nonzero(),
            ('author','affiliated-with','institution') : data['AvsI'].nonzero(),
            ('institution','hire','author') : data['AvsI'].transpose().nonzero(),
        })
    original_G = original_G.to(device)

    # Initialization
    original_G.node_dict = {}
    original_G.edge_dict = {}
    for ntype in original_G.ntypes:
        original_G.node_dict[ntype] = len(original_G.node_dict)
    for etype in original_G.etypes:
        original_G.edge_dict[etype] = len(original_G.edge_dict)
        original_G.edges[etype].data['id'] = th.ones(original_G.number_of_edges(etype), dtype=th.long).to(device) * original_G.edge_dict[etype] 
    for ntype in original_G.ntypes:
        emb = nn.Parameter(th.Tensor(original_G.number_of_nodes(ntype), 128), requires_grad = False).to(device)
        nn.init.xavier_uniform_(emb)
        original_G.nodes[ntype].data['inp'] = emb
    # Meaningful embedding for paper nodes
    original_G.nodes['paper'].data['inp'] = th.tensor(data['paper_node_feat'], dtype=torch.float32).to(device)
    
    # Split subgraph based on fields
    field = target
    fields = [field]
    
    field_papers = torch.tensor([]).to(device)
    field_papers = field_papers.type(torch.int64)
    for s in fields:
        field_papers = torch.concat([field_papers, (original_G.successors(s, etype='has'))])
    field_papers = torch.unique(field_papers)

    authors = torch.tensor([]).to(device)
    authors = authors.type(torch.int64)
    for p in field_papers:
        authors = torch.concat([authors, (original_G.successors(p, etype='written-by'))])
    authors = torch.unique(authors)

    citing_papers = torch.tensor([]).to(device)
    citing_papers = citing_papers.type(torch.int64)
    for p in field_papers:
        citing_papers = torch.concat([citing_papers, (original_G.successors(p, etype='citing'))])
    citing_papers = torch.unique(citing_papers)
    papers = torch.unique(torch.concat([citing_papers, field_papers]))
  
    institutions = torch.tensor([]).to(device)
    institutions = institutions.type(torch.int64)
    for a in authors:
        institutions = torch.concat([institutions, (original_G.successors(a, etype='affiliated-with'))])
    institutions = torch.unique(institutions)
        
    # G0    the graph with the 'citing' edges both from field papers and citing_papers, 
    #       the 'citing' edges from citing papers need to be deleted (if a citing paper 
    #        belongs to field papers, corresponding edges should be kept) 
    G0 = dgl.node_subgraph(original_G, {'field':fields,'author':authors, 'paper':papers, 'institution':institutions})
    
    # Delete the redundant edges in G0
    fields_in_G0 = [0]     # only one field
    field_papers_in_G0 = torch.tensor([]).to(device)
    field_papers_in_G0 = field_papers_in_G0.type(torch.int64)
    for s in fields_in_G0:
        field_papers_in_G0 = torch.concat([field_papers_in_G0,(G0.successors(s, etype='has'))])
    field_papers_in_G0 = torch.unique(field_papers_in_G0)
    
    etype = ('paper','citing','paper')
    src_citing_all, dst_citing_all, eid_citing_all = G0.edges(etype=etype,form='all')

    field_papers_in_G0_eid_list = []          # get all edge ids with field_papers as the source nodes
    for idx in range(0, len(src_citing_all)):
        if src_citing_all[idx] in field_papers_in_G0:
            field_papers_in_G0_eid_list.append(eid_citing_all[idx].item())

    eid_citing_delete = list(set(eid_citing_all.tolist())^set(field_papers_in_G0_eid_list))
    eid_citing_delete = th.tensor(eid_citing_delete, dtype=th.long).to(device)
    G0 = dgl.remove_edges(G0, eid_citing_delete, etype).to(device)
    G0 = G0.to('cpu')
    
    # Split train/eval/test
    train_ratio = 0.6
    eval_ratio = 0.2
    test_ratio = 1 - train_ratio - eval_ratio
    usage = 0.8
    train_data, eval_data, test_data, used_data = split_data(G0, 'citing', train_ratio, eval_ratio, test_ratio, usage)

    train_data_output = train_data
    eval_data_output = eval_data
    test_data_output = test_data
    used_data_output = used_data

    # Delete the positive edges in train/eval/test data in the graph G0
    used_pos = np.nonzero(used_data[:,2])
    used_pos_idx = used_pos[0]
    paper_paper_src_processed = used_data[used_pos_idx, 0]
    paper_paper_dst_processed = used_data[used_pos_idx, 1]
    paper_paper_src_processed = th.Tensor(paper_paper_src_processed).long()
    paper_paper_dst_processed = th.Tensor(paper_paper_dst_processed).long()

    etype = ('paper','citing','paper')
    eid_citing_train_eval_test_delete = G0.edge_ids(paper_paper_src_processed, paper_paper_dst_processed, etype='citing')
    G = dgl.remove_edges(G0, eid_citing_train_eval_test_delete, etype)
    G = G.to(device)

    train_data = torch.Tensor(train_data).long()
    eval_data = torch.Tensor(eval_data).long()
    test_data = torch.Tensor(test_data).long()
    
    G.node_dict = {}
    G.edge_dict = {}
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = th.ones(G.number_of_edges(etype), dtype=th.long).to(device) * G.edge_dict[etype] 

    # Using HGT model to obtain node embeddings 
    model = HGT(G, n_inp=128, n_hid=200, n_out=30, n_layers=2, n_heads=4, use_norm = True, MLP_dropout = 0).to(device)
    
    optimizer = th.optim.AdamW(model.parameters())
    scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=400, max_lr = 1e-3, pct_start=0.05)

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

    for epoch in range(400):
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
        
        if epoch % 80 == 0:
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


    return best_model, best_model_training_time, total_training_time, best_eval_acc, best_test_acc, time_list, train_acc_list, eval_acc_list, test_acc_list, train_data_output, eval_data_output, test_data_output, used_data_output

def transfer(target, trained_model, train_data, eval_data, test_data, used_data):
    # load data
    data_file_path = './HGTDGL/tmp/MAG.mat'
    data = scipy.io.loadmat(data_file_path)
    device = th.device("cuda:1")

    # Build the original graph G
    original_G = dgl.heterograph({
            ('author', 'writing', 'paper') : data['AvsP'].nonzero(),
            ('paper', 'written-by', 'author') : data['AvsP'].transpose().nonzero(),
            ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
            # ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
            ('paper', 'is-about', 'field') : data['PvsF'].nonzero(),
            ('field', 'has', 'paper') : data['PvsF'].transpose().nonzero(),
            ('author','affiliated-with','institution') : data['AvsI'].nonzero(),
            ('institution','hire','author') : data['AvsI'].transpose().nonzero(),
        })
    original_G = original_G.to(device)

    # Initialization
    original_G.node_dict = {}
    original_G.edge_dict = {}
    for ntype in original_G.ntypes:
        original_G.node_dict[ntype] = len(original_G.node_dict)
    for etype in original_G.etypes:
        original_G.edge_dict[etype] = len(original_G.edge_dict)
        original_G.edges[etype].data['id'] = th.ones(original_G.number_of_edges(etype), dtype=th.long).to(device) * original_G.edge_dict[etype] 
    for ntype in original_G.ntypes:
        emb = nn.Parameter(th.Tensor(original_G.number_of_nodes(ntype), 128), requires_grad = False).to(device)
        nn.init.xavier_uniform_(emb)
        original_G.nodes[ntype].data['inp'] = emb
    # Meaningful embedding for paper nodes
    original_G.nodes['paper'].data['inp'] = th.tensor(data['paper_node_feat'], dtype=torch.float32).to(device)

    # Split subgraph based on fields
    field = target
    fields = [field]
    
    field_papers = torch.tensor([]).to(device)
    field_papers = field_papers.type(torch.int64)
    for s in fields:
        field_papers = torch.concat([field_papers, (original_G.successors(s, etype='has'))])
    field_papers = torch.unique(field_papers)

    authors = torch.tensor([]).to(device)
    authors = authors.type(torch.int64)
    for p in field_papers:
        authors = torch.concat([authors, (original_G.successors(p, etype='written-by'))])
    authors = torch.unique(authors)

    citing_papers = torch.tensor([]).to(device)
    citing_papers = citing_papers.type(torch.int64)
    for p in field_papers:
        citing_papers = torch.concat([citing_papers, (original_G.successors(p, etype='citing'))])
    citing_papers = torch.unique(citing_papers)
    papers = torch.unique(torch.concat([citing_papers, field_papers]))
  
    institutions = torch.tensor([]).to(device)
    institutions = institutions.type(torch.int64)
    for a in authors:
        institutions = torch.concat([institutions, (original_G.successors(a, etype='affiliated-with'))])
    institutions = torch.unique(institutions)
    
    # G0    the graph with the 'citing' edges both from field papers and citing_papers, 
    #       the 'citing' edges from citing papers need to be deleted (if a citing paper 
    #        belongs to field papers, corresponding edges should be kept) 
    G0 = dgl.node_subgraph(original_G, {'field':fields,'author':authors, 'paper':papers, 'institution':institutions})
    
    # Delete the redundant edges in G0
    fields_in_G0 = [0]     # only one field
    field_papers_in_G0 = torch.tensor([]).to(device)
    field_papers_in_G0 = field_papers_in_G0.type(torch.int64)
    for s in fields_in_G0:
        field_papers_in_G0 = torch.concat([field_papers_in_G0,(G0.successors(s, etype='has'))])
    field_papers_in_G0 = torch.unique(field_papers_in_G0)
    
    etype = ('paper','citing','paper')
    src_citing_all, dst_citing_all, eid_citing_all = G0.edges(etype=etype,form='all')

    field_papers_in_G0_eid_list = []          # get all edge ids with field_papers as the source nodes
    for idx in range(0, len(src_citing_all)):
        if src_citing_all[idx] in field_papers_in_G0:
            field_papers_in_G0_eid_list.append(eid_citing_all[idx].item())

    eid_citing_delete = list(set(eid_citing_all.tolist())^set(field_papers_in_G0_eid_list))
    eid_citing_delete = th.tensor(eid_citing_delete, dtype=th.long).to(device)
    G0 = dgl.remove_edges(G0, eid_citing_delete, etype).to(device)
    G0 = G0.to('cpu')
    
    '''
    # Delete the positive edges in eval/test data in the graph G0
    train_pos = np.nonzero(train_data[:,2])
    train_pos_idx = train_pos[0]
    paper_paper_src_processed = train_data[train_pos_idx, 0]
    paper_paper_dst_processed = train_data[train_pos_idx, 1]
    '''

    # Delete the positive edges in train/eval/test data in the graph G0
    used_pos = np.nonzero(used_data[:,2])
    used_pos_idx = used_pos[0]
    paper_paper_src_processed = used_data[used_pos_idx, 0]
    paper_paper_dst_processed = used_data[used_pos_idx, 1]
    paper_paper_src_processed = th.Tensor(paper_paper_src_processed).long()
    paper_paper_dst_processed = th.Tensor(paper_paper_dst_processed).long()

    etype = ('paper','citing','paper')
    eid_citing_train_eval_test_delete = G0.edge_ids(paper_paper_src_processed, paper_paper_dst_processed, etype='citing')
    G = dgl.remove_edges(G0, eid_citing_train_eval_test_delete, etype)
    G = G.to(device)

    train_data = torch.Tensor(train_data).long()
    eval_data = torch.Tensor(eval_data).long()
    test_data = torch.Tensor(test_data).long()
    
    G.node_dict = {}
    G.edge_dict = {}
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        G.edges[etype].data['id'] = th.ones(G.number_of_edges(etype), dtype=th.long).to(device) * G.edge_dict[etype] 
    
    # Using HGT model to obtain node embeddings 
    model = trained_model
    optimizer = th.optim.AdamW(model.parameters())
    scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=400, max_lr = 1e-3, pct_start=0.05)

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

    for epoch in range(400):
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
        
        if epoch % 80 == 0:
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
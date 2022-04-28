import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import time
import torch as th
import joblib
import dgl.function as fn
from HGTDGL.model import *
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = './HGTDGL/tmp/ACM.mat'
urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
device = th.device("cuda:0")
# print('Keys in dataset:','\n',list(data.keys()))

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
        ('paper','publish-on','conference') : data['PvsC'].nonzero(),
    }).to(device)
# print(G)

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

# subgraph based on subjects
subjectCodes = []
for a in data['L']:
    subjectCodes.append(a[0][0])
# print('subjectCodes: ', subjectCodes)

target = 'H.2'
subject = subjectCodes.index(target)
subjects = [subject]

subject_papers = torch.tensor([]).to(device)
subject_papers = subject_papers.type(torch.int64)
for s in subjects:
    subject_papers = torch.concat([subject_papers, (G.successors(s, etype='has'))])
subject_papers = torch.unique(subject_papers)

conferences = torch.tensor([]).to(device)
conferences = conferences.type(torch.int64)
for p in subject_papers:
    conferences = torch.concat([conferences, (G.successors(p, etype='publish-on'))])
conferences = torch.unique(conferences)

authors = torch.tensor([]).to(device)
authors = authors.type(torch.int64)
for p in subject_papers:
    authors = torch.concat([authors, (G.successors(p, etype='written-by'))])
authors = torch.unique(authors)

cited_papers = torch.tensor([]).to(device)
cited_papers = cited_papers.type(torch.int64)
for p in subject_papers:
    cited_papers = torch.concat([cited_papers, (G.successors(p, etype='citing'))])
cited_papers = torch.unique(cited_papers)
papers = torch.unique(torch.concat([cited_papers, subject_papers]))

G0 = dgl.node_subgraph(G, {'subject':subjects,'author':authors, 'paper':papers, 'conference':conferences})

subjects_in_G0=[0]
subject_papers_in_G0 = torch.tensor([]).to(device)
subject_papers_in_G0 = subject_papers_in_G0.type(torch.int64)
for s in subjects_in_G0:
    subject_papers_in_G0 = torch.concat([subject_papers_in_G0, (G0.successors(s, etype='has'))])
subject_papers_in_G0 = torch.unique(subject_papers_in_G0)

etype = ('paper','publish-on','conference')
src_in_G0, dst_in_G0, eid_list_in_G0 = G0.edges(etype=etype,form='all')

# subject_papers_in_G0 --> eid_in_G0
subject_papers_eid_list_in_G0 = []
for idx in range(0, len(src_in_G0)):
    if src_in_G0[idx] in subject_papers_in_G0:
        subject_papers_eid_list_in_G0.append(eid_list_in_G0[idx].item())

# eid_delete: ID of edges not from the dst_subject_paper  --> eid_list_in_G0 - subject_papers_eid_list_in_G0
eid_delete = list(set(eid_list_in_G0.tolist())^set(subject_papers_eid_list_in_G0))
eid_delete = th.tensor(eid_delete, dtype=th.long).to(device)
G0 = dgl.remove_edges(G0, eid_delete, etype).to(device)

print(G0.nodes['paper'])

G0.node_dict = {}
G0.edge_dict = {}
for ntype in G0.ntypes:
    G0.node_dict[ntype] = len(G0.node_dict)
for etype in G0.etypes:
    G0.edge_dict[etype] = len(G0.edge_dict)
    G0.edges[etype].data['id'] = torch.ones(G0.number_of_edges(etype), dtype=torch.long).to(device) * G0.edge_dict[etype] 
    
#     Random initialize input feature
for ntype in G0.ntypes:
    emb = nn.Parameter(torch.Tensor(G0.number_of_nodes(ntype), 400), requires_grad = False).to(device)
    nn.init.xavier_uniform_(emb)
    G0.nodes[ntype].data['inp'] = emb

# Genetrate positive train/val/test split
val_ratio = 0.1
test_ratio = 0.2
val_edge_dict = {}
test_edge_dict = {}
train_edge_dict = {}
out_ntypes = []
target_link = [('paper','publish-on','conference')]

for i, etype in enumerate(target_link):
    num_edges = G0.num_edges(etype)
    random_int = th.randperm(num_edges)
    val_index = random_int[:int(num_edges * val_ratio)].to(device)
    val_edge = G0.find_edges(val_index, etype)
    test_index = random_int[int(num_edges * val_ratio):int((num_edges)* (test_ratio + val_ratio))].to(device)
    test_edge = G0.find_edges(test_index, etype)
    train_index = random_int[int((num_edges)* (test_ratio + val_ratio)):].to(device)
    train_edge = G0.find_edges(train_index, etype)
    
    val_edge_dict[etype] = val_edge
    test_edge_dict[etype] = test_edge
    train_edge_dict[etype] = train_edge

    out_ntypes.append(etype[0])
    out_ntypes.append(etype[2])
    train_graph = dgl.remove_edges(G0, th.cat((val_index, test_index)), etype).to(device)
    # val_graph = dgl.remove_edges(G0, th.cat((train_index, test_index)), etype).to(device)
    # test_graph = dgl.remove_edges(G0, th.cat((val_index, train_index)), etype).to(device)

out_ntypes = set(out_ntypes)
val_graph = dgl.heterograph(val_edge_dict,{ntype: G0.number_of_nodes(ntype) for ntype in set(out_ntypes)}).to(device)
test_graph = dgl.heterograph(test_edge_dict,{ntype: G0.number_of_nodes(ntype) for ntype in set(out_ntypes)}).to(device)

# Construct negative graph
def inter(a, b):
  return list(set(a) & set(b))

def my_custom_random(exclude, min_value, max_value, k):
  exclude=exclude
  min_value = min_value
  max_value = max_value
  k = k
  randInt = np.random.randint(min_value, max_value, (k,))
  return my_custom_random(exclude, min_value, max_value, k) if inter(randInt, exclude) else randInt

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    dst_list = dst.tolist()

    neg_dst_list = []
    exclude=[-1]
    for idx in range(0, dst.shape[0]):
        exclude[0] = dst_list[idx]
        min_value = 0
        max_value = graph.num_nodes(vtype)
        k_dst_except_exclude = my_custom_random(exclude, min_value, max_value, k)
        for idx_k in range(0,k):
            neg_dst_list.append(k_dst_except_exclude[idx_k])
    neg_dst = th.tensor(neg_dst_list).to(device)
    
    return dgl.heterograph({etype: (neg_src, neg_dst)}, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from the HGT
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

def compute_loss(pos_score, neg_score):
    scores = th.cat([pos_score, neg_score]).flatten().to(device)
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = th.cat([pos_score, neg_score]).cpu().numpy()
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]).cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_acc(pos_score, neg_score):
    scores = th.cat([pos_score, neg_score]).flatten().cpu().numpy()
    for idx in range(0,len(scores)):
        if scores[idx] < 0.5:
            scores[idx] = 0
        else:
            scores[idx] = 1
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]).cpu().numpy()
    return accuracy_score(labels, scores)
    
def score_to_label(input_scores):
    labels = input_scores
    for idx in range(0,len(labels)):
        if labels[idx] < 0.5:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return labels 

etype = ('paper','publish-on','conference')        
neg_val_graph = construct_negative_graph(val_graph, 1, etype).to(device)
neg_test_graph = construct_negative_graph(test_graph, 1, etype).to(device)
neg_train_graph = construct_negative_graph(train_graph, 1, etype).to(device)

model = HGT(train_graph, n_inp=400, n_hid=200, n_out=30, n_layers=2, n_heads=4, use_norm = True).to(device)
optimizer = th.optim.AdamW(model.parameters())
scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr = 1e-3, pct_start=0.05)
pred =  HeteroDotProductPredictor()

train_step = 0
best_val_acc = 0
best_test_acc = 0
start = time.time()
best_time = time.time()

for epoch in range(100):
    h = model(train_graph, 'paper','conference')

    pos_train_score = pred(train_graph, h, ('paper','publish-on','conference'))
    neg_train_score = pred(neg_train_graph, h, ('paper','publish-on','conference'))
    pos_val_score = pred(val_graph, h, ('paper','publish-on','conference'))
    neg_val_score = pred(neg_val_graph, h, ('paper','publish-on','conference'))
    pos_test_score = pred(test_graph, h, ('paper','publish-on','conference'))
    neg_test_score = pred(neg_test_graph, h, ('paper','publish-on','conference'))
    
    train_pred = score_to_label(th.cat([pos_train_score,neg_train_score]).flatten())
    train_labels = th.cat([th.ones(pos_train_score.shape[0]), th.zeros(neg_train_score.shape[0])]).to(device)

    val_pred = score_to_label(th.cat([pos_val_score,neg_val_score]).flatten())
    val_labels = th.cat([th.ones(pos_val_score.shape[0]), th.zeros(neg_val_score.shape[0])]).to(device)

    test_pred = score_to_label(th.cat([pos_test_score,neg_test_score]).flatten())
    test_labels = th.cat([th.ones(pos_test_score.shape[0]), th.zeros(neg_test_score.shape[0])]).to(device)

    train_acc = (train_pred == train_labels).float().mean()
    val_acc = (val_pred == val_labels).float().mean()
    test_acc = (test_pred == test_labels).float().mean()

    loss = compute_loss(pos_train_score, neg_train_score)
    
    val_loss = compute_loss(pos_val_score, neg_val_score)
    test_loss = compute_loss(pos_test_score, neg_test_score)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_time = time.time()

    train_step += 1
    scheduler.step(train_step)

    if epoch % 20 == 0:
        print('In epoch %d, loss: %.4f, val-loss: %.4f, test-loss: %.4f, train-acc: %.4f, val-acc: %.4f (Best %.4f), test-acc: %.4f (Best %.4f)' % (epoch, loss, val_loss, test_loss, train_acc, val_acc, best_val_acc.item(), test_acc, best_test_acc.item()))

end = time.time()
print("training time: ", end - start)
print("best time ", best_time - start)

with th.no_grad():
    pos_score = pred(test_graph, h, ('paper','publish-on','conference'))
    neg_score = pred(neg_test_graph, h, ('paper','publish-on','conference'))
    print('test auc:', compute_auc(pos_score, neg_score),'test acc:',compute_acc(pos_score, neg_score))

    scores = th.cat([pos_score, neg_score]).cpu().numpy()
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, sample_weight=None, drop_intermediate=None)
    plt.plot(fpr,tpr,marker = 'o')
    plt.show()

# transfer
print('Transfer')
target = 'H.3'
subject = subjectCodes.index(target)
subjects = [subject]

subject_papers = torch.tensor([]).to(device)
subject_papers = subject_papers.type(torch.int64)
for s in subjects:
    subject_papers = torch.concat([subject_papers, (G.successors(s, etype='has'))])

subject_papers = torch.unique(subject_papers)

conferences = torch.tensor([]).to(device)
conferences = conferences.type(torch.int64)
for p in subject_papers:
    conferences = torch.concat([conferences, (G.successors(p, etype='publish-on'))])

conferences = torch.unique(conferences)

authors = torch.tensor([]).to(device)
authors = authors.type(torch.int64)
for p in subject_papers:
    authors = torch.concat([authors, (G.successors(p, etype='written-by'))])
authors = torch.unique(authors)

cited_papers = torch.tensor([]).to(device)
cited_papers = cited_papers.type(torch.int64)

for p in subject_papers:
    cited_papers = torch.concat([cited_papers, (G.successors(p, etype='citing'))])
cited_papers = torch.unique(cited_papers)

papers = torch.unique(torch.concat([cited_papers, subject_papers]))
G1 = dgl.node_subgraph(G, {'subject':subjects,'author':authors, 'paper':papers, 'conference':conferences})

subjects_in_G1=[0]
subject_papers_in_G1 = torch.tensor([]).to(device)
subject_papers_in_G1 = subject_papers_in_G1.type(torch.int64)
for s in subjects_in_G1:
    subject_papers_in_G1 = torch.concat([subject_papers_in_G1, (G1.successors(s, etype='has'))])
subject_papers_in_G1 = torch.unique(subject_papers_in_G1)

etype = ('paper','publish-on','conference')
src_in_G1, dst_in_G1, eid_list_in_G1 = G1.edges(etype=etype,form='all')

# subject_papers_in_G1 --> eid_in_G1
subject_papers_eid_list_in_G1 = []
for idx in range(0, len(src_in_G1)):
    if src_in_G1[idx] in subject_papers_in_G1:
        subject_papers_eid_list_in_G1.append(eid_list_in_G1[idx].item())


# eid_delete: ID of edges not from the dst_subject_paper  --> eid_list_in_G1 - subject_papers_eid_list_in_G1
eid_delete = list(set(eid_list_in_G1.tolist())^set(subject_papers_eid_list_in_G1))
eid_delete = th.tensor(eid_delete, dtype=th.long).to(device)
G1 = dgl.remove_edges(G1, eid_delete, etype).to(device)

G1.node_dict = {}
G1.edge_dict = {}
for ntype in G1.ntypes:
    G1.node_dict[ntype] = len(G1.node_dict)
for etype in G1.etypes:
    G1.edge_dict[etype] = len(G1.edge_dict)
    G1.edges[etype].data['id'] = torch.ones(G1.number_of_edges(etype), dtype=torch.long).to(device) * G1.edge_dict[etype] 
    
#     Random initialize input feature
for ntype in G1.ntypes:
    emb = nn.Parameter(torch.Tensor(G1.number_of_nodes(ntype), 400), requires_grad = False).to(device)
    nn.init.xavier_uniform_(emb)
    G1.nodes[ntype].data['inp'] = emb

# Genetrate positive train/val/test split
val_ratio = 0.1
test_ratio = 0.2
val_edge_dict = {}
test_edge_dict = {}
train_edge_dict = {}
out_ntypes = []
target_link = [('paper','publish-on','conference')]

for i, etype in enumerate(target_link):
    num_edges = G1.num_edges(etype)
    random_int = th.randperm(num_edges)
    val_index = random_int[:int(num_edges * val_ratio)].to(device)
    val_edge = G1.find_edges(val_index, etype)
    test_index = random_int[int(num_edges * val_ratio):int((num_edges)* (test_ratio + val_ratio))].to(device)
    test_edge = G1.find_edges(test_index, etype)
    train_index = random_int[int((num_edges)* (test_ratio + val_ratio)):].to(device)
    train_edge = G1.find_edges(train_index, etype)
    
    val_edge_dict[etype] = val_edge
    test_edge_dict[etype] = test_edge
    train_edge_dict[etype] = train_edge

    out_ntypes.append(etype[0])
    out_ntypes.append(etype[2])
    train_graph = dgl.remove_edges(G1, th.cat((val_index, test_index)), etype).to(device)
    # val_graph = dgl.remove_edges(G, th.cat((train_index, test_index)), etype).to(device)
    # test_graph = dgl.remove_edges(G, th.cat((val_index, train_index)), etype).to(device)

out_ntypes = set(out_ntypes)
val_graph = dgl.heterograph(val_edge_dict,{ntype: G1.number_of_nodes(ntype) for ntype in set(out_ntypes)}).to(device)
test_graph = dgl.heterograph(test_edge_dict,{ntype: G1.number_of_nodes(ntype) for ntype in set(out_ntypes)}).to(device)

etype = ('paper','publish-on','conference')        
neg_val_graph = construct_negative_graph(val_graph, 1, etype).to(device)
neg_test_graph = construct_negative_graph(test_graph, 1, etype).to(device)
neg_train_graph = construct_negative_graph(train_graph, 1, etype).to(device)

optimizer = th.optim.AdamW(model.parameters())
scheduler = th.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=100, max_lr = 1e-3/2, pct_start=0.05)
pred =  HeteroDotProductPredictor()

train_step = 0
best_val_acc = 0
best_test_acc = 0
start = time.time()
best_time = time.time()

for epoch in range(100):
    h = model(train_graph, 'paper','conference')

    pos_train_score = pred(train_graph, h, ('paper','publish-on','conference'))
    neg_train_score = pred(neg_train_graph, h, ('paper','publish-on','conference'))
    pos_val_score = pred(val_graph, h, ('paper','publish-on','conference'))
    neg_val_score = pred(neg_val_graph, h, ('paper','publish-on','conference'))
    pos_test_score = pred(test_graph, h, ('paper','publish-on','conference'))
    neg_test_score = pred(neg_test_graph, h, ('paper','publish-on','conference'))
    
    train_pred = score_to_label(th.cat([pos_train_score,neg_train_score]).flatten())
    train_labels = th.cat([th.ones(pos_train_score.shape[0]), th.zeros(neg_train_score.shape[0])]).to(device)

    val_pred = score_to_label(th.cat([pos_val_score,neg_val_score]).flatten())
    val_labels = th.cat([th.ones(pos_val_score.shape[0]), th.zeros(neg_val_score.shape[0])]).to(device)

    test_pred = score_to_label(th.cat([pos_test_score,neg_test_score]).flatten())
    test_labels = th.cat([th.ones(pos_test_score.shape[0]), th.zeros(neg_test_score.shape[0])]).to(device)

    train_acc = (train_pred == train_labels).float().mean()
    val_acc = (val_pred == val_labels).float().mean()
    test_acc = (test_pred == test_labels).float().mean()

    loss = compute_loss(pos_train_score, neg_train_score)
    
    val_loss = compute_loss(pos_val_score, neg_val_score)
    test_loss = compute_loss(pos_test_score, neg_test_score)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_step += 1
    scheduler.step(train_step)
    
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_time = time.time()

    if epoch % 20 == 0:
        print('In epoch %d, loss: %.4f, val-loss: %.4f, test-loss: %.4f, train-acc: %.4f, val-acc: %.4f (Best %.4f), test-acc: %.4f (Best %.4f)' % (epoch, loss, val_loss, test_loss, train_acc, val_acc, best_val_acc.item(), test_acc, best_test_acc.item()))

end = time.time()
print("training time: ", end - start)
print("best time ", best_time - start)

with th.no_grad():
    pos_score = pred(test_graph, h, ('paper','publish-on','conference'))
    neg_score = pred(neg_test_graph, h, ('paper','publish-on','conference'))
    print('test auc:', compute_auc(pos_score, neg_score),'test acc:',compute_acc(pos_score, neg_score))

    scores = th.cat([pos_score, neg_score]).cpu().numpy()
    labels = th.cat([th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, sample_weight=None, drop_intermediate=None)
    plt.plot(fpr,tpr,marker = 'o')
    plt.show()
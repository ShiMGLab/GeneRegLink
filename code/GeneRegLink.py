import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.function as fn
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl.function as fn
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import roc_auc_score

from scipy import sparse
from built_WGCN import *
import dgl.data
import pandas as pd
Embedding_dims=128
auc_list = []
ap_list = []
class load_data():
    def __init__(self, data, normalize=True):
            self.data = data
            self.normalize = normalize

    def data_normalize(self, data):
            standard = StandardScaler()
            epr = standard.fit_transform(data.T)

            return epr.T

    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


feature = pd.read_csv('./data/Specific Dataset/hESC 500/BL--ExpressionData.csv').iloc[:, 1:]
loader = load_data(feature)
feature= loader.exp_data()
feature= torch.tensor(feature, dtype=torch.float32)

######################################################################
# Prepare training and testing setsdataset_knn = {KnnDGLFromDGL: 1} Dataset("STRENG", num_graphs=1, save_path=C:\Users\yypz\.dgl\STRENG)
# Split edge set for training and testing
train = pd.read_csv('./data/Specific Dataset/hESC 500/Train_set.csv')
val_g = pd.read_csv('./data/Specific Dataset/hESC 500/Validation_set.csv')
test_g = pd.read_csv('./data/Specific Dataset/hESC 500/Test_set.csv')
train_pos_number = train["Label"].value_counts()[1]
val_pos_number = val_g["Label"].value_counts()[1]
test_pos_number = test_g["Label"].value_counts()[1]
train_g = dgl.graph((train["TF"][:train_pos_number], train["Target"][:train_pos_number]), num_nodes=feature.shape[0])
train_g.ndata['feat'] = feature
dataset_knn = Built_WGCN("STRENG", train_g.ndata['feat'], topK=20, num_nodes=feature.shape[0])
    # build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, 256, 'mean')
        self.conv2 = SAGEConv(256, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = F.normalize(h, dim=1)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats, 32)
        self.W2 = nn.Linear(32, 1)

    def apply_edges(self, edges):
        h = torch.mul(edges.src['h'], edges.dst['h'])
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
        return g.edata['score']

class Attention(nn.Module):
    def __init__(self, emb_dim, hidden_size=64):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
                # nn.Tanh(), # 激活
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)  # n * 1 # [265 2 32]
            # print ('w')
            # print (w.shape)
        beta = torch.softmax(w, dim=1)  # eq. 9
        return (beta * z).sum(1), beta


train_pos_g = dgl.graph((train["TF"][:train_pos_number], train["Target"][:train_pos_number]), num_nodes=feature.shape[0])
train_neg_g = dgl.graph((train["TF"][train_pos_number:].to_numpy(), train["Target"][train_pos_number:].to_numpy()),
                            num_nodes=feature.shape[0])
val_pos_g = dgl.graph((val_g["TF"][:val_pos_number], val_g["Target"][:val_pos_number]), num_nodes=feature.shape[0])
val_neg_g = dgl.graph((val_g["TF"][val_pos_number:].to_numpy(), val_g["Target"][val_pos_number:].to_numpy()),
                          num_nodes=feature.shape[0])
test_pos_g = dgl.graph((test_g["TF"][:test_pos_number], test_g["Target"][:test_pos_number]), num_nodes=feature.shape[0])
test_neg_g = dgl.graph((test_g["TF"][test_pos_number:].to_numpy(), test_g["Target"][test_pos_number:].to_numpy()),
                           num_nodes=feature.shape[0])





attention = Attention(Embedding_dims)
model = GraphSAGE(train_g.ndata['feat'].shape[1], Embedding_dims)
model_knn = GraphSAGE(train_g.ndata['feat'].shape[1], Embedding_dims)
pred = MLPPredictor(Embedding_dims)
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def compute_ap(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()  # 真实标签值
    y_test = labels
    y_pred = scores
    ap = average_precision_score(y_test, y_pred)
    return ap


optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- training -------------------------------- #
for e in range(100):
    h = model(train_g, train_g.ndata['feat'])# 调用模型的Forward函数
    h_knn= model(dataset_knn.graph, train_g.ndata['feat'])
    h = torch.stack([h, h_knn], dim=1)  # [265,2,32]
    h, att = attention(h)

    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
        # loss += 1e+2 * bp_decoder.loss_full(h, sp_adj)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 5 == 0:
        with torch.no_grad():
            pos_score = pred(val_pos_g, h)
            neg_score = pred(val_neg_g, h)
            auc = compute_auc(pos_score, neg_score)
            ap = compute_ap(pos_score, neg_score)
            print('In epoch {}, loss: {:.3f}, AUC = {:.3f}, AP = {:.3f}'.format(e, loss, auc, ap))
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    auc = compute_auc(pos_score, neg_score)
    ap = compute_ap(pos_score, neg_score)


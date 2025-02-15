from dgl.data import DGLDataset
import dgl
import torch
import torch.nn.functional as F
import dgl.data
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
file_path = "datasets\\"
from scipy.stats import spearmanr
class Built_WGCN(DGLDataset):
    def __init__(self, name, feat_data, topK , num_nodes = 0):
        self.feat_data = feat_data
        #self.components = num_classes
        self.topk = topK
        self.num_nodes = num_nodes
        super(Built_WGCN, self).__init__(name = name)
    def process(self):

        labels = []  # 用于存放每个节点对应类别的列表


        # 属性信息

        adj_str = []
        adj_end = []

        sim_feat= np.corrcoef(self.feat_data)
        print(sim_feat.shape)
        #sim_feat=spearmanr(self.feat_data)
        sim_feat=np.abs(sim_feat)
        inds = []
        for i in range(sim_feat.shape[0]):
            ind = np.argpartition(sim_feat[i, :], -(self.topk + 1))[-(self.topk + 1):]
            inds.append(ind)
        for i, vs in enumerate(inds):
            for v in vs:
                if v == i: # 自己与自己不构成环
                    pass
                else:
                    adj_str.append(i)
                    adj_end.append(v)
        self.estr = adj_str + adj_end
        self.eend = adj_end + adj_str
        #feat_data = torch.tensor(self.feat_data)
        self.graph = dgl.graph((self.estr, self.eend), num_nodes = self.num_nodes)
        self.graph.ndata['feat'] = self.feat_data # feature
    def __getitem__(self, i):
        return self.graph

    @property
    def num_classes(self):
        return self.components

    def __len__(self):
        return 1
    # @property
    def edges(self):
        return torch.from_numpy(np.array(self.estr)), torch.from_numpy(np.array(self.eend))
    # @property
    def number_of_edges(self):
        return self.graph.adjacency_matrix().to_dense().float().sum().sum()
    def number_of_nodes(self):
        return self.num_nodes
from typing import List
import numpy as np
import scipy.sparse as sp
import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter


att_op_dict = {
    'sum': 'sum',
    'mul': 'mul',
    'concat': 'concat'
}

def normalize_adj(adj: sp.csr_matrix):
    """ Symmetrically normalize adjacency matrix. """
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(np.array(adj.sum(axis=1)), -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_diag = np.diag(d_inv_sqrt)
    return adj.dot(d_inv_sqrt_diag).transpose().dot(d_inv_sqrt_diag)

def preprocess_adjs(adjs: List[sp.csr_matrix]):
    """ Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. """
    max_length = max([a.shape[0] for a in adjs])
    mask = np.zeros((len(adjs), max_length, 1))         # mask for padding

    for i, adj in enumerate(adjs):
        adj_normalized = normalize_adj(adj)             # no self-loop
        pad = max_length - adj_normalized.shape[0]      # padding for each epoch

        adjs[i] = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adj.shape[0], :] = 1.
    return np.array(adjs), mask

def preprocess_features(features: List[np.ndarray]):
    """ Row-normalize feature matrix and convert to tuple representation """
    max_length = max([len(feature) for feature in features])

    for i, feature in enumerate(features):
        feature = np.array(feature)
        pad = max_length - feature.shape[0]         # padding for each epoch

        features[i] = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
    return np.array(features)


class GatedGNN(nn.Module):

    def __init__(self, feature_size, hidden_size, num_gnn_layers, dropout=0.5, act=f.relu):
        super(GatedGNN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.emb_encode = nn.Linear(feature_size, hidden_size).to(torch.float64)
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.z1 = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.r0 = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.r1 = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.h0 = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.h1 = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.soft_att = nn.Linear(hidden_size, 1).to(torch.float64)
        self.ln = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.to(torch.float64))
        z1 = self.z1(x.to(torch.float64))
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.to(torch.float64)) + self.r1(x.to(torch.float64)))
        # update embeddings
        h = self.act(self.h0(a.to(torch.float64)) + self.h1(r.to(torch.float64) * x.to(torch.float64)))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.to(torch.float64))
        x = x * mask
        for idx_layer in range(self.num_gnn_layers):
            x = self.gatedGNN(x.to(torch.float64), adj.to(torch.float64)) * mask.to(torch.float64)
        return x


class ResGatedGNN(GatedGNN):

    """GatedGNN with residual connection"""

    def __init__(self, feature_size, hidden_size, num_gnn_layers, dropout,
                 act=f.relu, residual=True, att_op='mul'):
        super(ResGatedGNN, self).__init__(feature_size, hidden_size, num_gnn_layers, dropout, act)
        self.residual = residual
        self.att_op = att_op
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.to(torch.float64))
        z1 = self.z1(x.to(torch.float64))
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.to(torch.float64)) + self.r1(x.to(torch.float64)))
        # update embeddings
        h = self.act(self.h0(a.to(torch.float64)) + self.h1(r.to(torch.float64) * x.to(torch.float64)))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x.to(torch.float64))
        x = x * mask
        for idx_layer in range(self.num_gnn_layers):
            if self.residual:
                # add residual connection, can use a weighted sum
                x = x + self.gatedGNN(x.to(torch.float64), adj.to(torch.float64)) * mask.to(torch.float64)
            else:
                x = self.gatedGNN(x.to(torch.float64), adj.to(torch.float64)) * mask.to(torch.float64)
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum/mean and max pooling

        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


class GraphConvolution(torch.nn.Module):

    """ Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """

    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)
        support = torch.matmul(x.to(torch.float64), self.weight.to(torch.float64))
        output = torch.matmul(adj.to(torch.float64), support.to(torch.float64))
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class ResGCN(nn.Module):

    """GCNs with residual connection"""

    def __init__(self, feature_size, hidden_size, num_gnn_layers, dropout,
                 act=f.relu, residual=True, att_op="mul"):
        super(ResGCN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.residual = residual
        self.att_op = att_op
        self.out_dim = hidden_size
        if self.att_op == att_op_dict['concat']:
            self.out_dim = hidden_size * 2

        self.gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_gnn_layers):
            if layer == 0:
                self.gnnlayers.append(GraphConvolution(feature_size, hidden_size, dropout, act=act))
            else:
                self.gnnlayers.append(GraphConvolution(hidden_size, hidden_size, dropout, act=act))
        self.soft_att = nn.Linear(hidden_size, 1).to(torch.float64)
        self.ln = nn.Linear(hidden_size, hidden_size).to(torch.float64)
        self.act = act

    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_gnn_layers):
            if idx_layer == 0:
                x = self.gnnlayers[idx_layer](x, adj) * mask
            else:
                if self.residual:
                    x = x + self.gnnlayers[idx_layer](x, adj) * mask  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnnlayers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x.to(torch.float64)).to(torch.float64))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.att_op == att_op_dict['sum']:
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == att_op_dict['concat']:
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings

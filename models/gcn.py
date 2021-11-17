"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import numpy
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 zeros=False):
        super(GCN, self).__init__()

        self.enable_zeros = zeros
        super(GCN, self).__init__()
        self.g = g
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(
            in_feats, n_hidden,  activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(
                n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stats = dict()
        for i in range(n_layers):
            self.layer_stats[i] = []

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if self.enable_zeros:
                #count all zeros:
                non_zeros=torch.count_nonzero(h)
                all_element=torch.numel(h)
                self.layer_stats[i].append((non_zeros,all_element))
        
        return h

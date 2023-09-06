from activation import get
import numpy as np
import re
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv,ChebConv, SGConv, GATConv, APPNP,GINConv, JumpingKnowledge
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull
from torch_geometric.utils import get_laplacian
import math
import argparse
import os
import os.path as osp
import pandas as pd


def projection_simplex(v, radius=1):
    """
    Pytorch implementation (maybe not optimal) of the projection into the simplex.
    """
    n_feat = v.shape[1]
    n_neuron = v.shape[0]
    u, _ = torch.sort(v)
    cssv = torch.cumsum(u, dim=1) - radius
    ind = torch.arange(n_feat, device=v.device).repeat(n_neuron, 1) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond].reshape(n_neuron, n_feat)[:, -1]
    theta = torch.div(cssv[cond].reshape(n_neuron, n_feat)[:, -1], rho)
    relu = torch.nn.ReLU()
    return relu(v - theta.reshape(n_neuron, 1))

def constraint(module):
    if module._get_name() == 'Linear':
        with torch.no_grad():
            module.weight = torch.nn.Parameter(projection_simplex(module.weight))
    return module

def linear_with_init(num_in, num_out, init='rand', weight_norm=False):

    if init == 'zero':
        lin_tmp = nn.Linear(num_in, num_out)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(torch.zeros_like(lin_tmp.weight))
            lin_tmp.bias = torch.nn.Parameter(torch.zeros_like(lin_tmp.bias))
    elif init == 'uniform':
        lin_tmp = nn.Linear(num_in, num_out)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(torch.ones_like(lin_tmp.weight) / num_in)
            lin_tmp.bias = torch.nn.Parameter(torch.ones_like(lin_tmp.bias) / num_in)
    elif init == 'identity':
        lin_tmp = nn.Linear(num_in, num_out)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(torch.eye(num_out, num_in))
            lin_tmp.bias = torch.nn.Parameter(torch.zeros_like(lin_tmp.bias))
    elif init == 'simplex':
        lin_tmp = nn.Linear(num_in, num_out, bias=False)
        with torch.no_grad():
            lin_tmp.weight = torch.nn.Parameter(projection_simplex(lin_tmp.weight))
    else:
        lin_tmp = nn.Linear(num_in, num_out)

    return torch.nn.utils.weight_norm(lin_tmp, name='weight') if weight_norm else lin_tmp


class GCNNet(nn.Module):

    def __init__(self,
                 num_features,
                 num_classes,
                 nhid = 16,
                 activation = 'atan',
                 dropout_prob=0.3,
                 version = 'bregman'):

        super(GCNNet, self).__init__()
        activation = activation.lower()
        version =  version.lower()
        # Hidden layers
        self.layers = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        self.activation, self.offset, self.range = get(activation_name=activation, version=version)
        self.drop1 = nn.Dropout(dropout_prob)

        # To do ...
        self.reparametrization.append(linear_with_init(num_features, nhid, init='simplex'))
        self.reparametrization.append(nn.Identity())  # return input as output
        self.reparametrization.append(linear_with_init(nhid, num_classes, init='simplex'))

        # Bregman GCN layers  (GCN can be replaced with GAT, ChebNet etc.)
        self.layers.append(GCNConv(in_channels = num_features, out_channels = nhid))
        # for bregman
        self.layers.append(GCNConv(in_channels= nhid, out_channels=nhid))
        # Output
        self.layers.append(GCNConv(in_channels=nhid, out_channels=num_classes))

    def forward(self, data, edge_index):
        x = data.x  # x has shape [num_nodes, num_input_features]
        self.reparametrization[0] = constraint(self.reparametrization[0])
        x_offset = torch.clamp(self.reparametrization[0](x), self.range[0], self.range[1])  # 限制self.reparametrization[0](x)的大小
        x = self.activation(self.offset(x_offset) + self.layers[0](x, edge_index))
        x = self.drop1(x)

        x_offset = torch.clamp(self.reparametrization[1](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[1](x, edge_index))
        x = self.drop1(x)

        self.reparametrization[2] = constraint(self.reparametrization[2])
        x_offset = torch.clamp(self.reparametrization[2](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[2](x, edge_index))
        #x = self.drop1(x)

        return F.log_softmax(x, dim=1)

class GATNet(nn.Module):

    def __init__(self,
                 num_features,
                 num_classes,
                 nhid = 16,
                 activation = 'atan',
                 dropout_prob=0.3,
                 version = 'bregman'):

        super(GATNet, self).__init__()
        activation = activation.lower()
        version = version.lower()
        # Hidden layers
        self.layers = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        self.activation, self.offset, self.range = get(activation_name=activation, version=version)
        self.drop1 = nn.Dropout(dropout_prob)


        # To do ...
        self.reparametrization.append(linear_with_init(num_features, nhid, init='simplex'))
        self.reparametrization.append(nn.Identity())  # return input as output
        self.reparametrization.append(linear_with_init(nhid, num_classes, init='simplex'))

        # Bregman GCN layers  (GCN can be replaced with GAT, ChebNet etc.)
        self.layers.append(GATConv(in_channels = num_features, out_channels = nhid))
        # for bregman
        self.layers.append(GATConv(in_channels= nhid, out_channels=nhid))
        # Output
        self.layers.append(GATConv(in_channels=nhid, out_channels=num_classes))


    def forward(self, data, edge_index):
        x = data.x  # x has shape [num_nodes, num_input_features]
        self.reparametrization[0] = constraint(self.reparametrization[0])
        x_offset = torch.clamp(self.reparametrization[0](x), self.range[0], self.range[1])  # 限制self.reparametrization[0](x)的大小
        x = self.activation(self.offset(x_offset) + self.layers[0](x, edge_index))
        x = self.drop1(x)

        x_offset = torch.clamp(self.reparametrization[1](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[1](x, edge_index))
        x = self.drop1(x)

        self.reparametrization[2] = constraint(self.reparametrization[2])
        x_offset = torch.clamp(self.reparametrization[2](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[2](x, edge_index))
        #x = self.drop1(x)

        return F.log_softmax(x, dim=1)

class SAGENet(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 nhid = 16,
                 activation = 'atan',
                 dropout_prob=0.3,
                 version = 'bregman'):

        super(SAGENet, self).__init__()
        activation = activation.lower()
        version =  version.lower()
        # Hidden layers
        self.layers = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        self.activation, self.offset, self.range = get(activation_name=activation, version=version)
        self.drop1 = nn.Dropout(dropout_prob)

        # To do ...
        self.reparametrization.append(linear_with_init(num_features, nhid, init='simplex'))
        self.reparametrization.append(nn.Identity())  # return input as output
        self.reparametrization.append(linear_with_init(nhid, num_classes, init='simplex'))

        # Bregman GCN layers  (GCN can be replaced with GAT, ChebNet etc.)
        self.layers.append(SAGEConv(in_channels = num_features, out_channels = nhid))
        # for bregman
        self.layers.append(SAGEConv(in_channels= nhid, out_channels=nhid))
        # Output
        self.layers.append(SAGEConv(in_channels=nhid, out_channels=num_classes))


    def forward(self, data, edge_index):
        x = data.x  # x has shape [num_nodes, num_input_features]
        self.reparametrization[0] = constraint(self.reparametrization[0])
        x_offset = torch.clamp(self.reparametrization[0](x), self.range[0], self.range[1])  # 限制self.reparametrization[0](x)的大小
        x = self.activation(self.offset(x_offset) + self.layers[0](x, edge_index))
        x = self.drop1(x)

        x_offset = torch.clamp(self.reparametrization[1](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[1](x, edge_index))
        x = self.drop1(x)

        self.reparametrization[2] = constraint(self.reparametrization[2])
        x_offset = torch.clamp(self.reparametrization[2](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[2](x, edge_index))
        #x = self.drop1(x)

        return F.log_softmax(x, dim=1)

class ChebNet(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 nhid = 16,
                 activation = 'atan',
                 dropout_prob=0.3,
                 version = 'bregman'):

        super(ChebNet, self).__init__()
        activation = activation.lower()
        version =  version.lower()
        # Hidden layers
        self.layers = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        self.activation, self.offset, self.range = get(activation_name=activation, version=version)
        self.drop1 = nn.Dropout(dropout_prob)

        # To do ...
        self.reparametrization.append(linear_with_init(num_features, nhid, init='simplex'))
        self.reparametrization.append(nn.Identity())  # return input as output
        self.reparametrization.append(linear_with_init(nhid, num_classes, init='simplex'))

        # Bregman GCN layers  (GCN can be replaced with GAT, ChebNet etc.)
        self.layers.append(ChebConv(in_channels = num_features, out_channels = nhid, K=2))
        # for bregman
        self.layers.append(ChebConv(in_channels= nhid, out_channels=nhid, K=2))
        # Output
        self.layers.append(ChebConv(in_channels=nhid, out_channels=num_classes, K=2))

    def forward(self, data, edge_index):
        x = data.x  # x has shape [num_nodes, num_input_features]
        self.reparametrization[0] = constraint(self.reparametrization[0])
        x_offset = torch.clamp(self.reparametrization[0](x), self.range[0], self.range[1])  # 限制self.reparametrization[0](x)的大小
        x = self.activation(self.offset(x_offset) + self.layers[0](x, edge_index))
        x = self.drop1(x)

        x_offset = torch.clamp(self.reparametrization[1](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[1](x, edge_index))
        x = self.drop1(x)

        self.reparametrization[2] = constraint(self.reparametrization[2])
        x_offset = torch.clamp(self.reparametrization[2](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[2](x, edge_index))
        #x = self.drop1(x)

        return F.log_softmax(x, dim=1)

class APPNPNet(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 nhid = 16,
                 activation = 'atan',
                 K=10,
                 alpha=0.1,
                 dropout_prob=0.3,
                 version = 'bregman'):

        super(APPNPNet, self).__init__()
        activation = activation.lower()
        version =  version.lower()
        # Hidden layers
        self.lins = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        self.activation, self.offset, self.range = get(activation_name=activation, version=version)
        self.drop1 = nn.Dropout(dropout_prob)

        # To do ...
        self.reparametrization.append(linear_with_init(num_features, nhid, init='simplex'))
        self.reparametrization.append(nn.Identity())  # return input as output
        self.reparametrization.append(linear_with_init(nhid, num_classes, init='simplex'))

        self.lins.append(torch.nn.Linear(num_features, nhid))
        self.lins.append(torch.nn.Linear(nhid, nhid))
        self.lins.append(torch.nn.Linear(nhid, num_classes))

        # Bregman GCN layers  (GCN can be replaced with GAT, ChebNet etc.)
        self.layers.append(APPNP(K, alpha))
        # for bregman
        self.layers.append(APPNP(K, alpha))
        # Output
        self.layers.append(APPNP(K, alpha))

    def forward(self, data, edge_index):
        x = data.x  # x has shape [num_nodes, num_input_features]
        self.reparametrization[0] = constraint(self.reparametrization[0])
        x_offset = torch.clamp(self.reparametrization[0](x), self.range[0], self.range[1])
        x = self.lins[0](x)
        x = self.activation(self.offset(x_offset) + self.layers[0](x, edge_index))
        x = self.drop1(x)

        x_offset = torch.clamp(self.reparametrization[1](x), self.range[0], self.range[1])
        x = self.lins[1](x)
        x = self.activation(self.offset(x_offset) + self.layers[1](x, edge_index))
        x = self.drop1(x)

        self.reparametrization[2] = constraint(self.reparametrization[2])
        x_offset = torch.clamp(self.reparametrization[2](x), self.range[0], self.range[1])
        x = self.lins[2](x)
        x = self.activation(self.offset(x_offset) + self.layers[2](x, edge_index))
        #x = self.drop1(x)

        return F.log_softmax(x, dim=1)

class GINNet(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 nhid = 16,
                 activation = 'atan',
                 dropout_prob=0.3,
                 version = 'bregman'):

        super(GINNet, self).__init__()
        activation = activation.lower()
        version =  version.lower()
        # Hidden layers
        self.lins = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        self.activation, self.offset, self.range = get(activation_name=activation, version=version)
        self.drop1 = nn.Dropout(dropout_prob)

        # To do ...
        self.reparametrization.append(linear_with_init(num_features, nhid, init='simplex'))
        self.reparametrization.append(nn.Identity())  # return input as output
        self.reparametrization.append(linear_with_init(nhid, num_classes, init='simplex'))

        self.lins.append(torch.nn.Linear(num_features, nhid))
        self.lins.append(torch.nn.Linear(nhid, nhid))
        self.lins.append(torch.nn.Linear(nhid, num_classes))

        # Bregman GCN layers  (GCN can be replaced with GAT, ChebNet etc.)
        self.layers.append(GINConv(self.lins[0]))
        # for bregman
        self.layers.append(GINConv(self.lins[1]))
        # Output
        self.layers.append(GINConv(self.lins[2]))

    def forward(self, data, edge_index):
        x = data.x  # x has shape [num_nodes, num_input_features]
        self.reparametrization[0] = constraint(self.reparametrization[0])
        x_offset = torch.clamp(self.reparametrization[0](x), self.range[0],
                               self.range[1])  # 限制self.reparametrization[0](x)的大小
        x = self.activation(self.offset(x_offset) + self.layers[0](x, edge_index))
        x = self.drop1(x)

        x_offset = torch.clamp(self.reparametrization[1](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[1](x, edge_index))
        x = self.drop1(x)

        self.reparametrization[2] = constraint(self.reparametrization[2])
        x_offset = torch.clamp(self.reparametrization[2](x), self.range[0], self.range[1])
        x = self.activation(self.offset(x_offset) + self.layers[2](x, edge_index))
        # x = self.drop1(x)

        return F.log_softmax(x, dim=1)


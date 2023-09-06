import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor
from torch_geometric.utils import num_nodes
from torch_geometric.utils import to_undirected
from torch_geometric.utils import get_laplacian
from scipy import sparse




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

def load_data(args):
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), '../../Data', dataname)
    dataset = args.dataset.lower()

    if dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=rootname, name=dataname) # root=the directory the data is stored, name=the name of dataset to load
    elif dataset in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=rootname, name=dataname)
    elif dataset == 'actor':
        dataset = Actor(root=rootname)
    elif dataset in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=rootname, name=dataname)
    elif dataset in ['computers', 'photo']:
        dataset = Amazon(root=rootname, name=dataname)
    elif dataset in ['cs', 'physics']:
        dataset = Coauthor(root=rootname, name=dataname)

    num_nodes = dataset[0].x.shape[0]
    nfeatures = dataset[0].x.shape[1]
    data = dataset[0].to(device)
    edge_index = dataset[0].edge_index.to(device)

    return dataset, data, num_nodes, nfeatures, edge_index


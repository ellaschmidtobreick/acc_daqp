import numpy as np
from ctypes import * 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.utils.class_weight import compute_class_weight
import wandb

import torch
from torch_geometric.loader import DataLoader

from generate_graph_data import generate_qp_graphs_train_val, generate_qp_graphs_train_val_flexible_H
import config 
from model import GNN
from model import EarlyStopping


# Set parameters
n = 2 #config.n
m = 5 #config.m

nth = config.nth
seed = config.seed
data_points = 10 #config.data_points 
lr = config.lr
number_of_epochs = config.number_of_epochs  
layer_width = config.layer_width
number_of_layers = config.number_of_layers
track_on_wandb = config.track_on_wandb
t = config.t # tuned by gridsearch threshold = np.arange(0.1,1,0.1)


# Generate QP problems and the corresponding graphs
#graph_train, graph_val,H,A = generate_qp_graphs_train_val(n,m,nth,seed,data_points)
np.random.seed(123)
f = np.random.randn(n)
F = np.random.randn(n,nth)
print(f)
print(F)
#print(H)
#print(A)
generate_qp_graphs_train_val_flexible_H(n,m,nth,seed,data_points)
# graph_train,n_train, m_train = generate_qp_graphs_different_sizes(n,n,m,m,nth,seed,data_points,"train",H=H,A =A)
# graph_val,n_val, m_val = generate_qp_graphs_different_sizes(n,n,m,m,nth,seed,data_points,"val",H=H,A =A)

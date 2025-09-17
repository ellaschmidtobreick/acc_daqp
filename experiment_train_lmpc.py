
import numpy as np
from generate_mpqp_v2 import generate_rhs 
from generate_graph_data import generate_qp_graphs_train_val_lmpc, generate_qp_graphs_test_data_only_lmpc
from train_model import train_GNN
from test_model import test_GNN

# data = np.load('data/mpc_mpqp_N50.npz')
# print(data.files)

# print(data['H'])    # 10x10
# print(data['f'])    # 10x1 (all 0)
# print(data['f_theta'])  # 10x7
# print(data['A'])    # (10+10)x10 (upper & lower constraints)
# print(data['b'])    # (10+10)x1 (all 2)
# print(data['W'])    # (10+10)x7 (all 0)


# Set parameters
n = [50]
m = [100]
nth = 7
seed = 123
data_points = 100 #5000
lr = 0.001
number_of_max_epochs = 20 #100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
t = 0.99
scale = 0.01

#scale_H = [1,0.1,0.01,0.001,0.0001]
#t_vector = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99] - 0.99 provides best results - scale again
#t_vector = [0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]
#for i in range(len(scale_H)):
# for i in range(len(t_vector)):
#     print("threshold:", t_vector[i])
#     #scale = scale_H[i]
#     t = t_vector[i]


n_number = n[0]
print(n_number)

train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc",scale_H=scale)
test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc")


# train_GNN([50],[100],7,123, 5000,0.001,20,128,3, True,0.6, False,False,f"model_{50}v_{100}c_lmpc")
# test_GNN([50],[100],7,123, 5000,128,3,0.6, False,False,f"model_{50}v_{100}c_lmpc")

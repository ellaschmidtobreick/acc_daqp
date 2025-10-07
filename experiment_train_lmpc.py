
import numpy as np
from generate_mpqp_v2 import generate_rhs 
from generate_graph_data import generate_qp_graphs_train_val_lmpc, generate_qp_graphs_test_data_only_lmpc
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import matplotlib.pyplot as plt
import time

# data = np.load('data/mpc_mpqp_N50.npz')
# print(data.files)

# print(data['H'])    # 10x10
# print(data['f'])    # 10x1 (all 0)
# print(data['f_theta'])  # 10x7
# print(data['A'])    # (10+10)x10 (upper & lower constraints)
# print(data['b'])    # (10+10)x1 (all 2)
# print(data['W'])    # (10+10)x7 (all 0)


# Set parameters
n = [10,25]
m = [20,50]
nth = 7
seed = 123
data_points = 5000 #5000
lr = 0.001
number_of_max_epochs = 100 # 20 #100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
t = 0.99 #9
scale = 0.01
n_number = n[0]

#scale_H = [1,0.1,0.01,0.001,0.0001]
#t_vector = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99] - 0.99 provides best results - scale again
#t_vector = [0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]
#for i in range(len(scale_H)):
# for i in range(len(t_vector)):
#     print("threshold:", t_vector[i])
#     #scale = scale_H[i]
#     t = t_vector[i]

# recall_scores = []
# precision_scores = []

# threshold tuning
# for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#     print("threshold:", t)

# train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc",scale_H=scale,dataset_type="lmpc")
# text_time_before, text_time_after, test_time_reduction, prediction_time = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc",dataset_type="lmpc")

# n = [50]
# m = [100]
# text_time_before, text_time_after, test_time_reduction, prediction_time = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc",dataset_type="lmpc")



# Fill the table including std
# text_time_before_vector , text_time_after_vector, test_time_reduction_vector, prediction_time_vector = [], [], [], []
# for i in range(5):
#     # train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc",scale_H=scale,dataset_type="lmpc")
#     # text_time_before, text_time_after, test_time_reduction, prediction_time = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n_number}v_{2*n_number}c_lmpc",dataset_type="lmpc")
    
#     # test MLP on given lmpc data 
#     train_MLP(n,m,7, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,0.99, False,False,"MLP_model_25v_50c_fixedHA_lmpc",dataset_type="lmpc")
#     text_time_before, text_time_after, test_time_reduction, prediction_time = test_MLP(n,m,7, seed, data_points,layer_width,number_of_layers,0.99, False,False,"MLP_model_25v_50c_fixedHA_lmpc",dataset_type="lmpc")



#     text_time_before_vector.append(text_time_before)
#     text_time_after_vector.append(text_time_after)
#     test_time_reduction_vector.append(test_time_reduction)
#     prediction_time_vector.append(prediction_time)
#     # recall_scores.append(recall)
#     # precision_scores.append(precision)

# print(f"Average test time before: {np.mean(text_time_before_vector), np.std(text_time_before_vector)}")
# print(f"Average test time after: {np.mean(text_time_after_vector), np.std(text_time_after_vector)}")   
# print(f"Average test time reduction: {np.mean(test_time_reduction_vector), np.std(test_time_reduction_vector)}")
# print(f"Average prediction time: {np.mean(prediction_time_vector), np.std(prediction_time_vector)}")

# n = [10] #5,10, 20,50
# m = [20] # 206,216, 236, 296

# train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n[0]}v_{m[0]}c_lmpc",scale_H=scale,dataset_type="lmpc")
# prediction_time,text_time_after = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n[0]}v_{m[0]}c_lmpc",dataset_type="lmpc")

n = [10] #5,10, 20,50
m = [216] # 206,216, 236, 296

train_time_start = time.time()
train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n[0]}v_{m[0]}c_lmpc",scale_H=scale,dataset_type="lmpc")
train_time_end = time.time()
# n=[20]
# m=[40]
prediction_time,text_time_after = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n[0]}v_{m[0]}c_lmpc",dataset_type="lmpc")


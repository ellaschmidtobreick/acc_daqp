from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP

import matplotlib.pyplot as plt
import time
# Set parameters
n = [20, 40, 60] #[10]
m = [40,80,120] #[40]
# n = [500]
# m = [1250]
nth = 7 #2
seed = 123
data_points = 1000 #5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
t = 0.6

#GNN
print("---- GNN ----")
train_time_start = time.time()
train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard")
train_time_end = time.time()
print(f"Training time (s): {train_time_end - train_time_start}")
# test_time_start = time.time()
# test_acc, test_prec, test_rec, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard") 
# test_time_end = time.time()
# print(f"Testing time (s): {test_time_end - test_time_start}")
n = [80] #[10]
m = [160]
print("--- Testing on larger problem ---")
test_acc, test_prec, test_rec, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard") 

n = [85] #[10]
m = [170]
print("--- Testing on larger problem ---")
test_acc, test_prec, test_rec, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard") 


n = [100] #[10]
m = [200]
print("--- Testing on larger problem ---")
test_acc, test_prec, test_rec, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard") 

# MLP 
# print()
# print("---- MLP ----")
# train_time_start = time.time()
# train_MLP(n,m,7, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,0.99, False,False,"MLP_model_100v_250c_fixedHA_lmpc",dataset_type="standard")
# train_time_end = time.time()
# print(f"Training time (s): {train_time_end - train_time_start}")
# test_time_start = time.time()
# MLP_test_acc, MLP_test_prec, MLP_test_rec, MLP_test_f1 = test_MLP(n,m,7, seed, data_points,layer_width,number_of_layers,0.99, False,False,"MLP_model_100v_250c_fixedHA_lmpc",dataset_type="standard")
# test_time_end = time.time()
# print(f"Testing time (s): {test_time_end - test_time_start}")
# n = [80] #[10]
# m = [160]
# print("--- Testing on larger problem ---")
# MLP_test_acc, MLP_test_prec, MLP_test_rec, MLP_test_f1 = test_MLP(n,m,7, seed, data_points,layer_width,number_of_layers,0.99, False,False,"MLP_model_100v_250c_fixedHA_lmpc",dataset_type="standard")


# n_vector = list(range(1, 16, 1))
# m_vector = list(range(4, 61, 4))
# n_vector = [1,2]
# m_vector = [4,8]
# nth = 7 #2

# test_acc_list = []
# test_prec_list = []
# test_rec_list = []
# test_f1_list = []

# MLP_test_acc_list = []
# MLP_test_prec_list = []
# MLP_test_rec_list = []
# MLP_test_f1_list = []

# for i in range(len(n_vector)):
#     n = [n_vector[i]]
#     m = [m_vector[i]]
#     print(f"n: {n}, m: {m}")

#     train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard")
#     test_acc, test_prec, test_rec, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard") 
#     test_acc_list.append(test_acc)
#     test_prec_list.append(test_prec)
#     test_rec_list.append(test_rec)
#     test_f1_list.append(test_f1)

#     train_MLP(n,m,7, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,0.99, False,False,"MLP_model_25v_50c_fixedHA_lmpc",dataset_type="standard")
#     MLP_test_acc, MLP_test_prec, MLP_test_rec, MLP_test_f1 = test_MLP(n,m,7, seed, data_points,layer_width,number_of_layers,0.99, False,False,"MLP_model_25v_50c_fixedHA_lmpc",dataset_type="standard")
#     MLP_test_acc_list.append(MLP_test_acc)
#     MLP_test_prec_list.append(MLP_test_prec)
#     MLP_test_rec_list.append(MLP_test_rec)
#     MLP_test_f1_list.append(MLP_test_f1)


# plt.figure(figsize=(10,6))
# colors = ['C0','C1','C2','C3']  # reuse same colors for both models

# # GNN metrics (solid)
# plt.plot(n_vector, test_acc_list,  marker='o', linestyle='-', color=colors[0], label='Accuracy (GNN)')
# plt.plot(n_vector, test_prec_list, marker='s', linestyle='-', color=colors[1], label='Precision (GNN)')
# plt.plot(n_vector, test_rec_list,  marker='^', linestyle='-', color=colors[2], label='Recall (GNN)')
# plt.plot(n_vector, test_f1_list,   marker='d', linestyle='-', color=colors[3], label='F1 (GNN)')

# # MLP metrics (dashed)
# plt.plot(n_vector, MLP_test_acc_list,  marker='o', linestyle='--', color=colors[0], label='Accuracy (MLP)')
# plt.plot(n_vector, MLP_test_prec_list, marker='s', linestyle='--', color=colors[1], label='Precision (MLP)')
# plt.plot(n_vector, MLP_test_rec_list,  marker='^', linestyle='--', color=colors[2], label='Recall (MLP)')
# plt.plot(n_vector, MLP_test_f1_list,   marker='d', linestyle='--', color=colors[3], label='F1 (MLP)')


# plt.xlabel("n")
# plt.ylabel("Metric value")
# plt.ylim(0, 1)
# plt.title("Test metrics vs n")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

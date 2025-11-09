import numpy as np
from train_model import train_GNN
from test_model import test_GNN
from utils import barplot_iterations, histogram_time, histogram_prediction_time, barplot_iter_reduction,histogram_iterations
import matplotlib.pyplot as plt
import time
import pickle
# Set parameters
n_train = [20, 40, 60] #[10]
m_train = [40,80,120] #[40]
#m = [80,160,240] #[40]

nth = 7
seed = 123
data_points = 2000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
t = 0.5
scale = 0.01
runs = 1 #5
model_name = f"model_{n_train}v_{m_train}c_multi"
cuda = 0

#GNN
# print("---- GNN ----")
# train_time_start = time.time()
# train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c_multi",dataset_type="standard")
# train_time_end = time.time()
# print(f"Training time (s): {train_time_end - train_time_start}")
# test_time_start = time.time()
# test_acc, test_prec, test_rec, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c_multi",dataset_type="standard") 
# test_time_end = time.time()
# print(f"Testing time (s): {test_time_end - test_time_start}")
# n = [80] #[10]
# m = [160]
# print("--- Testing on larger problem ---")
# prediction_time, test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard") 

# n = [85] #[10]
# m = [170]
# print("--- Testing on larger problem ---")
# prediction_time, test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_100v_250c_fixedHA",dataset_type="standard") 


n_test = [100] #[10]
m_test = [200] #[400]
print("---- GNN ----")
total_start_time = time.time()
test_time_before_vector,test_time_after_vector,test_iterations_before_vector, test_iterations_after_vector, test_iterations_difference_vector = [], [], [],[], []
for i in range(runs):
    # Train
    train_time_start = time.time()
    train_GNN(n_train,m_train,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,model_name,dataset_type="standard",cuda = cuda)



    #train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,model_name,scale_H=scale,dataset_type="lmpc",cuda = cuda)
    train_time_end = time.time()
    print(f"Training time: {train_time_end - train_time_start} seconds")
    # Test
    print("--- Testing on larger problem ---")
    _,test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference = test_GNN(n_test,m_test,nth,seed, data_points,layer_width,number_of_layers,t, False,False,model_name,dataset_type="standard",cuda = cuda)
    test_time_before_vector.append(test_time_before)
    test_time_after_vector.append(test_time_after)
    test_iterations_before_vector.append(test_iterations_before)
    test_iterations_after_vector.append(test_iterations_after)
    test_iterations_difference_vector.append(test_iterations_difference)

test_time_before_vector = np.stack(test_time_before_vector)
test_time_after_vector = np.stack(test_time_after_vector)
test_iterations_before_vector = np.stack(test_iterations_before_vector)
test_iterations_after_vector = np.stack(test_iterations_after_vector)
test_iterations_difference = np.stack(test_iterations_difference_vector)

# Compute average (elementwise mean across runs)
test_time_before_avg = test_time_before_vector.mean(axis=0)
test_time_after_avg = test_time_after_vector.mean(axis=0)
test_iterations_before_avg = test_iterations_before_vector.mean(axis=0)
test_iterations_after_avg = test_iterations_after_vector.mean(axis=0)
test_iterations_diff_avg = test_iterations_difference.mean(axis=0)

# Save data 
with open(f"data/multi_experiment_{n_train}v_{m_train}c_test_{n_test}v_{m_test}c.pkl", "wb") as f:
    pickle.dump((f"layer width: {layer_width}, data points: {data_points}, t: {t}", test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg), f)




# Load data
with open(f"data/multi_experiment_{n_train}v_{m_train}c_test_{n_test}v_{m_test}c.pkl", "rb") as f:
    parameters, test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg = pickle.load(f)


print(f"Test iter before: quantiles {np.percentile(test_iterations_before_avg, [10,25, 50, 75,90])}")
print(f'Test iter after: mean {np.mean(test_iterations_after_avg)}, min {np.min(test_iterations_after_avg)}, max {np.max(test_iterations_after_avg)}')
print(f"Test iter after: quantiles {np.percentile(test_iterations_after_avg, [10,25, 50, 75,90])}")
print(f'Test iter reduction: mean {np.mean(test_iterations_diff_avg)}, min {np.min(test_iterations_diff_avg)}, max {np.max(test_iterations_diff_avg)}')
print(f"Test iter after: quantiles {np.percentile(test_iterations_diff_avg, [5,10,20,30,40, 50, 60,70,80,90,95])}")

# Plot average results
histogram_time(test_time_before_avg, test_time_after_avg, f"{model_name}_test", save=True)
# barplot_iter_reduction(test_iterations_diff_avg, model_name, save=True)
barplot_iterations(test_iterations_before_avg, test_iterations_after_avg, model_name, save=True)
histogram_iterations(test_iterations_before_avg, test_iterations_after_avg, f"{model_name}_test", save=True)
total_end_time = time.time()
# print(f"Training time: {total_end_time-total_start_time} seconds")

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



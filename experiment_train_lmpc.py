
import numpy as np
from generate_mpqp_v2 import generate_rhs 
from generate_graph_data import generate_qp_graphs_train_val_lmpc, generate_qp_graphs_test_data_only_lmpc
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
from utils import barplot_iterations, histogram_time, histogram_prediction_time, barplot_iter_reduction
import matplotlib.pyplot as plt
import time
import pickle

# Set parameters
n = [10,25]
m = [20,50]
nth = 7
seed = 123
data_points = 2000
lr = 0.001
number_of_max_epochs = 100 #100# 20 #100
layer_width = 128
number_of_layers = 3#3
track_on_wandb = True #True
t = 0.9
scale = 0.01
n_number = n[0]

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

# n = [5,10,20] #5,10, 20,50
# m = [206,216,236] # 206,216, 236, 296
n = [50]
m = [296]
t = 0.5 # 0.6
runs = 1

test_time_before_vector,test_time_after_vector,test_iterations_before_vector, test_iterations_after_vector, test_iterations_difference_vector = [], [], [],[], []
for i in range(runs):
    train_time_start = time.time()
    train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n[0]}v_{m[0]}c_lmpc_R_00001",scale_H=scale,dataset_type="lmpc")
    train_time_end = time.time()
    test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n[0]}v_{m[0]}c_lmpc_R_00001",dataset_type="lmpc")

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
# with open("data/lmpc_experiment_50v_296c.pkl", "wb") as f:
#     pickle.dump((f"layer width: {layer_width}, data points: {data_points}, t: {t}", test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg), f)

# Load data
# with open("./data/lmpc_experiment_server.pkl", "rb") as f:
#     parameters, test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg = pickle.load(f)


# print(f"Test iter before: quantiles {np.percentile(test_iterations_before_avg, [10,25, 50, 75,90])}")
# print(f'Test iter after: mean {np.mean(test_iterations_after_avg)}, min {np.min(test_iterations_after_avg)}, max {np.max(test_iterations_after_avg)}')
# print(f"Test iter after: quantiles {np.percentile(test_iterations_after_avg, [10,25, 50, 75,90])}")
# print(f'Test iter reduction: mean {np.mean(test_iterations_diff_avg)}, min {np.min(test_iterations_diff_avg)}, max {np.max(test_iterations_diff_avg)}')
# print(f"Test iter after: quantiles {np.percentile(test_iterations_diff_avg, [5,10,20,30,40, 50, 60,70,80,90,95])}")

# Plot average results
model_name = f"model_{n[0]}v_{m[0]}c_lmpc_R_00001_avg"
# histogram_time(test_time_before_avg, test_time_after_avg, model_name, save=True)
# barplot_iter_reduction(test_iterations_diff_avg, model_name, save=True)
# barplot_iterations(test_iterations_before_avg, test_iterations_after_avg, model_name, save=True)


#print(f"Training time: {train_time_end - train_time_start} seconds")
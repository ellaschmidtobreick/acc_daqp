
import numpy as np
from generate_mpqp_v2 import generate_rhs 
from generate_graph_data import generate_qp_graphs_train_val_lmpc, generate_qp_graphs_test_data_only_lmpc
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
from utils import barplot_iterations, histogram_time, histogram_prediction_time, barplot_iter_reduction,histogram_iterations
import matplotlib.pyplot as plt
import time
import pickle

# Set parameters
n = [5]
m = [206]
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
model_name = f"model_{n[0]}v_{m[0]}c_lmpc_R_00001_avg"
cuda = 0


total_start_time = time.time()
test_time_before_vector,test_time_after_vector,test_iterations_before_vector, test_iterations_after_vector, test_iterations_difference_vector = [], [], [],[], []
for i in range(runs):
    train_time_start = time.time()
    #train_GNN(n,m,nth,seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,model_name,scale_H=scale,dataset_type="lmpc",cuda = cuda)
    train_time_end = time.time()
    print(f"Training time: {train_time_end - train_time_start} seconds")
    _,test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference = test_GNN(n,m,nth,seed, data_points,layer_width,number_of_layers,t, False,False,model_name,dataset_type="lmpc",cuda = cuda)
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
with open("data/lmpc_experiment_5v_206c.pkl", "wb") as f:
    pickle.dump((f"layer width: {layer_width}, data points: {data_points}, t: {t}", test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg), f)

# Load data
with open("./data/lmpc_experiment_5v_206c.pkl", "rb") as f:
    parameters, test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg = pickle.load(f)


print(f"Test iter before: quantiles {np.percentile(test_iterations_before_avg, [10,25, 50, 75,90])}")
print(f'Test iter after: mean {np.mean(test_iterations_after_avg)}, min {np.min(test_iterations_after_avg)}, max {np.max(test_iterations_after_avg)}')
print(f"Test iter after: quantiles {np.percentile(test_iterations_after_avg, [10,25, 50, 75,90])}")
print(f'Test iter reduction: mean {np.mean(test_iterations_diff_avg)}, min {np.min(test_iterations_diff_avg)}, max {np.max(test_iterations_diff_avg)}')
print(f"Test iter after: quantiles {np.percentile(test_iterations_diff_avg, [5,10,20,30,40, 50, 60,70,80,90,95])}")

# Plot average results
histogram_time(test_time_before_avg, test_time_after_avg, model_name, save=True)
# barplot_iter_reduction(test_iterations_diff_avg, model_name, save=True)
# barplot_iterations(test_iterations_before_avg, test_iterations_after_avg, model_name, save=True)
histogram_iterations(test_iterations_before_avg, test_iterations_after_avg, model_name, save=True)
total_end_time = time.time()
# print(f"Training time: {total_end_time-total_start_time} seconds")
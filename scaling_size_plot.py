import torch.nn as nn
import torch_geometric.nn as pyg_nn
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import numpy as np
from utils import plot_scaling,plot_scaling_iterations
import matplotlib.pyplot as plt
import pickle
import daqp
from generate_mpqp_v2 import generate_qp
import time
import torch

# Parameters
n = np.arange(0,501,10)[1:]
m = np.arange(0,501,10)[1:]*4

print(n)
print(m)

nth = 7
seed = 123
data_points = 2000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
t = 0.9 #0.6 # vary
A_flexible = False
H_flexible = False
conv_type = "LEConv"
num_runs = 5

torch.cuda.empty_cache()
start_time = time.time()

# Run experiments
prediction_time_mean, solving_time_mean, label_vector, iterations_after_mean = [], [], [], []
prediction_time_std, solving_time_std, iterations_after_std = [], [], []

for n_i,m_i in zip(n,m):
    n_i= [n_i]
    m_i= [m_i]
    print()

    # GNN model
    print(f"--- GNN, variables {n_i}, constraints {m_i} ---")
    prediction_time_vector, solving_time_vector, iterations_after_vector = [], [], []
    for i in range(num_runs):
        train_GNN(n_i,m_i,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_scaling",dataset_type="standard", conv_type=conv_type)
        prediction_time, test_time_after, iterations_after = test_GNN(n_i,m_i,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_scaling",dataset_type="standard",conv_type=conv_type) 
        prediction_time_vector.append(prediction_time)
        solving_time_vector.append(test_time_after)
        iterations_after_vector.append(iterations_after)
        print()

    # Compute mean and std
    label_vector.append(("GNN",f"{n_i}v{m_i}c"))
    prediction_time_mean.append(np.mean(prediction_time_vector))
    solving_time_mean.append(np.mean(solving_time_vector))
    iterations_after_mean.append(np.mean(iterations_after_vector))

    prediction_time_std.append(np.std(prediction_time_vector))
    solving_time_std.append(np.std(solving_time_vector))
    iterations_after_std.append(np.std(iterations_after_vector))

    # MLP model
    print(f"--- MLP, variables {n_i}, constraints {m_i} ---")
    prediction_time_vector, solving_time_vector, iterations_after_vector = [], [], []
    for i in range(num_runs):
        train_MLP(n_i,m_i,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_scaling",dataset_type="standard")
        prediction_time, test_time_after, iterations_after = test_MLP(n_i,m_i,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_scaling",dataset_type="standard")
        prediction_time_vector.append(prediction_time)
        solving_time_vector.append(test_time_after)
        iterations_after_vector.append(iterations_after)
        print()
    
    # Compute mean and std
    prediction_time_mean.append(np.mean(prediction_time_vector))
    solving_time_mean.append(np.mean(solving_time_vector))
    label_vector.append(("MLP",f"{n_i}v{m_i}c"))
    iterations_after_mean.append(np.mean(iterations_after_vector))

    prediction_time_std.append(np.std(prediction_time_vector))
    solving_time_std.append(np.std(solving_time_vector))
    iterations_after_std.append(np.std(iterations_after_vector))

    # Add non-learned model
    print(f"--- Non-learned model ---")
    # Initialization for data generation
    iter_train = int(np.rint(0.8*data_points))
    iter_val = int(np.rint(0.1*data_points))
    iter_test = int(np.rint(0.1*data_points))

    n_i = n_i[0]
    m_i = m_i[0]

    H,f,F,A,b,B,T = generate_qp(n_i,m_i,seed,nth)
    sense = np.zeros(m_i, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m_i)])

    prediction_time_vector, solving_time_vector, iterations_after_vector = [], [], []
    for i in range(num_runs):
        daqp_time= np.zeros((data_points))
        daqp_iterations= np.zeros((data_points))


        for j in range(data_points):
            if j<iter_train:
                given_seed = seed
            elif j<iter_train+iter_val:
                given_seed = seed + 1
            else:
                given_seed = seed + 2

            theta = np.random.randn(nth)
            
            if A_flexible == True:
                A = np.random.randn(m,n)
                B = A @ (-T)
                
            btot = b + B @ theta
            ftot = f + F @ theta
            
            if H_flexible == True:
                M = np.random.randn(n,n)
                H = M @ M.T 

            _,_,_,info = daqp.solve(H,ftot,A,btot,blower,sense)
            daqp_time[j]= list(info.values())[0]+ list(info.values())[1]
            daqp_iterations[j] = list(info.values())[2]

        print("Average solving time (s):", np.mean(daqp_time))
        print("Average solving time (s):", np.mean(daqp_iterations))

        solving_time_vector.append(np.mean(daqp_time))
        prediction_time_vector.append(np.mean(daqp_time))
        iterations_after_vector.append(np.mean(daqp_iterations))


    # Compute mean and std
    label_vector.append(("Cold-started", "-"))
    prediction_time_mean.append(np.mean(prediction_time_vector))
    solving_time_mean.append(np.mean(solving_time_vector))
    iterations_after_mean.append(np.mean(iterations_after_vector))

    prediction_time_std.append(np.std(prediction_time_vector))
    solving_time_std.append(np.std(solving_time_vector))
    iterations_after_std.append(np.std(iterations_after_vector))


    # Save data 
    points = list(zip(solving_time_mean,prediction_time_mean, solving_time_std,prediction_time_std))
    iterations = list(zip(iterations_after_mean,iterations_after_std))
    with open("./data/scaling_data_std_new.pkl", "wb") as f:
        pickle.dump((points, label_vector,iterations), f)

end_time = time.time()
print("Total time for experiments(s):", end_time - start_time)

# # Load data
# with open("./data/scaling_data_std.pkl", "rb") as f:
#     points_loaded, labels_loaded,iterations_after_loaded = pickle.load(f) # ,iterations_after_loaded

# plot_scaling(points_loaded, labels_loaded,"plots/scaling_plot_std")
# plot_scaling_iterations(iterations_after_loaded, labels_loaded,"plots/scaling_plot_iterations_std")
# print("Done")

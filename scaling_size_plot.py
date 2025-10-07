import torch.nn as nn
import torch_geometric.nn as pyg_nn
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import numpy as np
from utils import plot_scaling
import matplotlib.pyplot as plt
import pickle
import daqp
from generate_mpqp_v2 import generate_qp
import time
# Parameters
n = [20,40,60,80,100,120,140,160,180,200] # 2h 20 min 
m = [80,160,240,320,400,480,560,640,720,800]
# n= [2,4,8,16]
# m = [8,16,32,64]
nth = 7
seed = 123
data_points = 2000 #5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128 # vary
number_of_layers = 3     # vary
track_on_wandb = False #True
t = 0.9 #0.6 # vary
A_flexible = False
H_flexible = False
conv_type = "LEConv"

start_time = time.time()
# Run experiments
prediction_time_vector , solving_time_vector, label_vector = [], [], []
for n_i,m_i in zip(n,m):
    n_i= [n_i]
    m_i= [m_i]
    print(f"--- GNN, variables {n_i}, constraints {m_i} ---")
    train_GNN(n_i,m_i,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_scaling",dataset_type="standard", conv_type=conv_type)
    prediction_time, test_time_after = test_GNN(n_i,m_i,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_scaling",dataset_type="standard",conv_type=conv_type) 
    prediction_time_vector.append(prediction_time)
    solving_time_vector.append(test_time_after)
    label_vector.append(("GNN",f"{n_i}v{m_i}c"))
    print()

    print(f"--- MLP, variables {n_i}, constraints {m_i} ---")
    train_MLP(n_i,m_i,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_scaling",dataset_type="standard")
    prediction_time, test_time_after = test_MLP(n_i,m_i,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_scaling",dataset_type="standard")
    prediction_time_vector.append(prediction_time)
    solving_time_vector.append(test_time_after)
    label_vector.append(("MLP",f"{n_i}v{m_i}c"))
    print()

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
    daqp_time= np.zeros((data_points))


    for i in range(data_points):
        if i<iter_train:
            given_seed = seed
        elif i<iter_train+iter_val:
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
        daqp_time[i]= list(info.values())[0]

    print("Average solving time (s):", np.mean(daqp_time))
    solving_time_vector.append(np.mean(daqp_time))
    prediction_time_vector.append(np.mean(daqp_time))
    label_vector.append(("Non-learned", "-"))

end_time = time.time()
print("Total time for experiments (s):", end_time - start_time)

# Save data 
points = list(zip(solving_time_vector,prediction_time_vector))
labels = label_vector
with open("./data/scaling_data.pkl", "wb") as f:
    pickle.dump((points, label_vector), f)


# Load data
with open("./data/scaling_data.pkl", "rb") as f:
    points_loaded, label_vector_loaded = pickle.load(f)

plot_scaling(points_loaded, label_vector_loaded,"plots/scaling_plot1.pdf")

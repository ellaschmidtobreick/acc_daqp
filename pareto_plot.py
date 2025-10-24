from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import numpy as np
from utils import plot_pareto
import pickle
import daqp
from generate_mpqp_v2 import generate_qp

# Parameters
n = [200] #[250]
m = [800] #[1000]
nth = 7
seed = 123
data_points = 20#00 #5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128 # vary
number_of_layers = 3     # vary
track_on_wandb = False #True
t = 0.9 #0.6 # vary
A_flexible = False
H_flexible = False

conv_types = ["GAT", "LEConv"]
layer_width = [64, 128] #,256]
number_of_layers = [3, 4,5]


# Run experiments
prediction_time_vector , solving_time_vector, label_vector = [], [], []
for j in layer_width:
    for k in number_of_layers:
        for i in conv_types:
            print(f"--- GNN, Conv: {i}, Layer width: {j}, Number of layers: {k} ---")
            train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,j,k, track_on_wandb,t, False,False,"model_pareto",dataset_type="standard", conv_type=i)
            prediction_time, test_time_after,_ = test_GNN(n,m,nth, seed, data_points,j,k,t, False,False,"model_pareto",dataset_type="standard",conv_type=i) 
            prediction_time_vector.append(prediction_time)
            solving_time_vector.append(test_time_after)
            label_vector.append((f"GNN - {i}", f"{k} layers"))
            print()

        print(f"--- MLP, Layer width: {j}, Number of layers: {k} ---")
        train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,j,k, track_on_wandb,t, False,False,"MLP_model_pareto",dataset_type="standard")
        prediction_time, test_time_after,_ = test_MLP(n,m,nth, seed, data_points,j,k,t, False,False,"MLP_model_pareto",dataset_type="standard")
        prediction_time_vector.append(prediction_time)
        solving_time_vector.append(test_time_after)
        label_vector.append(("MLP",f"{k} layers"))
        print()

# Add non-learned model
print(f"--- Non-learned model ---")
# Initialization for data generation
iter_train = int(np.rint(0.8*data_points))
iter_val = int(np.rint(0.1*data_points))
iter_test = int(np.rint(0.1*data_points))

for i in range(len(n)):
    n_i = n[i]
    m_i = m[i]
    print
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

# Save data 
points = list(zip(solving_time_vector,prediction_time_vector))
labels = label_vector
with open("./data/pareto_data_200_800.pkl", "wb") as f:
    pickle.dump((points, label_vector), f)


# Load data
with open("./data/pareto_data_200_800.pkl", "rb") as f:
    points_loaded, label_vector_loaded = pickle.load(f)

plot_pareto(points_loaded, label_vector_loaded,"plots/pareto_plot_200_800.pdf")

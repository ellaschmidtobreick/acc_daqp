from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import pickle

n = [10]
m = [40]
nth = 7
seed = 123
data_points = 1000 #0 
lr = 0.001
number_of_max_epochs = 100
track_on_wandb = False #True

model_configurations_gnn = [("LEConv", 128, 3), ("GAT", 128, 3), ("GCN", 128, 3), ("LEConv", 64, 3), ("GAT", 64, 3), ("GCN", 64, 3), ("LEConv", 128, 4), ("GAT", 128, 4), ("GCN", 128, 4),("LEConv", 64, 4), ("GAT", 64, 4), ("GCN", 64, 4)]
# , ("MLP", 128, 3), ("MLP", 128, 4),("MLP", 64, 3),, ("MLP", 64, 4)
model_confs_with_t = []
# threshold tuning
for (i,j,k) in model_configurations_gnn:
    print(f"--- Conv: {i}, Layer width: {j}, Number of layers: {k} ---")
    val_acc_highest = 0
    best_t = 0
    for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print("threshold:", t)
        val_acc = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,j,k, track_on_wandb,t, False,False,"model_10v_40c_pareto",scale_H=1,dataset_type="standard", conv_type=i)
        print("New best t found:", "t",t, "val_acc",val_acc)
        val_acc_highest = val_acc
        best_t = t
    prediction_time, test_time_after = test_GNN(n,m,nth, seed, data_points,j,k,best_t, False,False,"model_10v_40c_pareto",dataset_type="standard",conv_type=i) 
    model_confs_with_t.append((i,j,k,best_t,prediction_time,test_time_after))         

print(model_confs_with_t)
# model_configurations_mlp= [("MLP", 128, 3), ("MLP", 128, 4),("MLP", 64, 3), ("MLP", 64, 4)]
# model_confs_with_t = []

# # threshold tuning
# for (i,j,k) in model_configurations_mlp:
#     print(f"--- Conv: {i}, Layer width: {j}, Number of layers: {k} ---")
#     val_acc_highest = 0
#     best_t = 0
#     for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#         print("threshold:", t)
#         val_acc = train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,j,k, track_on_wandb,t, False,False,"model_10v_40c_pareto",scale_H=1,dataset_type="standard", conv_type=i)
#         print("New best t found:", "t",t, "val_acc",val_acc)
#         val_acc_highest = val_acc
#         best_t = t
#     prediction_time, test_time_after = test_MLP(n,m,nth, seed, data_points,j,k,best_t, False,False,"model_10v_40c_pareto",dataset_type="standard",conv_type=i) 
#     model_confs_with_t.append((i,j,k,best_t,prediction_time,test_time_after))         



# Save
with open("model_results.pkl", "wb") as f:
    pickle.dump(model_confs_with_t, f)

# Load later
# with open("model_results.pkl", "rb") as f:
#     loaded_results = pickle.load(f)

from train_model import train_GNN
from test_model import test_GNN
import matplotlib.pyplot as plt
import numpy as np

# Set parameters
n = [10]
m = [40]
nth = 2
seed_vector = [123,124,125,126,127]
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
runs = 5
layer_width = 128
data_points = 5000
t = 0.8 # threshold 
# to get the final results on an average of 5 runs

train_acc_vector , train_prec_vector, train_recall_vector, train_f1_vector = [], [], [], []
test_acc_vector , test_prec_vector, test_recall_vector, test_f1_vector= [], [], [], []

for i in range(runs):
    seed = seed_vector[i]
    train_acc,train_prec, train_recall, train_f1 = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c",dataset_type="standard")
    test_acc,test_prec, test_recall, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c",dataset_type="standard") 

    train_acc_vector.append(train_acc)
    train_prec_vector.append(train_prec)
    train_recall_vector.append(train_recall)
    train_f1_vector.append(train_f1)
    test_acc_vector.append(test_acc)
    test_prec_vector.append(test_prec)
    test_recall_vector.append(test_recall)
    test_f1_vector.append(test_f1)

# Store results
results = {
    "Parameters": (f"layer width: {layer_width}, data points: {data_points}, t: {t}, dense", None),
    "Train Accuracy": (np.mean(train_acc_vector), np.std(train_acc_vector)),
    "Train Precision": (np.mean(train_prec_vector), np.std(train_prec_vector)),
    "Train Recall": (np.mean(train_recall_vector), np.std(train_recall_vector)),
    "Train F1 Score": (np.mean(train_f1_vector), np.std(train_f1_vector)),
    "Test Accuracy" : (np.mean(test_acc_vector), np.std(test_acc_vector)),
    "Test Precision": (np.mean(test_prec_vector), np.std(test_prec_vector)),
    "Test Recall": (np.mean(test_recall_vector), np.std(test_recall_vector)),
    "Test F1 Score": (np.mean(test_f1_vector), np.std(test_f1_vector)),
}

with open("data/acc_results_threshold_tuning_dense.txt", "a") as f: 
    for key, (mean, std) in results.items():
        if std is None:
            f.write(f"{key}: {mean}\n")
        else:
            f.write(f"{key}: mean = {mean:.6f}, std = {std:.6f}\n")
    f.write("\n" + "-"*60 + "\n\n")
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import matplotlib.pyplot as plt
import numpy as np

# Set parameters
n = [10]
m = [40]
nth = 2
seed = 123
data_points = 2000 #5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 64 #128
number_of_layers = 3
track_on_wandb = False #True
t = 0.9 # 0.6

# Train 4 different models to compare the metrics depending on H being fixed or flexible
# Test the 4 different models on the according test data
# train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard")
# test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard") 

# train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,True,"model_10v_40c_fixedH",dataset_type="standard")
# test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,True,"model_10v_40c_fixedH",dataset_type="standard")

# train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,False,"model_10v_40c_fixedA",dataset_type="standard")
# test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, True,False,"model_10v_40c_fixedA",dataset_type="standard")

# train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,True,"model_10v_40c_flex",dataset_type="standard")
# test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, True,True,"model_10v_40c_flex",dataset_type="standard")

#text_time_before_vector , text_time_after_vector, test_time_reduction_vector, prediction_time_vector = [], [], [], []
train_acc_vector , train_prec_vector, train_recall_vector, train_f1_vector = [], [], [], []
test_acc_vector , test_prec_vector, test_recall_vector, test_f1_vector= [], [], [], []
for i in range(5):
    train_acc,train_prec, train_recall, train_f1 = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard")
    test_acc,test_prec, test_recall, test_f1 = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard") 

    # test MLP on given lmpc data 
    # train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"MLP_model_10v_40c_fixedH",dataset_type="standard")
    # text_time_before, text_time_after, test_time_reduction, prediction_time = test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"MLP_model_10v_40c_fixedH",dataset_type="standard")

    train_acc_vector.append(train_acc)
    train_prec_vector.append(train_prec)
    train_recall_vector.append(train_recall)
    train_f1_vector.append(train_f1)
    test_acc_vector.append(test_acc)
    test_prec_vector.append(test_prec)
    test_recall_vector.append(test_recall)
    test_f1_vector.append(test_f1)

    # text_time_before_vector.append(text_time_before)
    # text_time_after_vector.append(text_time_after)
    # test_time_reduction_vector.append(test_time_reduction)
    # prediction_time_vector.append(prediction_time)
    # recall_scores.append(recall)
    # precision_scores.append(precision)

# Store results
results = {
    "Train Accuracy": (np.mean(train_acc_vector), np.std(train_acc_vector)),
    "Train Precision": (np.mean(train_prec_vector), np.std(train_prec_vector)),
    "Train Recall": (np.mean(train_recall_vector), np.std(train_recall_vector)),
    "Train F1 Score": (np.mean(train_f1_vector), np.std(train_f1_vector)),
    "Test Accuracy" : (np.mean(test_acc_vector), np.std(test_acc_vector)),
    "Test Precision": (np.mean(test_prec_vector), np.std(test_prec_vector)),
    "Test Recall": (np.mean(test_recall_vector), np.std(test_recall_vector)),
    "Test F1 Score": (np.mean(test_f1_vector), np.std(test_f1_vector)),
}

with open("data/timing_results.txt", "w") as f:
    for key, (mean, std) in results.items():
        f.write(f"{key}: mean = {mean:.6f}, std = {std:.6f}\n")


# print(f"Average test time before: {np.mean(text_time_before_vector), np.std(text_time_before_vector)}")
# print(f"Average test time after: {np.mean(text_time_after_vector), np.std(text_time_after_vector)}")   
# print(f"Average test time reduction: {np.mean(test_time_reduction_vector), np.std(test_time_reduction_vector)}")
# print(f"Average prediction time: {np.mean(prediction_time_vector), np.std(prediction_time_vector)}")


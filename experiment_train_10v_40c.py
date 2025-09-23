from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import matplotlib.pyplot as plt
import numpy as np

# Set parameters
n = [10]
m = [40]
nth = 2
seed = 123
data_points = 5000 #5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = False #True
t = 0.6

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

text_time_before_vector , text_time_after_vector, test_time_reduction_vector, prediction_time_vector = [], [], [], []
for i in range(5):
    # train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard")
    # text_time_before, text_time_after, test_time_reduction, prediction_time = test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c_fixedHA",dataset_type="standard") 

    # test MLP on given lmpc data 
    train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"MLP_model_10v_40c_fixedH",dataset_type="standard")
    text_time_before, text_time_after, test_time_reduction, prediction_time = test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"MLP_model_10v_40c_fixedH",dataset_type="standard")



    text_time_before_vector.append(text_time_before)
    text_time_after_vector.append(text_time_after)
    test_time_reduction_vector.append(test_time_reduction)
    prediction_time_vector.append(prediction_time)
    # recall_scores.append(recall)
    # precision_scores.append(precision)

print(f"Average test time before: {np.mean(text_time_before_vector), np.std(text_time_before_vector)}")
print(f"Average test time after: {np.mean(text_time_after_vector), np.std(text_time_after_vector)}")   
print(f"Average test time reduction: {np.mean(test_time_reduction_vector), np.std(test_time_reduction_vector)}")
print(f"Average prediction time: {np.mean(prediction_time_vector), np.std(prediction_time_vector)}")


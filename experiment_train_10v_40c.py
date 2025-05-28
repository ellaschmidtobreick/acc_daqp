from train_model import train_GNN
from test_model import test_GNN
import matplotlib.pyplot as plt

# Set parameters
n = [10]
m = [40]
nth = 2
seed = 123
data_points = 5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = True
t = 0.6

# Train 4 different models to compare the metrics depending on H being fixed or flexible
# Test the 4 different models on the according test data
train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_10v_40c_fixedHA")
test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_10v_40c_fixedHA") 

train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,True,"model_10v_40c_fixedH")
test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,True,"model_10v_40c_fixedH")

train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,False,"model_10v_40c_fixedA")
test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, True,False,"model_10v_40c_fixedA")

train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,True,"model_10v_40c_flex")
test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, True,True,"model_10v_40c_flex")


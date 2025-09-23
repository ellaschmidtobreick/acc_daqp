from train_model import train_GNN
from test_model import test_GNN

# Set parameters
n = [25]
m = [50]
nth = 7
seed = 123
data_points = 1000# 5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = True
t = 0.6

# Train the model on scaled data
train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_50v_100c_fixedHA",dataset_type="standard")
test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_50v_100c_fixedHA",dataset_type="standard") 

# make the 100 constraints 2x50 (from left and right)

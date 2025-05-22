from train_model import train_GNN
from test_model import test_GNN
from utils import map_train_acc

# Set parameters
n = [10,25]
m = [40,100]
nth = 2
seed = 123
data_points = 2500
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = True
t = 0.6

# Train model on dataset with different graph sizes
#train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,f"model_{n}v_{m}c_fixedHA")

#test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,f"model_{n}v_{m}c_fixedHA") 
test_GNN([10],[40],nth, seed, data_points*2,layer_width,number_of_layers,t, False,False,f"model_{n}v_{m}c_fixedHA") 
#test_GNN([25],[100],nth, seed, data_points*2,layer_width,number_of_layers,t, False,False,f"model_{n}v_{m}c_fixedHA") 

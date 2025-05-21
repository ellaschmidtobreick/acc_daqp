from train_model import train_GNN
from test_model import test_GNN
from utils import map_train_acc

# Set parameters
n = [25]
m = [100]
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
train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"model_25v_100c_fixedHA")
test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"model_25v_100c_fixedHA") 


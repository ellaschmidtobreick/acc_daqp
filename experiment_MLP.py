from train_model import train_MLP
from test_model import test_MLP

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
track_on_wandb = False
t = 0.6

# train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"MLP_model_10v_40c_fixedH",dataset_type="standard")
# test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,False,"MLP_model_10v_40c_fixedH",dataset_type="standard")

# train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,True,"MLP_model_10v_40c_fixedH")
# test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t, False,True,"MLP_model_10v_40c_fixedH")

# train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,False,"MLP_model_10v_40c_fixedA")
# test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t, True,False,"MLP_model_10v_40c_fixedA")

# train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,True,"MLP_model_10v_40c_flex")
# test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t, True,True,"MLP_model_10v_40c_flex")


# train_MLP([25],[100],nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False,"MLP_model_25v_100c_fixedHA")
# test_MLP([25],[100],nth, seed, data_points,layer_width,number_of_layers,t, False,False,"MLP_model_25v_100c_fixedHA")


# test MLP on given lmpc data 
train_MLP([25],[50],7, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,0.99, False,False,"MLP_model_25v_50c_fixedHA_lmpc",dataset_type="lmpc")
test_MLP([25],[50],7, seed, data_points,layer_width,number_of_layers,0.99, False,False,"MLP_model_25v_50c_fixedHA_lmpc",dataset_type="lmpc")

from train_model import train_GNN
import numpy as np

# Set parameters
n = 10 #25
m = 40 #100

# n = [10,11,12,13,14,15] #[2,3]#[10,11] #config.n
# m = [40,44,48,52,56,60] #[5,7]#[40,44] #config.m

nth = 2
seed = 123
data_points = 5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128
number_of_layers = 3
track_on_wandb = True
t = 0.4

threshold = np.arange(0.1,1,0.1)
#Threshold tuning
best_threshold = 0
best_mean = np.inf
for t in threshold:
    epoch, train_acc_save, val_acc_save, train_loss_save, val_loss_save, train_prec_save, val_prec_save, train_rec_save, val_rec_save, train_f1_save, val_f1_save, acc_graph_train_save, acc_graph_val_save, val_perc_wrongly_pred_nodes_per_graph_save, val_mean_wrongly_pred_nodes_per_graph_save = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False)

    #Threshold tuning
    if val_mean_wrongly_pred_nodes_per_graph_save[epoch-6] < best_mean:
        best_threshold = t
        best_mean = val_mean_wrongly_pred_nodes_per_graph_save[epoch-6]

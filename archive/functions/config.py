from train_model import train_GNN
import numpy as np

# Set parameters
n = [10] #25
m = [40] #100

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
t = 0.6

epoch_fixedHA, train_acc_fixedHA, val_acc_fixedHA, train_loss_fixedHA, val_loss_fixedHA, train_prec_fixedHA, val_prec_fixedHA, train_rec_fixedHA, val_rec_fixedHA, train_f1_fixedHA, val_f1_fixedHA, acc_graph_train_fixedHA, acc_graph_val_fixedHA, val_perc_wrongly_pred_nodes_per_graph_fixedHA, val_mean_wrongly_pred_nodes_per_graph_fixedHA = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,False)
epoch_fixedH, train_acc_fixedH, val_acc_fixedH, train_loss_fixedH, val_loss_fixedH, train_prec_fixedH, val_prec_fixedH, train_rec_fixedH, val_rec_fixedH, train_f1_fixedH, val_f1_fixedH, acc_graph_train_fixedH, acc_graph_val_fixedH, val_perc_wrongly_pred_nodes_per_graph_fixedH, val_mean_wrongly_pred_nodes_per_graph_fixedH = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, False,True)
epoch_fixedA, train_acc_fixedA, val_acc_fixedA, train_loss_fixedA, val_loss_fixedA, train_prec_fixedA, val_prec_fixedA, train_rec_fixedA, val_rec_fixedA, train_f1_fixedA, val_f1_fixedA, acc_graph_train_fixedA, acc_graph_val_fixedA, val_perc_wrongly_pred_nodes_per_graph_fixedA, val_mean_wrongly_pred_nodes_per_graph_fixedA = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,False)
epoch_flex, train_acc_flex, val_acc_flex, train_loss_flex, val_loss_flex, train_prec_flex, val_prec_flex, train_rec_flex, val_rec_flex, train_f1_flex, val_f1_flex, acc_graph_train_flex, acc_graph_val_flex, val_perc_wrongly_pred_nodes_per_graph_flex, val_mean_wrongly_pred_nodes_per_graph_flex = train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, True,True)



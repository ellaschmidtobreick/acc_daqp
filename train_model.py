import numpy as np
from ctypes import * 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.utils.class_weight import compute_class_weight
import wandb

import torch
from torch_geometric.loader import DataLoader

from generate_graph_data import generate_qp_graphs_train_val, generate_qp_graphs_train_val_flexible_H
import config 
from model import GNN
from model import EarlyStopping


# Set parameters
n = 10 #config.n
m = 40 #config.m

nth = config.nth
seed = config.seed
data_points = config.data_points 
lr = config.lr
number_of_epochs = config.number_of_epochs  
layer_width = config.layer_width
number_of_layers = 5 #config.number_of_layers
track_on_wandb = config.track_on_wandb
t = config.t # tuned by gridsearch threshold = np.arange(0.1,1,0.1)

# Threshold tuning
# best_threshold = 0
# best_mean = np.inf
#for t in threshold:

# Generate QP problems and the corresponding graphs
graph_train, graph_val,H,A = generate_qp_graphs_train_val_flexible_H(n,m,nth,seed,data_points)#generate_qp_graphs_train_val(n,m,nth,seed,data_points)

#graph_train,graph_val = generate_qp_graphs_different_sizes(n,n,m,m,nth,seed,data_points,"train",H=H,A =A)
# graph_val,n_val, m_val = generate_qp_graphs_different_sizes(n,n,m,m,nth,seed,data_points,"val",H=H,A =A)

# Load Data
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
val_loader =DataLoader(graph_val,batch_size = len(graph_val), shuffle = False)

# Compute class weights for imbalanced classes
all_labels = torch.cat([data.y for data in graph_train])
class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Instantiate model and optimizer
model = GNN(input_dim=4, output_dim=1,layer_width = 128)  # Output dimension 1 for binary classification
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

# Early stopping
early_stopping = EarlyStopping(patience=5, delta=0.001)

#epoch = 0
#acc = 0

if track_on_wandb ==True:
    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="ella-schmidtobreick-4283-me",
        project="Thesis",
        # Track hyperparameters and run metadata.
        config={
            "variables": f"{n}",
            "constraints": f"{m}",
            "datapoints": f"{data_points}",
            "epochs": f"{number_of_epochs}",
            "architecture": "LEConv with weights",
            "learning_rate": f"{lr}",
            "layer width": f"{layer_width}",
            "number of layers": f"{number_of_layers}",
            "threshold": f"{t}"
        },
    )


for epoch in range(number_of_epochs):
    #print(f"Epoch {epoch}")
#while acc != 1:
    epoch += 1
    train_loss = 0
    train_all_labels = []
    train_preds = []
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch,number_of_layers)
        loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()])(output.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        
        # Compute loss
        train_loss += loss.item()

        # Convert output to binary prediction (0 or 1)
        #save_loss += output.tolist()
        preds = (output.squeeze() > t).long()

        #save_preds += preds.tolist()
        train_preds.extend(preds.numpy())   # Store predictions
        train_all_labels.extend(batch.y.numpy())

    # Compute the loss
    train_loss /= len(train_loader)

    # Compute metrics
    train_acc = accuracy_score(train_all_labels, train_preds)
    train_prec = precision_score(train_all_labels,train_preds)
    train_rec = recall_score(train_all_labels, train_preds)
    train_f1 = f1_score(train_all_labels,train_preds)
    
    # over graph metrices
    all_label_graph = np.array(train_all_labels).reshape(-1,n+m)
    train_preds_graph = np.array(train_preds).reshape(-1,n+m)

    # Compute average over graphs
    acc_graph_train = np.mean(np.all(all_label_graph == train_preds_graph, axis=1))
    
    # Validation step
    model.eval()
    val_loss = 0
    val_mean_wrongly_pred_nodes_per_graph = 0
    val_num_wrongly_pred_nodes_per_graph = 0
    val_all_labels = []
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch,number_of_layers)
            loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())
            val_loss += loss.item()
            preds = (output.squeeze() > t).long()
            #correct += (preds == batch.y).sum().item()
            # total += batch.y.size(0)
            val_preds.extend(preds.numpy())   # Store predictions
            val_all_labels.extend(batch.y.numpy()) # Store true labels

    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_all_labels, val_preds)
    val_prec = precision_score(val_all_labels,val_preds)
    val_rec = recall_score(val_all_labels, val_preds)
    val_f1 = f1_score(val_all_labels,val_preds)
    
    # over graph metrices
    val_all_label_graph = np.array(val_all_labels).reshape(-1,n+m)
    val_preds_graph = np.array(val_preds).reshape(-1,n+m)

    # Compute average over graphs
    acc_graph_val = np.mean(np.all(val_all_label_graph == val_preds_graph, axis=1))
    val_mean_wrongly_pred_nodes_per_graph = np.mean((n+m) - np.sum(val_all_label_graph == val_preds_graph, axis=1))
    val_num_wrongly_pred_nodes_per_graph = (n+m) - np.sum(val_all_label_graph == val_preds_graph, axis=1)
    val_perc_wrongly_pred_nodes_per_graph = val_mean_wrongly_pred_nodes_per_graph/(n+m)
    
    # Log metrics to wandb.
    if track_on_wandb == True:
        run.log({"acc_train": train_acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec_train": train_prec, "prec_val":val_prec, "rec_train": train_rec, "rec_val": val_rec, "f1_train": train_f1,"f1_val":val_f1, "acc_graph_train": acc_graph_train, "acc_graph_val": acc_graph_val,"perc_wrong_pred_nodes_per_graph_val": val_perc_wrongly_pred_nodes_per_graph,"num_wrong_pred_nodes_per_graph_val":val_mean_wrongly_pred_nodes_per_graph, "threshold": t})

    early_stopping(val_loss, model) #val_mean_wrongly_pred_nodes_per_graph
    if early_stopping.early_stop:
        print(f"Early stopping after {epoch} epochs.")
        break
    
    # Threshold tuning
    # if val_mean_wrongly_pred_nodes_per_graph < best_mean:
    #     best_threshold = t
    #     best_mean = val_mean_wrongly_pred_nodes_per_graph

# Load the best model
early_stopping.load_best_model(model)

torch.save(model.state_dict(), f"saved_models/model_{n}v_{m}c_flex_H_5 layers.pth")

# Finish the run and upload any remaining data.
if track_on_wandb == True:
    run.finish()
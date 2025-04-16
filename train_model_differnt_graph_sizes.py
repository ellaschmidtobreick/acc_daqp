import numpy as np
from ctypes import * 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.utils.class_weight import compute_class_weight
import wandb

import torch
from torch_geometric.loader import DataLoader

from generate_graph_data import generate_qp_graphs_train_val, generate_qp_graphs_different_sizes
import config 
from model import GNN
from model import EarlyStopping


# Set parameters
n = config.n
m = config.m

n_min = 10
n_max = 10
m_min = 40
m_max = 40

nth = config.nth
seed = config.seed
data_points = config.data_points #5000
lr = config.lr
number_of_epochs = config.number_of_epochs #70 #300 # 500 
layer_width = config.layer_width
number_of_layers = config.number_of_layers
track_on_wandb = config.track_on_wandb
#threshold = np.arange(0.1,1,0.1)
t = config.t # tuned by gridsearch threshold = np.arange(0.1,1,0.1)

# Threshold tuning
# best_threshold = 0
# best_mean = np.inf
#for t in threshold:

# Generate QP problems and the corresponding graphs
# graph_train, graph_val,H,A = generate_qp_graphs_train_val(n,m,nth,seed,data_points)
graph_train, n_train, m_train = generate_qp_graphs_different_sizes(n_min,n_max,m_min,m_max,nth,seed,data_points, "train")
graph_val,n_val, m_val = generate_qp_graphs_different_sizes(n_min,n_max,m_min,m_max,nth,seed,data_points, "val")

# print(H)
# print(A)
# Load Data
train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
val_loader =DataLoader(graph_val,batch_size = len(graph_val), shuffle = False)

# Compute class weights for imbalanced classes
all_labels = torch.cat([data.y for data in graph_train])
class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Instantiate model and optimizer
model = GNN(input_dim=3, output_dim=1,layer_width = 128)  # Output dimension 1 for binary classification
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

# Early stopping
early_stopping = EarlyStopping(patience=5, delta=0.001)

epoch = 0
acc = 0

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
            "architecture": "dataset with different graph sizes",
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
    #correct = 0
    #num_batches = 0
    #save_preds = []
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
        #num_batches += 1

        # Convert output to binary prediction (0 or 1)
        #save_loss += output.tolist()
        preds = (output.squeeze() > t).long()
        #save_preds += preds.tolist()
        train_preds.extend(preds.numpy())   # Store predictions
        train_all_labels.extend(batch.y.numpy())

    # Compute the loss
    #avg_train_loss /= train_loss / num_batches
    train_loss /= len(train_loader)

    # Compute metrics
    acc = accuracy_score(train_all_labels, train_preds)
    prec = precision_score(train_all_labels,train_preds)
    rec = recall_score(train_all_labels, train_preds)
    f1 = f1_score(train_all_labels,train_preds)
    
    graph_train_correct = []
    # over graph metrices
    for i in range(len(n_train)):
        n = n_train[i]
        m = m_train[i]
        all_label_graph_i = np.array(train_all_labels)[:n+m]
        train_preds_graph = np.array(train_preds)[:n+m]
        graph_train_correct.append(np.all(all_label_graph_i == train_preds_graph))
        train_all_labels = train_all_labels[n+m :]
        train_preds = train_preds[n+m:]

    # Compute average over graphs
    acc_graph = np.mean(graph_train_correct)
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
    #val_acc = correct / total
    val_acc = accuracy_score(val_all_labels, val_preds)

    # over graph metrices
    # val_all_label_graph = np.array(val_all_labels).reshape(-1,n+m)
    # val_preds_graph = np.array(val_preds).reshape(-1,n+m)

    graph_val_correct = []
    val_wrongly_pred_nodes_graph_i_percentage = []
    val_wrongly_pred_nodes_graph_i_save = []
    # over graph metrices
    for i in range(len(n_val)):
        n = n_val[i]
        m = m_val[i]
        val_all_label_graph = np.array(val_all_labels)[:n+m]
        val_preds_graph = np.array(val_preds)[:n+m]
        graph_val_correct.append(np.all(val_all_label_graph == val_preds_graph))

        val_wrongly_pred_nodes_graph_i = (n+m) - np.sum(val_all_label_graph == val_preds_graph)
        val_wrongly_pred_nodes_graph_i_save.append(val_wrongly_pred_nodes_graph_i)
        #print(n+m, val_wrongly_pred_nodes_graph_i)
        # convert number of wrongly predicted graphs in a percetage to be comparable for different graph sizes
        val_wrongly_pred_nodes_graph_i_percentage.append(val_wrongly_pred_nodes_graph_i/(n+m))
        
        # remove the values that belong to graph i for the next iteration
        val_all_label_graph = val_all_label_graph[n+m:]
        val_preds_graph = val_preds_graph[n+m:]

    
    # Compute average over graphs
    acc_graph_val = np.mean(graph_val_correct)
    #print(val_wrongly_pred_nodes_graph_i_percentage)
    val_mean_wrongly_pred_nodes_per_graph = np.mean(val_wrongly_pred_nodes_graph_i_percentage)
    #print(val_mean_wrongly_pred_nodes_per_graph)
    #val_num_wrongly_pred_nodes_per_graph = (n+m) - np.sum(val_all_label_graph == val_preds_graph, axis=1)
    #print(val_num_wrongly_pred_nodes_per_graph)
    #print(f"Mean of wrongly predicted nodes per graph: {val_mean_wrongly_pred_nodes_per_graph}")
    
    # Log metrics to wandb.
    if track_on_wandb == True:
        run.log({"acc_train": acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec": prec, "rec": rec, "f1": f1, "acc_graph": acc_graph, "acc_graph_val": acc_graph_val,"acc_wrong_pred_nodes_per_graph":val_mean_wrongly_pred_nodes_per_graph,"num_wrong_pred_nodes_per_graph":np.mean(val_wrongly_pred_nodes_graph_i_save), "threshold": t})

#     early_stopping(val_mean_wrongly_pred_nodes_per_graph, model)
#     if early_stopping.early_stop:
#         print(f"Early stopping after {epoch} epochs.")
#         break
    
#     # Threshold tuning
#     # if val_mean_wrongly_pred_nodes_per_graph < best_mean:
#     #     best_threshold = t
#     #     best_mean = val_mean_wrongly_pred_nodes_per_graph

# # Load the best model
# early_stopping.load_best_model(model)

# torch.save(model.state_dict(), f"saved_models/model_{n_min}_{n_max}v_{m_min}_{m_max}c.pth")

# Finish the run and upload any remaining data.
if track_on_wandb == True:
    run.finish()
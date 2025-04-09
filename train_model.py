import numpy as np
from ctypes import * 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.utils.class_weight import compute_class_weight
import wandb

import torch
from torch_geometric.loader import DataLoader

from generate_graph_data import generate_qp_graphs_train_val 
import config 
from model import GNN
from model import EarlyStopping


# Set parameters
n = config.n
m = config.m

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
graph_train, graph_val,H,A = generate_qp_graphs_train_val(n,m,nth,seed,data_points)
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
    
    # over graph metrices
    all_label_graph = np.array(train_all_labels).reshape(-1,n+m)
    train_preds_graph = np.array(train_preds).reshape(-1,n+m)

    # Compute average over graphs
    acc_graph = np.mean(np.all(all_label_graph == train_preds_graph, axis=1))
    
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
    val_all_label_graph = np.array(val_all_labels).reshape(-1,n+m)
    val_preds_graph = np.array(val_preds).reshape(-1,n+m)

    # Compute average over graphs
    acc_graph_val = np.mean(np.all(val_all_label_graph == val_preds_graph, axis=1))
    val_mean_wrongly_pred_nodes_per_graph = np.mean((n+m) - np.sum(val_all_label_graph == val_preds_graph, axis=1))
    val_num_wrongly_pred_nodes_per_graph = (n+m) - np.sum(val_all_label_graph == val_preds_graph, axis=1)
    #print(val_num_wrongly_pred_nodes_per_graph)
    #print(f"Mean of wrongly predicted nodes per graph: {val_mean_wrongly_pred_nodes_per_graph}")
    
    # Log metrics to wandb.
    if track_on_wandb == True:
        run.log({"acc_train": acc,"acc_test": val_acc,"loss_train": train_loss, "loss_test": val_loss, "prec": prec, "rec": rec, "f1": f1, "acc_graph": acc_graph, "acc_graph_test": acc_graph_val,"num_wrong_pred_nodes_per_graph":val_mean_wrongly_pred_nodes_per_graph, "threshold": t})

    early_stopping(val_mean_wrongly_pred_nodes_per_graph, model)
    if early_stopping.early_stop:
        print(f"Early stopping after {epoch} epochs.")
        break
    
    # Threshold tuning
    # if val_mean_wrongly_pred_nodes_per_graph < best_mean:
    #     best_threshold = t
    #     best_mean = val_mean_wrongly_pred_nodes_per_graph

# Load the best model
early_stopping.load_best_model(model)

torch.save(model.state_dict(), "current_model.pth")

#Final evaluation on test data
# model.eval()
# correct = 0
# total = 0
# test_num_wrongly_pred_nodes_per_graph = 0

# test_all_labels = []
# test_preds = []
# #test_time_before = np.zeros(len(test_loader))
# test_time_after = np.zeros(len(test_loader))
# #test_iterations_before = np.zeros(len(test_loader))
# test_iterations_after = np.zeros(len(test_loader))
# test_iterations_difference = np.zeros(len(test_loader))
# W_diff_FN = 0
# output_FN = []
# with torch.no_grad():
#     for i,batch in enumerate(test_loader):
#         output = model(batch,number_of_layers)
#         preds = (output.squeeze() > t).long()
#         correct += (preds == batch.y).sum().item()
#         total += batch.y.size(0)
#         # Store predictions and labels
#         test_preds.extend(preds.numpy())
#         test_all_labels.extend(batch.y.numpy())
        
#         # reshape predictions for graph accuracy
#         preds = preds.reshape(-1,n+m)
        
#         # solve full QP
#         #W = []
#         # start_time_before = time.perf_counter()
#         # x_before, lambda_before, _,it_before  = self_implement_daqp.daqp_self(H,f_test[i,:],A,b_test[i,:],sense,W)
#         # end_time_before = time.perf_counter()
#         # W_before = [j for j, value in enumerate(lambda_before) if value != 0]
#         # x,_,_,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense)
#         #lambda_true =list(info.values())[4]
#         W_true = (batch.y.numpy()[n:] != 0).astype(int).nonzero()[0]
#         # test_iterations_before[i] = it_before
#         # test_time_before[i] = end_time_before- start_time_before
                
#         # solve the reduced QPs to see the reduction
#         preds_constraints = preds.flatten()[n:]
#         W_pred = torch.nonzero(preds_constraints, as_tuple=True)[0].numpy()
#         # start_time_after = time.perf_counter()
#         # x_after, lambda_after, _, it_after = self_implement_daqp.daqp_self(H,f_test[i,:],A,b_test[i,:],sense,W_pred) # sense flag 1 if active, 4 if ignore
#         # end_time_after = time.perf_counter()
#         sense_active = preds.flatten().numpy().astype(np.int32)[n:]
        
#         x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_active)
#         lambda_after= list(info.values())[4]
#         test_iterations_after[i] = list(info.values())[2]
#         test_time_after[i]= list(info.values())[0]
        
#         output_constraints = output.flatten()[n:]
#         W_pred_set = set(W_pred)
#         W_true_set = set(W_true)
#         W_diff = sorted(W_true_set.symmetric_difference(W_pred_set))
#         W_diff_FN = sorted(W_true_set.difference(W_pred_set))
#         output_FN.extend(output.flatten().numpy()[n:][W_diff_FN])
#         # print(f"Difference in W {len(set(W_true)^set(W_pred))}, It before {it_before}, If after {it_after}, It diff {it_before - it_after}")
#         print(f" W true {W_true} \n W pred {W_pred}")
#         print(output.flatten().numpy()[n:][W_diff_FN])
#         print(output.flatten().numpy()[n:][W_diff])
#         #print(output_constraints[W_pred].numpy())
#         #print(output.flatten().numpy()[n:])
#         # test_iterations_after[i] = it_after
#         # test_time_after[i] = end_time_after - start_time_after
#         test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
        
# # over graph metrices
# test_all_label_graph = np.array(test_all_labels).reshape(-1,n+m)
# test_preds_graph = np.array(test_preds).reshape(-1,n+m)

# # Compute average over graphs
# acc_graph_test = np.mean(np.all(test_all_label_graph == test_preds_graph, axis=1))
# test_num_wrongly_pred_nodes_per_graph = np.abs((n+m) - np.sum(test_all_label_graph == test_preds_graph, axis=1)) ## CHECK THIS
# print(f'correct:{correct}, total: {total}')
# print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')
# print(f'Number of graphs: {test_all_label_graph.shape[0]}, Correctly predicted graphs: {np.sum(np.all(test_all_label_graph == test_preds_graph, axis=1))}')
# print(f'Graph accuracy of the model on the test data: {100 * acc_graph_test:.2f}%')
# print(test_num_wrongly_pred_nodes_per_graph)
# print(f'Mean: {np.mean(test_num_wrongly_pred_nodes_per_graph)}')
# print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
# print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
# print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
# print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
# print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')
# print(output_FN)
# # threshold tuning
# # print(best_threshold)
# # print(best_mean)


# #Boxplot to show reduction
# boxplot_time(test_time_before,test_time_after,"time",save = False)
# boxplot_time(test_iterations_before,test_iterations_after, "iterations",save = False)
# barplot_iterations(test_iterations_before,test_iterations_after, "iterations",save = False)

# # boxplot iter differ
# plt.boxplot([test_iterations_difference],showfliers=False)
# plt.ylabel("iter difference")
# plt.xticks([1], ['from without to with GNN'])
# plt.show()

# Finish the run and upload any remaining data.
if track_on_wandb == True:
    run.finish()
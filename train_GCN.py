import numpy as np
import scipy
import daqp
import numpy as np
from ctypes import * 
import wandb
import ctypes.util
from sympy import Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

from generate_mpqp_v2 import generate_qp

import torch
import torch.nn.functional as func
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv,GraphConv, LEConv
from torch_geometric.nn import MessagePassing

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from naive_model import naive_model
from generate_graph_data import generate_qp_graphs 


n = 15
m = 21
nth = 2
seed = 123
data_points = 5000
lr = 0.001
number_of_epochs = 300
layer_width = 128
number_of_layers = 3

lambda_values = [1,5,10]
best_lambda = None
best_graph_accuracy = 0
best_model = None
for lambda_graph in lambda_values:
    # Define a simple GNN model for binary classification
    class GNN(torch.nn.Module):
        def __init__(self, input_dim, output_dim,layer_width):
            torch.manual_seed(123)
            super(GNN, self).__init__()
            self.conv1 = LEConv(input_dim, layer_width)
            self.conv2 = LEConv(layer_width,layer_width)
            self.conv3 = LEConv(layer_width,layer_width)
            self.conv4 = LEConv(layer_width,layer_width)
            self.conv5 = LEConv(layer_width, output_dim)
        def forward(self, data,number_of_layers):
            x, edge_index,edge_weight = data.x.float(), data.edge_index, data.edge_attr.float()
            x = func.leaky_relu(self.conv1(x, edge_index,edge_weight),negative_slope = 0.1)
            for i in range(number_of_layers-2):
                x = func.leaky_relu(self.conv2(x,edge_index,edge_weight),negative_slope = 0.1)
            #x = func.leaky_relu(self.conv2(x,edge_index,edge_weight),negative_slope = 0.1)
            #x = func.leaky_relu(self.conv3(x,edge_index,edge_weight),negative_slope = 0.1)
            #x = func.leaky_relu(self.conv4(x,edge_index,edge_weight),negative_slope = 0.1)
            x = func.leaky_relu(self.conv5(x,edge_index,edge_weight),negative_slope = 0.1)
            return torch.sigmoid(x)  


    class EarlyStopping: # https://www.geeksforgeeks.org/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
        def __init__(self, patience=50, delta=0):
            self.patience = patience
            self.delta = delta
            self.best_score = None
            self.early_stop = False
            self.counter = 0
            self.best_model_state = None

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.best_model_state = model.state_dict()
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_model_state = model.state_dict()
                self.counter = 0

        def load_best_model(self, model):
            model.load_state_dict(self.best_model_state)

    # Generate QP problems and the corresponding graphs
    graph_train, graph_test, graph_val = generate_qp_graphs(n,m,nth,seed,data_points)

    # Load Data
    train_loader = DataLoader(graph_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(graph_test, batch_size = len(graph_test), shuffle = True)
    val_loader =DataLoader(graph_val,batch_size = len(graph_val),shuffle = True)

    # Compute class weights for imbalanced classes
    all_labels = torch.cat([data.y for data in graph_train])
    class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Instantiate model and optimizer
    model = GNN(input_dim=3, output_dim=1,layer_width = 128)  # Output dimension 1 for binary classification
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=10, delta=0.0001)


    epoch = 0
    acc = 0
    
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
            "number of layers": f"{number_of_layers}"
        },
    )


    for epoch in range(number_of_epochs):
    #while acc != 1:
        epoch += 1
        train_loss = 0
        correct = 0
        #num_batches = 0
        #save_preds = []
        train_all_labels = []
        train_preds = []
        model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch,number_of_layers)
            loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()])(output.squeeze(), batch.y.float())
            
            # also include graph loss
            label_graph = batch.y.float().reshape(-1,n+m)
            preds_graph = output.squeeze().reshape(-1,n+m)
            preds_graphlevel = np.mean(np.array((label_graph ==preds_graph)*1),axis=1)
            label_graphlevel = np.array([1 for i in range(preds_graphlevel.shape[0])])
            graph_loss = torch.nn.BCELoss()(torch.tensor(preds_graphlevel,dtype=torch.float64), torch.tensor(label_graphlevel,dtype = torch.float64))
            loss = loss + lambda_graph*graph_loss
            
            loss.backward()
            optimizer.step()
            
            # Compute loss
            train_loss += loss.item()
            #num_batches += 1

            # Convert output to binary prediction (0 or 1)
            #save_loss += output.tolist()
            preds = (output.squeeze() > 0.50).long()
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
        if acc_graph > best_graph_accuracy:
            best_graph_accuracy = acc_graph
            best_lambda = lambda_graph
            best_model = model.state_dict()
        
        print(f"Lambda: {lambda_graph}, Accuracy: {acc_graph}")  
        # Validation step
        model.eval()
        val_loss = 0
        val_all_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch,number_of_layers)
                loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())
                val_loss += loss.item()
                preds = (output.squeeze() > 0.5).long()
                correct += (preds == batch.y).sum().item()
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
        
        # Log metrics to wandb.
        run.log({"acc_train": acc,"acc_test": val_acc,"loss_train": train_loss, "loss_test": val_loss, "prec": prec, "rec": rec, "f1": f1, "acc_graph": acc_graph, "acc_graph_test": acc_graph_val}) #"prec_graph":prec_graph, "rec_graph":rec_graph,"f1_graph":f1_graph})

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs.")
            break
        
    # Load the best model
    early_stopping.load_best_model(model)

    #Final evaluation on test data
    model.eval()
    correct = 0
    total = 0
    test_all_labels = []
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch,number_of_layers)
            preds = (output.squeeze() > 0.5).long()
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
            test_preds.extend(preds.numpy())   # Store predictions
            test_all_labels.extend(batch.y.numpy()) # Store true labels
            
    # over graph metrices
    test_all_label_graph = np.array(test_all_labels).reshape(-1,n+m)
    test_preds_graph = np.array(test_preds).reshape(-1,n+m)

    # Compute average over graphs
    acc_graph_test = np.mean(np.all(test_all_label_graph == test_preds_graph, axis=1))

    print(f'correct:{correct}, total: {total}')
    print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')
    print(f'Number of graphs: {test_all_label_graph.shape[0]}, Correctly predicted graphs: {np.sum(np.all(test_all_label_graph == test_preds_graph, axis=1))}')
    print(f'Graph accuracy of the model on the test data: {100 * acc_graph_test:.2f}%')

            
        
    # Finish the run and upload any remaining data.
    run.finish()
    
# Once the best lambda is found, you can use the model for inference
print(f"Best lambda: {best_lambda}, Best accuracy: {best_graph_accuracy}")
model.load_state_dict(best_model) 
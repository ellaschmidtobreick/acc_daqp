import numpy as np
from ctypes import * 
from sklearn.utils.class_weight import compute_class_weight
import wandb

import torch
from torch.utils.data import DataLoader as MLPDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

from generate_graph_data import generate_qp_graphs_train_val,generate_qp_graphs_train_val_lmpc
from generate_MLP_data import generate_qp_MLP_train_val
from model import GNN,MLP
from model import EarlyStopping
import matplotlib.pyplot as plt

def train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, H_flexible,A_flexible,modelname,scale_H=1,dataset_type="standard",conv_type="LEConv",two_sided = False,cuda = 0):
    
    # Initialization      
    graph_train = []
    graph_val = []
    n_vector_train = []
    m_vector_train = []
    n_vector_val = []
    m_vector_val = []

    # Select device
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate QP problems and the corresponding graphs
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        if dataset_type == "standard":
            graph_train_i, graph_val_i = generate_qp_graphs_train_val(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)
        elif dataset_type == "lmpc":
            graph_train_i, graph_val_i = generate_qp_graphs_train_val_lmpc(n_i,m_i,nth,seed,data_points,scale=scale_H,two_sided=two_sided)


        graph_train = graph_train + graph_train_i
        graph_val = graph_val + graph_val_i
        n_vector_train = n_vector_train + [n_i for j in range(len(graph_train_i))]
        m_vector_train= m_vector_train + [m_i for j in range(len(graph_train_i))]
        n_vector_val = n_vector_val + [n_i for j in range(len(graph_val_i))]
        m_vector_val = m_vector_val + [m_i for j in range(len(graph_val_i))]

    # Load Data into DataLoader
    train_batch_size = 64
    train_loader = GraphDataLoader(graph_train, batch_size=train_batch_size, shuffle=True)
    val_loader = GraphDataLoader(graph_val,batch_size = train_batch_size , shuffle = False) #len(graph_val)

    # Compute class weights for imbalanced classes
    all_labels = torch.cat([data.y for data in graph_train]).to(device)
    unique_classes = torch.unique(all_labels)
    class_weights_np = compute_class_weight('balanced', classes=unique_classes.cpu().numpy(), y=all_labels.cpu().numpy()) # torch.unique(all_labels).numpy()
    #class_weights_np[1] = class_weights_np[1]*0.5
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device) # torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("class weights: ", class_weights_np)

    # Instantiate model and optimizer
    if dataset_type == "standard" or two_sided == False:
        input_size = 4
    else : 
        input_size = 6
    model = GNN(input_dim=input_size, output_dim=1,layer_width = layer_width,conv_type = conv_type) #Input dimensions: # features 4 # Output dimension 1 for binary classification
    p_pos = (all_labels.sum() / len(all_labels)).item()
    print("positive weight", p_pos)
    model.init_weights(p_pos)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Track parameters on wandb
    if track_on_wandb ==True:
        run = wandb.init(
            entity="ella-schmidtobreick-4283-me",
            project="L4DC",
            config={
                "variables": f"{n}",
                "constraints": f"{m}",
                "datapoints": f"{data_points}",
                "epochs": f"{number_of_max_epochs}",
                "architecture": "LEConv with weights",
                "learning_rate": f"{lr}",
                "layer width": f"{layer_width}",
                "number of layers": f"{number_of_layers}",
                "threshold": f"{t}"
            },
        )

    # Training
    for epoch in range(number_of_max_epochs):
        if epoch % 10:
            print(f"Epoch {epoch}")

        ################
        # Training
        ################

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_TP = 0
        train_FP = 0
        train_FN = 0
        train_num_wrong_nodes = 0
        train_total_nodes = 0
        
        for i,batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch,number_of_layers,conv_type)
            preds = (output.squeeze() > t).long()

            if dataset_type == "standard":
                loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()].to(device))(output.squeeze(), batch.y.float())
            elif dataset_type == "lmpc":
                sparsity_loss = output.squeeze().sum()/batch.num_graphs
                BCE_loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()].to(device))(output.squeeze(), batch.y.float())
                loss = BCE_loss + 0.1 * sparsity_loss
                # if epoch % 10 == 0:  # Log occasionally
                #     print(f"BCE: {BCE_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}, Total: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if device == "cuda" and i % 50 == 0:
                torch.cuda.empty_cache()
            
            # Node-level metrics
            labels = batch.y
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()
            train_TP += ((preds == 1) & (labels == 1)).sum().item()
            train_FP += ((preds == 1) & (labels == 0)).sum().item()
            train_FN += ((preds == 0) & (labels == 1)).sum().item()
            train_num_wrong_nodes += (preds != labels).sum().item()
            train_total_nodes += labels.numel()


        # Compute metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_prec = train_TP / (train_TP + train_FP + 1e-8)
        train_rec = train_TP / (train_TP + train_FN + 1e-8)
        train_f1 = 2 * train_prec * train_rec / (train_prec + train_rec + 1e-8)

        ####################
        # VALIDATION
        ####################
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_TP = 0
        val_FP = 0
        val_FN = 0
        val_num_wrong_nodes = 0
        val_total_nodes = 0

        # best_f1 = 0.0
        # best_threshold = 0.0
        # thresholds = np.linspace(0, 1, 11)        
        # for t in thresholds:
        #     val_correct = val_total = val_TP = val_FP = val_FN = 0

        with torch.no_grad():
            for i,batch in enumerate(val_loader):
                batch = batch.to(device)
                output = model(batch,number_of_layers,conv_type)
                preds = (output.squeeze() > t).long()

                if dataset_type == "standard":
                    loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())
                elif dataset_type == "lmpc":
                    sparsity_loss = output.squeeze().sum()/batch.num_graphs
                    BCE_loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()].to(device))(output.squeeze(), batch.y.float())
                    loss = BCE_loss + 0.1 * sparsity_loss
                    # print(f"BCE: {BCE_loss.item():.4f}, Sparsity: {sparsity_loss.item():.4f}, Total: {loss.item():.4f}")

                val_loss += loss.item()

                # Node-level metrics
                labels = batch.y
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
                val_TP += ((preds == 1) & (labels == 1)).sum().item()
                val_FP += ((preds == 1) & (labels == 0)).sum().item()
                val_FN += ((preds == 0) & (labels == 1)).sum().item()
                val_num_wrong_nodes += (preds != labels).sum().item()
                val_total_nodes += labels.numel()

                if device == "cuda" and i % 50 == 0:
                    torch.cuda.empty_cache()             
        
        # Compute metrics
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_prec = val_TP / (val_TP + val_FP + 1e-8)
        val_rec = val_TP / (val_TP + val_FN + 1e-8)
        val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec + 1e-8)

        #     if val_f1 > best_f1:
        #         best_f1 = val_f1
        #         best_threshold = t

        # print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")

        # Log metrics to wandb.
        if track_on_wandb == True:
            run.log({"acc_train": train_acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec_train": train_prec, "prec_val":val_prec, "rec_train": train_rec, "rec_val": val_rec, "f1_train": train_f1,"f1_val":val_f1})

        # Early stopping
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    # Save best model
    torch.save(model.state_dict(), f"saved_models/{modelname}.pth")

    # Finish the run and upload any remaining data.
    if track_on_wandb == True:
        run.finish()
    
    # Print metrics
    print("TRAINING")
    print(f"Accuracy (node level) of the final model: {train_acc}")
    print(f"Precision of the model on the test data: {train_prec}")
    print(f"Recall of the model on the test data: {train_rec}")
    print(f"F1-Score of the model on the test data: {train_f1}")

    print("VALIDATION")
    print(f"Accuracy (node level) of the final model: {val_acc}")
    print(f"Precision of the model on the test data: {val_prec}")
    print(f"Recall of the model on the test data: {val_rec}")
    print(f"F1-Score of the model on the test data: {val_f1}")

    # return val_acc_save[best_epoch-1]
    return train_acc, train_prec, train_rec, train_f1

def train_MLP(n,m,nth, seed, number_of_graphs,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, H_flexible,A_flexible,modelname,dataset_type="standard",cuda = 0):
    
    # Initialization      
    data_train = []
    data_val = []
    n_vector_train = []
    m_vector_train = []
    n_vector_val = []
    m_vector_val = []
    
    device = torch.device("cuda{cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate QP problems and the corresponding graphs
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        data_train_i, data_val_i = generate_qp_MLP_train_val(n_i,m_i,nth,seed,number_of_graphs,H_flexible=H_flexible,A_flexible=A_flexible,dataset_type=dataset_type)
        data_train.extend(data_train_i)
        data_val.extend(data_val_i)
        n_vector_train = n_vector_train + [n_i for i in range(len(data_train_i))]
        m_vector_train= m_vector_train + [m_i for i in range(len(data_train_i))]
        n_vector_val = n_vector_val + [n_i for i in range(len(data_val_i))]
        m_vector_val = m_vector_val + [m_i for i in range(len(data_val_i))] 

    # Load Data
    train_batch_size = 64
    train_loader = MLPDataLoader(data_train, batch_size=train_batch_size, shuffle=True)
    val_loader = MLPDataLoader(data_val,batch_size = len(data_val), shuffle = False)

    # Compute class weights for imbalanced classes
    all_labels = torch.cat([data[1] for data in data_train])
    class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy().flatten())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Instantiate model and optimizer
    input_dimension = n[0]*n[0]+m[0]*n[0]+n[0]+m[0]
    output_dimension = n[0] + m[0]
    model = MLP(input_dim=input_dimension, output_dim=output_dimension,layer_width = layer_width)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Track parameters on wandb
    if track_on_wandb ==True:
        run = wandb.init(
            entity="ella-schmidtobreick-4283-me",
            project="L4DC",
            config={
                "variables": f"{n}",
                "constraints": f"{m}",
                "datapoints": f"{number_of_graphs}",
                "epochs": f"{number_of_max_epochs}",
                "architecture": "LEConv with weights",
                "learning_rate": f"{lr}",
                "layer width": f"{layer_width}",
                "number of layers": f"{number_of_layers}",
                "threshold": f"{t}"
            },
        )

    #############
    # Training
    #############
    for epoch in range(number_of_max_epochs):
        #print(f"Epoch {epoch}")
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_TP = 0
        train_FP = 0
        train_FN = 0
        train_num_wrong_nodes = 0
        train_total_nodes = 0
        
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            inputs, labels = batch
            optimizer.zero_grad()
            output = model(inputs)
            preds = (output.squeeze() > t).long()
            sparsity_loss = output.squeeze().sum()/len(batch)
            loss = torch.nn.BCELoss(weight=class_weights[batch[1]].to(device))(output.squeeze(), labels.float())#+0.1*sparsity_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Node-level metrics
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()
            train_TP += ((preds == 1) & (labels == 1)).sum().item()
            train_FP += ((preds == 1) & (labels == 0)).sum().item()
            train_FN += ((preds == 0) & (labels == 1)).sum().item()
            train_num_wrong_nodes += (preds != labels).sum().item()
            train_total_nodes += labels.numel()

        # Compute metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_prec = train_TP / (train_TP + train_FP + 1e-8)
        train_rec = train_TP / (train_TP + train_FN + 1e-8)
        train_f1 = 2 * train_prec * train_rec / (train_prec + train_rec + 1e-8)

        # Validation step
        model.eval()  
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_TP = 0
        val_FP = 0
        val_FN = 0
        val_num_wrong_nodes = 0
        val_total_nodes = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                inputs, labels = batch
                output = model(inputs)
                sparsity_loss = output.squeeze().sum()/len(batch)
                loss = torch.nn.BCELoss()(output.squeeze(), labels.float())# + 0.1*sparsity_loss
                val_loss += loss.item()
                preds = (output.squeeze() > t).long()

                # Node-level metrics
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
                val_TP += ((preds == 1) & (labels == 1)).sum().item()
                val_FP += ((preds == 1) & (labels == 0)).sum().item()
                val_FN += ((preds == 0) & (labels == 1)).sum().item()
                val_num_wrong_nodes += (preds != labels).sum().item()
                val_total_nodes += labels.numel()

                if device == "cuda" and i % 50 == 0:
                    torch.cuda.empty_cache()              
        
        # Compute metrics
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_prec = val_TP / (val_TP + val_FP + 1e-8)
        val_rec = val_TP / (val_TP + val_FN + 1e-8)
        val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec + 1e-8)

        # Log metrics to wandb.
        if track_on_wandb == True:
            run.log({"acc_train": train_acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec_train": train_prec, "prec_val":val_prec, "rec_train": train_rec, "rec_val": val_rec, "f1_train": train_f1,"f1_val":val_f1, "threshold": t})

        # Early stopping
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    # Save best model
    torch.save(model.state_dict(), f"saved_models/{modelname}.pth")

    # Finish the run and upload any remaining data.
    if track_on_wandb == True:
        run.finish()
    
    # Print metrics
    print("TRAINING")
    print(f"Accuracy (node level) of the final model: {train_acc}")
    print(f"Precision of the model on the test data: {train_prec}")
    print(f"Recall of the model on the test data: {train_rec}")
    print(f"F1-Score of the model on the test data: {train_f1}")

    print("VALIDATION")
    print(f"Accuracy (node level) of the final model: {val_acc}")
    print(f"Precision of the model on the test data: {val_prec}")
    print(f"Recall of the model on the test data: {val_rec}")
    print(f"F1-Score of the model on the test data: {val_f1}")

    return val_acc

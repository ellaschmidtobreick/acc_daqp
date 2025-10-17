import numpy as np
from ctypes import * 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.utils.class_weight import compute_class_weight
import wandb

import torch
from torch.utils.data import DataLoader as MLPDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader

from generate_graph_data import generate_qp_graphs_train_val,generate_qp_graphs_train_val_lmpc
from generate_MLP_data import generate_qp_MLP_train_val
from model import GNN,MLP
from model import EarlyStopping


def train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, H_flexible,A_flexible,modelname,scale_H=1,dataset_type="standard",conv_type="LEConv"):
    
    # Initialization      
    graph_train = []
    graph_val = []
    n_vector_train = []
    m_vector_train = []
    n_vector_val = []
    m_vector_val = []

    train_acc_save = []
    val_acc_save = []
    train_loss_save = []
    val_loss_save = []
    train_prec_save = []
    val_prec_save = []
    train_rec_save = []
    val_rec_save = []
    train_f1_save = []
    val_f1_save = []
    acc_graph_train_save = []
    acc_graph_val_save = []
    val_perc_wrongly_pred_nodes_per_graph_save = []
    val_mean_wrongly_pred_nodes_per_graph_save = []
    train_perc_wrongly_pred_nodes_per_graph_save = []
    train_mean_wrongly_pred_nodes_per_graph_save = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Generate QP problems and the corresponding graphs
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        if dataset_type == "standard":
            graph_train_i, graph_val_i = generate_qp_graphs_train_val(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)
        elif dataset_type == "lmpc":
            graph_train_i, graph_val_i = generate_qp_graphs_train_val_lmpc(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible,scale=scale_H)

        graph_train = graph_train + graph_train_i
        graph_val = graph_val + graph_val_i
        n_vector_train = n_vector_train + [n_i for i in range(len(graph_train_i))]
        m_vector_train= m_vector_train + [m_i for i in range(len(graph_train_i))]
        n_vector_val = n_vector_val + [n_i for i in range(len(graph_val_i))]
        m_vector_val = m_vector_val + [m_i for i in range(len(graph_val_i))]


    # Load Data
    train_batch_size = 16 #32 # 64
    train_loader = GraphDataLoader(graph_train, batch_size=train_batch_size, shuffle=True)
    val_loader = GraphDataLoader(graph_val,batch_size = len(graph_val), shuffle = False)

    # Compute class weights for imbalanced classes
    all_labels = torch.cat([data.y for data in graph_train]).to(device)
    unique_classes = torch.unique(all_labels)
    class_weights_np = compute_class_weight('balanced', classes=unique_classes.cpu().numpy(), y=all_labels.cpu.numpy()) # torch.unique(all_labels).numpy()
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device) # torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Instantiate model and optimizer
    model = GNN(input_dim=4, output_dim=1,layer_width = layer_width,conv_type = conv_type)  # Output dimension 1 for binary classification
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Track parameters on wandb
    if track_on_wandb ==True:
        run = wandb.init(
            entity="ella-schmidtobreick-4283-me",
            project="Thesis",
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
        # print(f"Epoch {epoch}")
        train_loss = 0
        train_all_labels = []
        train_preds = []
        model.train()
        output_train = []
        train_all_label_graph = []
        train_preds_graph = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch,number_of_layers,conv_type)
            output_train.extend(output.squeeze().detach().cpu().numpy().reshape(-1))
            loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()].to(device))(output.squeeze(), batch.y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Convert output to binary prediction (0 or 1)
            preds = (output.squeeze() > t).long()

            # Store predictions and true labels
            train_preds.extend(preds.cpu().numpy())   
            train_all_labels.extend(batch.y.cpu().numpy())
            

            # print("true labels",batch.y.shape, torch.nonzero(batch.y).squeeze().detach().numpy())
            # print("preds",preds.shape, torch.nonzero(preds).squeeze().detach().numpy())
            # print("output",output.squeeze()[torch.nonzero(batch.y).squeeze().detach().numpy()].detach().numpy())
            # print("loss",loss.item())

            # Save per graph predictions and labels
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                preds_graph = preds[mask].cpu().numpy()
                labels_graph = batch.y[mask].cpu().numpy()
                
                #print(np.sum(preds_graph),np.sum(labels_graph))

                train_preds_graph.append(preds_graph)
                train_all_label_graph.append(labels_graph)

        # Compute metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_all_labels, train_preds)
        train_prec = precision_score(train_all_labels,train_preds)
        train_rec = recall_score(train_all_labels, train_preds)
        train_f1 = f1_score(train_all_labels,train_preds)
        acc_graph_train = np.mean([np.all(pred == true) for pred, true in zip(train_preds_graph, train_all_label_graph)]) # average on graph level

        train_num_wrongly_pred_nodes_per_graph = [np.abs(int(n_i + m_i) - np.sum(pred == label)) for pred, label, n_i, m_i in zip(train_preds_graph, train_all_label_graph, n_vector_train, m_vector_train)]
        train_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n_i + m_i) for wrong, n_i, m_i in zip(train_num_wrongly_pred_nodes_per_graph, n_vector_train, m_vector_train)])
        train_mean_wrongly_pred_nodes_per_graph = np.mean(train_num_wrongly_pred_nodes_per_graph)
        
        # Validation step
        model.eval()
        val_loss = 0
        val_mean_wrongly_pred_nodes_per_graph = 0
        val_num_wrongly_pred_nodes_per_graph = 0
        val_all_labels = []
        val_preds = []
        output_val = []
        val_preds_graph = []
        val_all_label_graph = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch,number_of_layers,conv_type)
                output_val.extend(output.squeeze().detach().cpu().numpy().reshape(-1))
                loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())

                val_loss += loss.item()
                preds = (output.squeeze() > t).long()

                # Store predictions and labels
                val_preds.extend(preds.cpu().numpy())
                val_all_labels.extend(batch.y.cpu().numpy())
                
                # Store per graph predictions and labels
                for i in range(batch.num_graphs):
                    mask = batch.batch == i
                    preds_graph = preds[mask].cpu().numpy()
                    labels_graph = batch.y[mask].cpu().numpy()

                    val_preds_graph.append(preds_graph)
                    val_all_label_graph.append(labels_graph)                
        
        # Compute metrics      
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_all_labels, val_preds)
        val_prec = precision_score(val_all_labels,val_preds)
        val_rec = recall_score(val_all_labels, val_preds)
        val_f1 = f1_score(val_all_labels,val_preds)
        acc_graph_val = np.mean([np.all(pred == true) for pred, true in zip(val_preds_graph, val_all_label_graph)]) # accuracy on graph level

        val_num_wrongly_pred_nodes_per_graph = [np.abs(int(n_i + m_i) - np.sum(pred == label)) for pred, label, n_i, m_i in zip(val_preds_graph, val_all_label_graph, n_vector_val, m_vector_val)]
        val_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n_i + m_i) for wrong, n_i, m_i in zip(val_num_wrongly_pred_nodes_per_graph, n_vector_val, m_vector_val)])
        val_mean_wrongly_pred_nodes_per_graph = np.mean(val_num_wrongly_pred_nodes_per_graph)
        
        # Log metrics to wandb.
        if track_on_wandb == True:
            run.log({"acc_train": train_acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec_train": train_prec, "prec_val":val_prec, "rec_train": train_rec, "rec_val": val_rec, "f1_train": train_f1,"f1_val":val_f1, "acc_graph_train": acc_graph_train, "acc_graph_val": acc_graph_val,"perc_wrong_pred_nodes_per_graph_val": val_perc_wrongly_pred_nodes_per_graph,"num_wrong_pred_nodes_per_graph_val":val_mean_wrongly_pred_nodes_per_graph, "threshold": t})

        # Save metrics
        train_acc_save.append(train_acc)
        val_acc_save.append(val_acc)
        train_loss_save.append(train_loss)
        val_loss_save.append(val_loss)
        train_prec_save.append(train_prec)
        val_prec_save.append(val_prec)
        train_rec_save.append(train_rec)
        val_rec_save.append(val_rec)
        train_f1_save.append(train_f1)
        val_f1_save.append(val_f1)
        acc_graph_train_save.append(acc_graph_train)
        acc_graph_val_save.append(acc_graph_val)
        val_perc_wrongly_pred_nodes_per_graph_save.append(val_perc_wrongly_pred_nodes_per_graph)
        val_mean_wrongly_pred_nodes_per_graph_save.append(val_mean_wrongly_pred_nodes_per_graph)
        train_perc_wrongly_pred_nodes_per_graph_save.append(train_perc_wrongly_pred_nodes_per_graph)
        train_mean_wrongly_pred_nodes_per_graph_save.append(train_mean_wrongly_pred_nodes_per_graph)

        # Early stopping
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    # Save best model
    best_epoch = early_stopping.load_best_model(model)
    torch.save(model.state_dict(), f"saved_models/{modelname}.pth")

    # Finish the run and upload any remaining data.
    if track_on_wandb == True:
        run.finish()
    
    # Print metrics
    print("TRAINING")
    print(f"Accuracy (node level) of the final model: {train_acc_save[best_epoch-1]}")
    print(f"Precision of the model on the test data: {train_prec_save[best_epoch-1]}")
    print(f"Recall of the model on the test data: {train_rec_save[best_epoch-1]}")
    print(f"F1-Score of the model on the test data: {train_f1_save[best_epoch-1]}")
    print(f"Accuracy (graph level) of the model on the test data: {acc_graph_train_save[best_epoch-1]}")
    print(f"Perc num_wrongly_pred_nodes_per_graph: {train_perc_wrongly_pred_nodes_per_graph_save[best_epoch-1]}")

    print("VALIDATION")
    print(f"Accuracy (node level) of the final model: {val_acc_save[best_epoch-1]}")
    print(f"Precision of the model on the test data: {val_prec_save[best_epoch-1]}")
    print(f"Recall of the model on the test data: {val_rec_save[best_epoch-1]}")
    print(f"F1-Score of the model on the test data: {val_f1_save[best_epoch-1]}")
    print(f"Accuracy (graph level) of the model on the test data: {acc_graph_val_save[best_epoch-1]}")
    print(f"Perc num_wrongly_pred_nodes_per_graph: {val_perc_wrongly_pred_nodes_per_graph_save[best_epoch-1]}")

    print(np.sum(train_all_labels), len(train_all_labels),np.sum(train_all_labels)/len(train_all_labels))
    return val_acc_save[best_epoch-1]

def train_MLP(n,m,nth, seed, number_of_graphs,lr,number_of_max_epochs,layer_width,number_of_layers, track_on_wandb,t, H_flexible,A_flexible,modelname,dataset_type="standard"):
    
    # Initialization      
    data_train = []
    data_val = []
    n_vector_train = []
    m_vector_train = []
    n_vector_val = []
    m_vector_val = []
    
    train_acc_save = []
    val_acc_save = []
    train_loss_save = []
    val_loss_save = []
    train_prec_save = []
    val_prec_save = []
    train_rec_save = []
    val_rec_save = []
    train_f1_save = []
    val_f1_save = []
    acc_graph_train_save = []
    acc_graph_val_save = []
    val_perc_wrongly_pred_nodes_per_graph_save = []
    val_mean_wrongly_pred_nodes_per_graph_save = []
    train_perc_wrongly_pred_nodes_per_graph_save = []
    train_mean_wrongly_pred_nodes_per_graph_save = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    #data_train, data_val = generate_qp_MLP_train_val(n,m,nth,seed,number_of_graphs)
   

    # Load Data
    train_batch_size = 64
    train_loader = MLPDataLoader(data_train, batch_size=train_batch_size, shuffle=True)
    val_loader = MLPDataLoader(data_val,batch_size = len(data_val), shuffle = False)

    # Compute class weights for imbalanced classes
    #all_labels = data_train.tensors[1]
    all_labels = torch.cat([data[1] for data in data_train])
    class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy().flatten())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Instantiate model and optimizer
    input_dimension = n[0]*n[0]+m[0]*n[0]+n[0]+m[0]
    output_dimension = n[0] + m[0]
    model = MLP(input_dim=input_dimension, output_dim=output_dimension,layer_width = layer_width)  # Output dimension 1 for binary classification
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    # Track parameters on wandb
    if track_on_wandb ==True:
        run = wandb.init(
            entity="ella-schmidtobreick-4283-me",
            project="Thesis",
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

    # Training
    for epoch in range(number_of_max_epochs):
        #print(f"Epoch {epoch}")
        train_loss = 0
        train_all_labels = []
        train_preds = []
        model.train()
        output_train = []
        train_all_label_graph = []
        train_preds_graph = []
        
        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            output = model(batch[0])
            output_train.extend(output.squeeze().detach().cpu().numpy().reshape(-1))
            loss = torch.nn.BCELoss(weight=class_weights[batch[1]].to(device))(output.squeeze(), batch[1].float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Convert output to binary prediction (0 or 1)
            preds = (output.squeeze() > t).long()

            # Store predictions and true labels
            train_preds.extend(preds.cpu().numpy().flatten())   
            train_all_labels.extend(batch[1].cpu().numpy().flatten())
            
            # # Save per graph predictions and labels
            for i in range(batch[1].shape[0]):
                #mask = batch.batch == i
                preds_graph = preds[i].cpu().numpy()
                labels_graph = batch[1][i].cpu().numpy()

                train_preds_graph.append(preds_graph)
                train_all_label_graph.append(labels_graph)

        # Compute metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_all_labels, train_preds)
        train_prec = precision_score(train_all_labels,train_preds)
        train_rec = recall_score(train_all_labels, train_preds)
        train_f1 = f1_score(train_all_labels,train_preds)
        acc_graph_train = np.mean([np.all(pred == true) for pred, true in zip(train_preds_graph, train_all_label_graph)]) # average on graph level

        # train_num_wrongly_pred_nodes_per_graph = [np.abs(int(n + m) - np.sum(pred == label)) for pred, label in zip(train_preds_graph, train_all_label_graph)]
        # train_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n + m) for wrong  in train_num_wrongly_pred_nodes_per_graph])
        # train_mean_wrongly_pred_nodes_per_graph = np.mean(train_num_wrongly_pred_nodes_per_graph)
        
        train_num_wrongly_pred_nodes_per_graph = [np.abs(int(n_i + m_i) - np.sum(pred == label)) for pred, label, n_i, m_i in zip(train_preds_graph, train_all_label_graph, n_vector_train, m_vector_train)]
        train_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n_i + m_i) for wrong, n_i, m_i in zip(train_num_wrongly_pred_nodes_per_graph, n_vector_train, m_vector_train)])
        train_mean_wrongly_pred_nodes_per_graph = np.mean(train_num_wrongly_pred_nodes_per_graph)
        
        
        # Validation step
        model.eval()
        val_loss = 0
        val_mean_wrongly_pred_nodes_per_graph = 0
        val_num_wrongly_pred_nodes_per_graph = 0
        val_all_labels = []
        val_preds = []
        output_val = []
        val_preds_graph = []
        val_all_label_graph = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                output = model(batch[0])
                output_val.extend(output.squeeze().detach().cpu().numpy().reshape(-1))
                loss = torch.nn.BCELoss()(output.squeeze(), batch[1].float())
                val_loss += loss.item()
                preds = (output.squeeze() > t).long()

                # Store predictions and labels
                val_preds.extend(preds.cpu().numpy().flatten())
                val_all_labels.extend(batch[1].cpu().numpy().flatten())
                
                # Store per graph predictions and labels
                for i in range(batch[1].shape[0]):
                    #mask = batch.batch == i
                    preds_graph = preds[i].cpu().numpy()
                    labels_graph = batch[1][i].cpu().numpy()

                    val_preds_graph.append(preds_graph)
                    val_all_label_graph.append(labels_graph)                
        
        # Compute metrics      
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_all_labels, val_preds)
        val_prec = precision_score(val_all_labels,val_preds)
        val_rec = recall_score(val_all_labels, val_preds)
        val_f1 = f1_score(val_all_labels,val_preds)
        acc_graph_val = np.mean([np.all(pred == true) for pred, true in zip(val_preds_graph, val_all_label_graph)]) # accuracy on graph level

        val_num_wrongly_pred_nodes_per_graph = [np.abs(int(n_i + m_i) - np.sum(pred == label)) for pred, label, n_i, m_i in zip(val_preds_graph, val_all_label_graph, n_vector_val, m_vector_val)]
        val_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n_i + m_i) for wrong, n_i, m_i in zip(val_num_wrongly_pred_nodes_per_graph, n_vector_val, m_vector_val)])
        val_mean_wrongly_pred_nodes_per_graph = np.mean(val_num_wrongly_pred_nodes_per_graph)
        
        # val_num_wrongly_pred_nodes_per_graph = [np.abs(int(n + m) - np.sum(pred == label)) for pred, label in zip(val_preds_graph, val_all_label_graph)]
        # val_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n + m) for wrong in val_num_wrongly_pred_nodes_per_graph])
        # val_mean_wrongly_pred_nodes_per_graph = np.mean(val_num_wrongly_pred_nodes_per_graph)
        
        # Log metrics to wandb.
        if track_on_wandb == True:
            run.log({"acc_train": train_acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec_train": train_prec, "prec_val":val_prec, "rec_train": train_rec, "rec_val": val_rec, "f1_train": train_f1,"f1_val":val_f1, "acc_graph_train": acc_graph_train, "acc_graph_val": acc_graph_val,"perc_wrong_pred_nodes_per_graph_val": val_perc_wrongly_pred_nodes_per_graph,"num_wrong_pred_nodes_per_graph_val":val_mean_wrongly_pred_nodes_per_graph, "threshold": t})

        # Save metrics
        train_acc_save.append(train_acc)
        val_acc_save.append(val_acc)
        train_loss_save.append(train_loss)
        val_loss_save.append(val_loss)
        train_prec_save.append(train_prec)
        val_prec_save.append(val_prec)
        train_rec_save.append(train_rec)
        val_rec_save.append(val_rec)
        train_f1_save.append(train_f1)
        val_f1_save.append(val_f1)
        acc_graph_train_save.append(acc_graph_train)
        acc_graph_val_save.append(acc_graph_val)
        val_perc_wrongly_pred_nodes_per_graph_save.append(val_perc_wrongly_pred_nodes_per_graph)
        val_mean_wrongly_pred_nodes_per_graph_save.append(val_mean_wrongly_pred_nodes_per_graph)
        train_perc_wrongly_pred_nodes_per_graph_save.append(train_perc_wrongly_pred_nodes_per_graph)
        train_mean_wrongly_pred_nodes_per_graph_save.append(train_mean_wrongly_pred_nodes_per_graph)

        # Early stopping
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    # Save best model
    best_epoch = early_stopping.load_best_model(model)
    torch.save(model.state_dict(), f"saved_models/{modelname}.pth")

    # Finish the run and upload any remaining data.
    if track_on_wandb == True:
        run.finish()
    
    # Print metrics
    print("TRAINING")
    print(f"Accuracy (node level) of the final model: {train_acc_save[best_epoch-1]}")
    print(f"Precision of the model on the test data: {train_prec_save[best_epoch-1]}")
    print(f"Recall of the model on the test data: {train_rec_save[best_epoch-1]}")
    print(f"F1-Score of the model on the test data: {train_f1_save[best_epoch-1]}")
    print(f"Accuracy (graph level) of the model on the test data: {acc_graph_train_save[best_epoch-1]}")
    print(f"Perc num_CORRECTLY_pred_nodes_per_graph: {1-train_perc_wrongly_pred_nodes_per_graph_save[best_epoch-1]}")

    print("VALIDATION")
    print(f"Accuracy (node level) of the final model: {val_acc_save[best_epoch-1]}")
    print(f"Precision of the model on the test data: {val_prec_save[best_epoch-1]}")
    print(f"Recall of the model on the test data: {val_rec_save[best_epoch-1]}")
    print(f"F1-Score of the model on the test data: {val_f1_save[best_epoch-1]}")
    print(f"Accuracy (graph level) of the model on the test data: {acc_graph_val_save[best_epoch-1]}")
    print(f"Perc num_CORRECTLY_pred_nodes_per_graph: {1-val_perc_wrongly_pred_nodes_per_graph_save[best_epoch-1]}")

    return val_acc_save[best_epoch-1]

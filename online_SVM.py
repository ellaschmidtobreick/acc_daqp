import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from ctypes import * 
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data

from model import GNN
from model import EarlyStopping

# Gaussian kernel
def rbf_kernel(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X
    X_norm = np.sum(X**2, axis=1)[:, None]
    Y_norm = np.sum(Y**2, axis=1)[None, :]
    K = np.exp(-gamma * (X_norm + Y_norm - 2 * X @ Y.T))
    return K

def solve_svm_dual(X, y, C=1.0, gamma=1.0):
    n = X.shape[0]
    K = rbf_kernel(X, gamma=gamma)
    # QP variables
    alpha = cp.Variable(n)
    # Dual objective
    obj = cp.Maximize(cp.sum(alpha) - 0.5 * cp.quad_form(cp.multiply(y, alpha), K))
    # Constraints
    constraints = [alpha >= 0, alpha <= C, cp.sum(cp.multiply(alpha, y)) == 0]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver="OSQP", verbose=False)
    return alpha.value

# Decision function
def decision_function(X_test, Xs, ys, alpha, gamma):
    K = rbf_kernel(X_test, Xs, gamma=gamma)
    return (alpha * ys) @ K.T

def contour_plot(X, y, Xs, ys, alpha, gamma):
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = decision_function(X_grid, Xs, ys, alpha, gamma).reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")   # decision boundary
    plt.contour(xx, yy, Z, levels=[-0.1, 0.1], linestyles="--", colors="k")  # margins
    plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr, edgecolors="k")
    plt.scatter(Xs[:,0], Xs[:,1], s=100, facecolors="none", edgecolors="g", label="Support vectors")
    plt.legend()
    plt.title("SVM Decision Boundary")
    plt.show()


## predict support vectors using GNN

# training dataset - data in each online step
# label - if a point is in the support set or not

# generate all data first, act as "time series" and split it test, val, train later
def generate_qp_graphs_svm(X,y,budget,C,gamma,seed):
    np.random.seed(seed)
    # Generate the graph from the data
    graphs = []
    support_set = []

    for t in range(len(y)): 
        support_set.append(t)
        if len(support_set) > budget:
            # Simple budget policy: drop oldest
            support_set.pop(0)
            # Remove the one least contributing to the margin
            # alphas = solve_svm_dual(X[support_set], y[support_set], C=C, gamma=gamma)
            # min_alpha_idx = np.argmin(alphas)
            # support_set.pop(min_alpha_idx)
        Xs, ys = X[support_set], y[support_set]
        #print(Xs.shape,ys.shape, support_set)
        
        alpha = solve_svm_dual(Xs, ys, C=C, gamma=gamma)

        # optimal labels for current support set
        #y_train1 = torch.tensor(((np.zeros((Xs.shape[0])*3)))) #,np.where(np.array([i in support_set for i in range(len(ys))]), 1, 0),np.where(np.array([i in support_set for i in range(len(ys))]), 1, 0)))))      #### HERE HERE HERE        # 
        y_train = torch.tensor(((np.hstack((np.zeros((X[:(t+1)].shape[0])),np.where(np.array([i in support_set for i in range(len(y[:(t+1)]))]), 1, 0),np.where(np.array([i in support_set for i in range(len(y[:(t+1)]))]), 1, 0))))))

        ### WRONG wRONG WRONG
        # check def. of y_train and what should be predicted

        #print(y_train.shape)
        #print(y_train)
        K = rbf_kernel(Xs, gamma=gamma)
        K = rbf_kernel(X[:(t+1)], gamma=gamma)
        

        # = np.transpose(ys)*K * ys
        H = np.transpose(y[:(t+1)])*K * y[:(t+1)]
        f = np.ones(len(y[:(t+1)])) * -1    #np.where(np.array([i in support_set for i in range(len(y))]), -1, 0)
        A = np.vstack((-np.eye(t+1), np.eye(t+1)))# , ys, -ys))
        b = np.hstack(((np.zeros(t+1)), C*np.ones(t+1))) #, 0, 0))
        print(K.shape,H.shape,f.shape,A.shape,b.shape)

        # graph structure does not change, only vertex features
        #combine H and A
        edge_matrix = np.block([[H,A.T],[A,np.zeros((np.shape(A)[0],np.shape(A)[0]))]])
        
        # create edge_index and edge_attributes
        edge_index = torch.tensor([])
        edge_attr = torch.tensor([])
        for j in range(np.shape(edge_matrix)[0]):
            for k in range(np.shape(edge_matrix)[1]):
                # add edge
                if edge_matrix[j,k] != 0:
                    edge_index = torch.cat((edge_index,torch.tensor([[j,k]])),0)
                    edge_attr = torch.cat((edge_attr,torch.tensor([edge_matrix[j,k]])),0)
        edge_index = edge_index.long().T
        
        # create new vectors filled with zeros to capture vertex features better
        f1_train = np.hstack((f,np.zeros(np.shape(b))))
        b1_train = np.hstack((np.zeros(np.shape(f)),b))
        eq1_train = np.hstack((np.zeros(np.shape(f)),(np.zeros(np.shape(b)))))
        node_type_train = np.hstack((np.zeros(np.shape(f)),(np.ones(np.shape(b)))))

        #print(f1_train.shape,b1_train.shape,eq1_train.shape)

        features = np.array([f1_train, b1_train, eq1_train,node_type_train]).T
        x_train = torch.tensor(features, dtype=torch.float32)
        # print(y_train)
        data_point = Data(x= x_train, edge_index=edge_index, edge_attr=edge_attr,y=y_train)
        #print(data_point)
        # list of graph elements
        graphs.append(data_point)
        print(data_point)
    return graphs

def train_val_test_GNN_oSVM(graph_train, graph_val,graph_test, number_of_max_epochs,layer_width,number_of_layers,t, conv_type, model_name):
    
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

    # Load Data
    train_batch_size = 16
    train_loader = GraphDataLoader(graph_train, batch_size=train_batch_size, shuffle=False)
    val_loader = GraphDataLoader(graph_val,batch_size = len(graph_val), shuffle = False)

    # Compute class weights for imbalanced classes
    all_labels = np.concat([data.y for data in graph_train])
    # class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy())
    # class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Instantiate model and optimizer
    model = GNN(input_dim=4, output_dim=1,layer_width = layer_width,conv_type = conv_type)  # Output dimension 1 for binary classification
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)


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
            optimizer.zero_grad()
            output = model(batch,number_of_layers,conv_type)
            # print(output.shape)
            # print(output)
            output_train.extend(output.squeeze().detach().numpy().reshape(-1))
            # print(batch.y.shape)
            print("dimensions",output.squeeze().shape, batch.y.shape)
            loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float()) #weight=class_weights[batch.y.long()]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Convert output to binary prediction (0 or 1)
            preds = (output.squeeze() > t).long()

            # Store predictions and true labels
            train_preds.extend(preds.numpy())   
            train_all_labels.extend(batch.y.numpy())
            

            # print("true labels",batch.y.shape, torch.nonzero(batch.y).squeeze().detach().numpy())
            # print("preds",preds.shape, torch.nonzero(preds).squeeze().detach().numpy())
            # print("output",output.squeeze()[torch.nonzero(batch.y).squeeze().detach().numpy()].detach().numpy())
            # print("loss",loss.item())

            # Save per graph predictions and labels
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                preds_graph = preds[mask].numpy()
                labels_graph = batch.y[mask].numpy()
                
                #print(np.sum(preds_graph),np.sum(labels_graph))

                train_preds_graph.append(preds_graph)
                train_all_label_graph.append(labels_graph)

        # Compute metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_all_labels, train_preds)
        train_prec = precision_score(train_all_labels,train_preds)
        train_rec = recall_score(train_all_labels, train_preds)
        train_f1 = f1_score(train_all_labels,train_preds)

        # Validation step
        model.eval()
        val_loss = 0
        val_all_labels = []
        val_preds = []
        output_val = []
        val_preds_graph = []
        val_all_label_graph = []
        
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch,number_of_layers,conv_type)
                output_val.extend(output.squeeze().detach().numpy().reshape(-1))
                loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())
                val_loss += loss.item()
                preds = (output.squeeze() > t).long()

                # Store predictions and labels
                val_preds.extend(preds.numpy())
                val_all_labels.extend(batch.y.numpy())
                
                # Store per graph predictions and labels
                for i in range(batch.num_graphs):
                    mask = batch.batch == i
                    preds_graph = preds[mask].numpy()
                    labels_graph = batch.y[mask].numpy()

                    val_preds_graph.append(preds_graph)
                    val_all_label_graph.append(labels_graph)                
        
        # Compute metrics      
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_all_labels, val_preds)
        val_prec = precision_score(val_all_labels,val_preds)
        val_rec = recall_score(val_all_labels, val_preds)
        val_f1 = f1_score(val_all_labels,val_preds)

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

        # Early stopping
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping after {epoch} epochs.")
            break

    # Save best model
    best_epoch = early_stopping.load_best_model(model)
    torch.save(model.state_dict(), f"saved_models/{model_name}.pth")

        # Print metrics
    print("TRAINING")
    print(f"Accuracy (node level) of the final model: {train_acc_save[best_epoch-1]}")
    print(f"Precision of the model on the test data: {train_prec_save[best_epoch-1]}")
    print(f"Recall of the model on the test data: {train_rec_save[best_epoch-1]}")
    print(f"F1-Score of the model on the test data: {train_f1_save[best_epoch-1]}")

    print("VALIDATION")
    print(f"Accuracy (node level) of the final model: {val_acc_save[best_epoch-1]}")
    print(f"Precision of the model on the test data: {val_prec_save[best_epoch-1]}")
    print(f"Recall of the model on the test data: {val_rec_save[best_epoch-1]}")
    print(f"F1-Score of the model on the test data: {val_f1_save[best_epoch-1]}")

#def test_GNN_oSVM(graph_test,layer_width,number_of_layers,t,model_name,conv_type):

    # Load Data
    test_loader = GraphDataLoader(graph_test, batch_size = 1, shuffle = False)

    # Load model
    model = GNN(input_dim=4, output_dim=1,layer_width = layer_width,conv_type=conv_type) 
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth",weights_only=True))
    model.eval()
    
    # Initialization for testing 
    test_loss = 0
    test_all_labels = []
    test_preds = []
    output_test = []
    test_preds_graph = []
    test_all_label_graph = []
    
    # Test on data 
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch,number_of_layers,conv_type)
            output_test.extend(output.squeeze().detach().numpy().reshape(-1))
            loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())
            val_loss += loss.item()
            preds = (output.squeeze() > t).long()

            # Store predictions and labels
            test_preds.extend(preds.numpy())
            test_all_labels.extend(batch.y.numpy())
            
            # Store per graph predictions and labels
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                preds_graph = preds[mask].numpy()
                labels_graph = batch.y[mask].numpy()

                test_preds_graph.append(preds_graph)
                test_all_label_graph.append(labels_graph)   
                print(preds_graph, labels_graph)    
            
    # Compute metrics
    test_acc = accuracy_score(test_all_labels, test_preds)
    test_prec = precision_score(test_all_labels, test_preds)
    test_rec = recall_score(test_all_labels, test_preds)
    test_f1 = f1_score(test_all_labels, test_preds)

    # Compute average over graphs
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")

# Simulated online process
number_of_graphs = 100
support_set = []
budget = 10
C = 1.0
gamma = 0.5
seed = 0

np.random.seed(0)
X = np.random.randn(number_of_graphs, 2)
y = np.random.choice([-1, 1], size=number_of_graphs)


graphs = generate_qp_graphs_svm(X,y,budget,C,gamma,seed)

# train, val test split
graph_train = graphs[:int(0.8*number_of_graphs)]
graph_val = graphs[int(0.8*number_of_graphs):int(0.9*number_of_graphs)]
graph_test = graphs[int(0.9*number_of_graphs):]

# training parameters
layer_width = 128
conv_type = "LEConv"
number_of_max_epochs = 100
number_of_layers = 3
t = 0.9
model_name = "GNN_oSVM_test"

print(len(graph_train), len(graph_val),len(graph_test))
print(graph_val[0],graph_val[1])
train_val_test_GNN_oSVM(graph_train, graph_val,graph_test, number_of_max_epochs,layer_width,number_of_layers,t, conv_type, model_name)


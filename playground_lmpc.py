# import numpy
# from lmpc import MPC,ExplicitMPC

# # import os, julia
# # os.add_dll_directory(r"acc_daqp_env/julia_env/pyjuliapkg/install/bin")

# # Continuous time system dx = A x + B u

# # system matrix: 4×4 for 4 states: cart position, cart velocity, pendulum angle, pendulum angular velocity) (per row)
# # x^dot is a column vector with [x_cart, v_cart, theta, theta^dot]^T
# # the rows determine which variable gets influenced by which other variable
# # the columns determine which other variables each variable influences
# A = numpy.array([[0, 1, 0, 0], [0, -10, 9.81, 0], [0, 0, 0, 1], [0, -20, 39.24, 0]])
# # input matrix (4×1, scaled by 100 here)
# B = 100*numpy.array([0,1.0,0,2.0])
# # output matrix (maps states to outputs, here measuring cart position and pendulum angle)
# C = numpy.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])

# # These matrices define F, G, C in the discrete-time QP formula after discretization:
# # x_k+1 = Fx_k+Gu_k,    y_k = Cx_k


<<<<<<< HEAD
# create an MPC control with sample time 0.01, prediction horizon 10 and control horizon 5 
Np,Nc = 50,5
Ts = 0.01
mpc = MPC(A,B,Ts,C=C,Nc=Nc,Np=Np);
=======
# # create an MPC control with sample time 0.01, prediction horizon 10 and control horizon 5 
# # prediction horizon Np: N in the formula (number of future steps considered)
# # Nc: control horizon (number of free inputs to optimize; remaining inputs may be held constant)
# Np,Nc = 10,5
# # Ts: sample time (discrete-time conversion of A, B to F, G)
# Ts = 0.01
# # creates the QP problem template with constraints
# mpc = MPC(A,B,Ts,C=C,Nc=Nc,Np=Np);
>>>>>>> d223bb6a0c6bb14ea2be0923d6b8d14966a7371c

# # set the objective functions weights
# # Q: weight on output y=Cx for each measured output (cart position and pendulum angle)
# # R: weight on input magnitude uTRu (here zero - input magnitude is not penalized)
# # Rr: weight on input change delta u(k) = u(k)-u(k-1)
# # directly used to built matrices H and f (long complicated formula to explicitly built it)
# mpc.set_objective(Q=[1.44,1],R=[0.0],Rr=[1.0])

# # set actuator limits
# mpc.set_bounds(umin=[-2.0],umax=[2.0])

# # run 
# res = mpc.mpqp(singlesided=True)

# H = res["H"]
# f = res["f"]
# H_theta = res["H_theta"]
# f_theta = res["f_theta"]
# A = res["A"]
# b = res["b"]
# W = res["W"]
# senses = res["senses"]
# prio = res["prio"]
# has_binaries = res["has_binaries"]

# print("H",H)
# print("condition number",numpy.linalg.cond(H))
# print("f",f)
# print("H_theta",H_theta)
# print("f_theta",f_theta)
# print("A",A)
# print("b",b)
# print("W",W)
# print("senses",senses)
# print("prio",prio)
# print("has_binaries",has_binaries)
#numpy.savez(f"data/mpc_mpqp_N{int(A.shape[1])}.npz", H=H, f=f, f_theta=f_theta, A=A, b=b, W=W)

import numpy as np
# from generate_mpqp_v2 import generate_rhs 
# from generate_graph_data import generate_qp_graphs_train_val_lmpc, generate_qp_graphs_test_data_only_lmpc
# from train_model import train_GNN
# from test_model import test_GNN
# import daqp
# import torch
# from torch_geometric.data import Data

data = np.load('data/mpc_mpqp_N5.npz')

# ['H', 'f', 'f_theta', 'A', 'b', 'W']
print(data.files)

print(data['H'])    # 5x5
print(data['f'])    # 5x1 (all 0)
print(data['f_theta'])  # 5x7
print(data['A'])    # (5+5)x5 (upper & lower constraints)
print(data['b'])    # (5+5))x1 (all 2)
print(data['W'])    # (5+5)x7 (all 0)

# print("here")
# data = np.load('data/mpc_mpqp_N50.npz')
# print(data.files)

# print(data['H'])    # 10x10
# print(data['f'])    # 10x1 (all 0)
# print(data['f_theta'])  # 10x7
# print(data['A'])    # (10+10)x10 (upper & lower constraints)
# print(data['b'])    # (10+10)x1 (all 2)
# print(data['W'])    # (10+10)x7 (all 0)

# 7 is the dimension of theta (=nth)

# ftot,btot = generate_rhs(data['f'],data['f_theta'],data['b'],data['W'],7,123)

# graph_train, graph_val = generate_qp_graphs_train_val_lmpc(10,20,7,123,10,False, False)
# graph_test = generate_qp_graphs_test_data_only_lmpc(10,20,7,123,10,False, False)

# print(graph_train[0])
# print(graph_train[1])
# train_GNN([10],[20],7,123, 5000,0.001,20,128,3, False,0.6, False,False,f"model_{10}v_{20}c_lmpc")
# test_GNN([10],[20],7,123, 5000,128,3,0.6, False,False,f"model_{10}v_{20}c_lmpc")


# check how many graphs are actually produced
# print(len(graph_train))

# data = np.load('data/mpc_mpqp_N25.npz')
# print(data.files)

# data = np.load('data/mpc_mpqp_N50.npz')
# print(data.files)


# Set parameters
# n = [10]
# m = [20]
# nth = 7 #2
# seed = 123
# data_points = 100 #5000
# lr = 0.001
# number_of_max_epochs = 100
# layer_width = 2560 #128
# number_of_layers = 3
# track_on_wandb = False #True
# t = 0.6

# H_flexible = False
# A_flexible = False
# modelname = f"model_{n}v_{m}c_lmpc"

# import wandb
# from torch_geometric.loader import DataLoader
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from model import GNN, EarlyStopping
# import torch.nn.functional as func
# import torch




# #spit generated problems into train, test, val
# # iter_train = int(np.rint(0.8*number_of_graphs))
# # iter_val = int(np.rint(0.1*number_of_graphs))

# np.random.seed(seed)
# data_path = 'data/mpc_mpqp_N50.npz'
# data = np.load('data/mpc_mpqp_N50.npz')
# # H,f,F,A,b,B,T = generate_qp(n,m,seed)
# import re

# n = int(re.search(r"N(\d+)", data_path).group(1))
# m = 2*n
# print("n=",n)  # 10
# print("m=",m)  # 20

# H,f,F,A,b,B = data["H"], data["f"], data["f_theta"], data["A"], data["b"], data["W"]
# print(H.shape,f.shape,F.shape,A.shape,b.shape,B.shape) #,T.shape)
# print("condition number",np.linalg.cond(H))
# # dimension reduction for daqp solver
# f = f.squeeze()
# b=b.squeeze()

# np.savez(f"data/generated_qp_data_{n}v_{m}c_lmpc.npz", H=H, f=f, F=F, A=A, b=b, B=B) #,T=T)
# sense = np.zeros(m, dtype=np.int32)
# blower = np.array([-np.inf for i in range(m)])

# # Generate training set - only change theta and A
# x_train = np.zeros((iter_train,n))
# lambda_train = np.zeros((iter_train,m))
# train_iterations = np.zeros((iter_train))
# train_time= np.zeros((iter_train))

# # Generate the graph from the training data
# graph_train = []

# for i in range(iter_train):
#     theta = np.random.randn(nth)
        
#     btot = b + B @ theta
#     ftot = f + F @ theta
#     print("btot", btot)
#     print("ftot",ftot)
#     _,_,_,info = daqp.solve(H,ftot,A,btot,blower,sense)
#     lambda_train[i,:]= list(info.values())[4]
#     train_iterations[i] = list(info.values())[2]
#     train_time[i]= list(info.values())[0]

#     # get optimal active set (y)
#     train_active_set = (lambda_train != 0).astype(int)
#     #print(train_active_set)
#     y_train = torch.tensor((np.hstack((np.zeros((iter_train,n)),train_active_set)))) 
    
#     # graph structure does not change, only vertex features
#     #combine H and A
#     edge_matrix = np.block([[H,A.T],[A,np.zeros((np.shape(A)[0],np.shape(A)[0]))]])
    
#     # create edge_index and edge_attributes
#     edge_index = torch.tensor([])
#     edge_attr = torch.tensor([])
#     for j in range(np.shape(edge_matrix)[0]):
#         for k in range(np.shape(edge_matrix)[1]):
#             # add edge
#             if edge_matrix[j,k] != 0:
#                 edge_index = torch.cat((edge_index,torch.tensor([[j,k]])),0)
#                 edge_attr = torch.cat((edge_attr,torch.tensor([edge_matrix[j,k]])),0)
#     edge_index = edge_index.long().T
    
#     # create new vectors filled with zeros to capture vertex features better
#     f1_train = np.hstack((ftot,np.zeros(np.shape(btot))))
#     b1_train = np.hstack((np.zeros(np.shape(ftot)),btot))
#     eq1_train = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
#     node_type_train = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))

#     #print(f1_train.shape,b1_train.shape,eq1_train.shape)

#     features = np.array([f1_train, b1_train, eq1_train,node_type_train]).T
#     x_train = torch.tensor(features, dtype=torch.float32)
#     data_point = Data(x= x_train, edge_index=edge_index, edge_attr=edge_attr,y=y_train[i,:])
#     #print(data_point)
#     # list of graph elements
#     graph_train.append(data_point)


# # Initialization      
# graph_train = []
# graph_val = []
# n_vector_train = []
# m_vector_train = []
# n_vector_val = []
# m_vector_val = []

# train_acc_save = []
# val_acc_save = []
# train_loss_save = []
# val_loss_save = []
# train_prec_save = []
# val_prec_save = []
# train_rec_save = []
# val_rec_save = []
# train_f1_save = []
# val_f1_save = []
# acc_graph_train_save = []
# acc_graph_val_save = []
# val_perc_wrongly_pred_nodes_per_graph_save = []
# val_mean_wrongly_pred_nodes_per_graph_save = []
# train_perc_wrongly_pred_nodes_per_graph_save = []
# train_mean_wrongly_pred_nodes_per_graph_save = []

# # Generate QP problems and the corresponding graphs
# for i in range(len(n)):
#     n_i = n[i]
#     m_i = m[i]
#     graph_train_i, graph_val_i = generate_qp_graphs_train_val_lmpc(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)
#     graph_train = graph_train + graph_train_i
#     graph_val = graph_val + graph_val_i
#     n_vector_train = n_vector_train + [n_i for i in range(len(graph_train_i))]
#     m_vector_train= m_vector_train + [m_i for i in range(len(graph_train_i))]
#     n_vector_val = n_vector_val + [n_i for i in range(len(graph_val_i))]
#     m_vector_val = m_vector_val + [m_i for i in range(len(graph_val_i))]

# # print(graph_train[0])
# # Load Data
# train_batch_size = 64
# train_loader = DataLoader(graph_train, batch_size=train_batch_size, shuffle=True)
# val_loader = DataLoader(graph_val,batch_size = len(graph_val), shuffle = False)

# # Compute class weights for imbalanced classes
# all_labels = torch.cat([data.y for data in graph_train])
# class_weights = compute_class_weight('balanced', classes=torch.unique(all_labels).numpy(), y=all_labels.numpy())
# class_weights = torch.tensor(class_weights, dtype=torch.float32)

# # Instantiate model and optimizer
# model = GNN(input_dim=4, output_dim=1,layer_width = 128)  # Output dimension 1 for binary classification
# optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

# # Early stopping
# early_stopping = EarlyStopping(patience=5, delta=0.001)

# # Track parameters on wandb
# if track_on_wandb ==True:
#     run = wandb.init(
#         entity="ella-schmidtobreick-4283-me",
#         project="Thesis",
#         config={
#             "variables": f"{n}",
#             "constraints": f"{m}",
#             "datapoints": f"{data_points}",
#             "epochs": f"{number_of_max_epochs}",
#             "architecture": "LEConv with weights",
#             "learning_rate": f"{lr}",
#             "layer width": f"{layer_width}",
#             "number of layers": f"{number_of_layers}",
#             "threshold": f"{t}"
#         },
#     )

# # Training
# for epoch in range(number_of_max_epochs):
#     train_loss = 0
#     train_all_labels = []
#     train_preds = []
#     model.train()
#     output_train = []
#     train_all_label_graph = []
#     train_preds_graph = []
    
#     for batch in train_loader:
#         optimizer.zero_grad()
#         output = model(batch,number_of_layers)
#         output_train.extend(output.squeeze().detach().numpy().reshape(-1))
#         loss = torch.nn.BCELoss(weight=class_weights[batch.y.long()])(output.squeeze(), batch.y.float())
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#         # Convert output to binary prediction (0 or 1)
#         preds = (output.squeeze() > t).long()

#         # Store predictions and true labels
#         train_preds.extend(preds.numpy())   
#         train_all_labels.extend(batch.y.numpy())
#         # print("true labels",batch.y.shape,torch.nonzero(batch.y).squeeze())
#         # print("preds",preds.shape, torch.nonzero(preds).squeeze())
#         # print("output",output.squeeze())
#         # print("loss",loss.item())
        
#         # Save per graph predictions and labels
#         for i in range(batch.num_graphs):
#             mask = batch.batch == i
#             preds_graph = preds[mask].numpy()
#             labels_graph = batch.y[mask].numpy()

#             train_preds_graph.append(preds_graph)
#             train_all_label_graph.append(labels_graph)

#     # Compute metrics
#     train_loss /= len(train_loader)
#     train_acc = accuracy_score(train_all_labels, train_preds)
#     train_prec = precision_score(train_all_labels,train_preds)
#     train_rec = recall_score(train_all_labels, train_preds)
#     train_f1 = f1_score(train_all_labels,train_preds)
#     acc_graph_train = np.mean([np.all(pred == true) for pred, true in zip(train_preds_graph, train_all_label_graph)]) # average on graph level

#     train_num_wrongly_pred_nodes_per_graph = [np.abs(int(n_i + m_i) - np.sum(pred == label)) for pred, label, n_i, m_i in zip(train_preds_graph, train_all_label_graph, n_vector_train, m_vector_train)]
#     train_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n_i + m_i) for wrong, n_i, m_i in zip(train_num_wrongly_pred_nodes_per_graph, n_vector_train, m_vector_train)])
#     train_mean_wrongly_pred_nodes_per_graph = np.mean(train_num_wrongly_pred_nodes_per_graph)
    
    
#     # Validation step
#     model.eval()
#     val_loss = 0
#     val_mean_wrongly_pred_nodes_per_graph = 0
#     val_num_wrongly_pred_nodes_per_graph = 0
#     val_all_labels = []
#     val_preds = []
#     output_val = []
#     val_preds_graph = []
#     val_all_label_graph = []
    
#     with torch.no_grad():
#         for batch in val_loader:
#             output = model(batch,number_of_layers)
#             output_val.extend(output.squeeze().detach().numpy().reshape(-1))
#             loss = torch.nn.BCELoss()(output.squeeze(), batch.y.float())
#             val_loss += loss.item()
#             preds = (output.squeeze() > t).long()

#             # Store predictions and labels
#             val_preds.extend(preds.numpy())
#             val_all_labels.extend(batch.y.numpy())
            
#             # Store per graph predictions and labels
#             for i in range(batch.num_graphs):
#                 mask = batch.batch == i
#                 preds_graph = preds[mask].numpy()
#                 labels_graph = batch.y[mask].numpy()

#                 val_preds_graph.append(preds_graph)
#                 val_all_label_graph.append(labels_graph)                
    
#     # Compute metrics      
#     val_loss /= len(val_loader)
#     val_acc = accuracy_score(val_all_labels, val_preds)
#     val_prec = precision_score(val_all_labels,val_preds)
#     val_rec = recall_score(val_all_labels, val_preds)
#     val_f1 = f1_score(val_all_labels,val_preds)
#     acc_graph_val = np.mean([np.all(pred == true) for pred, true in zip(val_preds_graph, val_all_label_graph)]) # accuracy on graph level

#     val_num_wrongly_pred_nodes_per_graph = [np.abs(int(n_i + m_i) - np.sum(pred == label)) for pred, label, n_i, m_i in zip(val_preds_graph, val_all_label_graph, n_vector_val, m_vector_val)]
#     val_perc_wrongly_pred_nodes_per_graph = np.mean([wrong / (n_i + m_i) for wrong, n_i, m_i in zip(val_num_wrongly_pred_nodes_per_graph, n_vector_val, m_vector_val)])
#     val_mean_wrongly_pred_nodes_per_graph = np.mean(val_num_wrongly_pred_nodes_per_graph)
    
#     # Log metrics to wandb.
#     if track_on_wandb == True:
#         run.log({"acc_train": train_acc,"acc_val": val_acc,"loss_train": train_loss, "loss_val": val_loss, "prec_train": train_prec, "prec_val":val_prec, "rec_train": train_rec, "rec_val": val_rec, "f1_train": train_f1,"f1_val":val_f1, "acc_graph_train": acc_graph_train, "acc_graph_val": acc_graph_val,"perc_wrong_pred_nodes_per_graph_val": val_perc_wrongly_pred_nodes_per_graph,"num_wrong_pred_nodes_per_graph_val":val_mean_wrongly_pred_nodes_per_graph, "threshold": t})

#     # Save metrics
#     train_acc_save.append(train_acc)
#     val_acc_save.append(val_acc)
#     train_loss_save.append(train_loss)
#     val_loss_save.append(val_loss)
#     train_prec_save.append(train_prec)
#     val_prec_save.append(val_prec)
#     train_rec_save.append(train_rec)
#     val_rec_save.append(val_rec)
#     train_f1_save.append(train_f1)
#     val_f1_save.append(val_f1)
#     acc_graph_train_save.append(acc_graph_train)
#     acc_graph_val_save.append(acc_graph_val)
#     val_perc_wrongly_pred_nodes_per_graph_save.append(val_perc_wrongly_pred_nodes_per_graph)
#     val_mean_wrongly_pred_nodes_per_graph_save.append(val_mean_wrongly_pred_nodes_per_graph)
#     train_perc_wrongly_pred_nodes_per_graph_save.append(train_perc_wrongly_pred_nodes_per_graph)
#     train_mean_wrongly_pred_nodes_per_graph_save.append(train_mean_wrongly_pred_nodes_per_graph)

#     # Early stopping
#     early_stopping(val_loss, model,epoch)
#     if early_stopping.early_stop:
#         print(f"Early stopping after {epoch} epochs.")
#         break

# # Save best model
# best_epoch = early_stopping.load_best_model(model)
# torch.save(model.state_dict(), f"saved_models/{modelname}.pth")

# # Finish the run and upload any remaining data.
# if track_on_wandb == True:
#     run.finish()

# # Print metrics
# print("TRAINING")
# print(f"Accuracy (node level) of the final model: {train_acc_save[best_epoch-1]}")
# print(f"Precision of the model on the test data: {train_prec_save[best_epoch-1]}")
# print(f"Recall of the model on the test data: {train_rec_save[best_epoch-1]}")
# print(f"F1-Score of the model on the test data: {train_f1_save[best_epoch-1]}")
# print(f"Accuracy (graph level) of the model on the test data: {acc_graph_train_save[best_epoch-1]}")
# print(f"Perc num_wrongly_pred_nodes_per_graph: {train_perc_wrongly_pred_nodes_per_graph_save[best_epoch-1]}")

# print("VALIDATION")
# print(f"Accuracy (node level) of the final model: {val_acc_save[best_epoch-1]}")
# print(f"Precision of the model on the test data: {val_prec_save[best_epoch-1]}")
# print(f"Recall of the model on the test data: {val_rec_save[best_epoch-1]}")
# print(f"F1-Score of the model on the test data: {val_f1_save[best_epoch-1]}")
# print(f"Accuracy (graph level) of the model on the test data: {acc_graph_val_save[best_epoch-1]}")
# print(f"Perc num_wrongly_pred_nodes_per_graph: {val_perc_wrongly_pred_nodes_per_graph_save[best_epoch-1]}")
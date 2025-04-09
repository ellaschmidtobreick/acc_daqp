import numpy as np
import daqp
import numpy as np
from ctypes import * 

from generate_mpqp_v2 import generate_qp

import torch
from torch_geometric.data import Data


# n - number of variables
# m - number of constraints
# nth - dimension of theta
# seed?

#default train - test - val split: 80%, 10%, 10%

def generate_qp_graphs_train_val(n,m,nth,seed,number_of_graphs):

    #spit generated problems into train, test, val
    iter_train = int(np.rint(0.8*number_of_graphs))
    iter_val = int(np.rint(0.1*number_of_graphs))
    
    np.random.seed(seed)
    H,f,F,A,b,B = generate_qp(n,m,seed)
    print(H.shape,f.shape,F.shape,A.shape,b.shape,B.shape)
    np.savez("generated_qp_data.npz", H=H, f=f, F=F, A=A, b=b, B=B)
    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])

    # Generate training set - only change theta
    x_train = np.zeros((iter_train,n))
    lambda_train = np.zeros((iter_train,m))
    train_iterations = np.zeros((iter_train))
    train_time= np.zeros((iter_train))
    f_train = np.zeros((iter_train,n))
    b_train = np.zeros((iter_train,m))
    theta_train = np.zeros((iter_train,nth))
    for i in range(iter_train):
        theta = np.random.randn(nth)
        theta_train[i,:]=theta
        btot = b + B @ theta
        ftot = f + F @ theta
        b_train[i,:]= btot
        f_train[i,:]= ftot
        
        x,fval,exitflag,info = daqp.solve(H,ftot,A,btot,blower,sense)
        x_train[i,:]= x
        lambda_train[i,:]= list(info.values())[4]
        train_iterations[i] = list(info.values())[2]
        train_time[i]= list(info.values())[0]

    # Generate val set
    np.random.seed(seed+1)
    x_val = np.zeros((iter_val,n))
    lambda_val = np.zeros((iter_val,m))
    val_iterations = np.zeros((iter_val))
    val_time = np.zeros((iter_val))
    f_val = np.zeros((iter_val,n))
    b_val = np.zeros((iter_val,m))
    for i in range(iter_val):
        theta = np.random.randn(nth)
        btot = b + B @ theta
        ftot = f + F @ theta
        b_val[i,:]=btot
        f_val[i,:]=ftot
        x,fval,exitflag,info = daqp.solve(H,ftot,A,btot,blower,sense)
        x_val[i,:]= x
        lambda_val[i,:]= list(info.values())[4]
        val_iterations[i] = list(info.values())[2]
        val_time[i] = list(info.values())[0]
        
    # get optimal active set (y)
    train_active_set = (lambda_train != 0).astype(int)
    y_train = torch.tensor((np.hstack((np.zeros((iter_train,n)),train_active_set)))) 
    val_active_set = (lambda_val != 0).astype(int)
    y_val = torch.tensor((np.hstack((np.zeros((iter_val,n)),val_active_set))))
    
    
    # Generate the graph from the training data
    graph_train = []
    graph_val = []


    # graph structure does not change, only vertex features
    #combine H and A
    edge_matrix = np.block([[H,A.T],[A,np.zeros((np.shape(A)[0],np.shape(A)[0]))]])
    #print("edge matrix shape",edge_matrix.shape)

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
    f1_train = np.hstack((f_train,np.zeros(np.shape(b_train))))
    b1_train = np.hstack((np.zeros(np.shape(f_train)),b_train))
    eq1_train = np.hstack((np.zeros(np.shape(f_train)),(np.zeros(np.shape(b_train)))))
    #print(f1_train.shape,b1_train.shape,eq1_train.shape)

    # create matrix with vertex features
    x_train = torch.tensor([])
    for i in range(iter_train):
        features = np.array([f1_train[i], b1_train[i], eq1_train[i]]).T
        x_train = torch.tensor(features, dtype=torch.float32)
        #x_train = torch.tensor([f1_train[i],b1_train[i], eq1_train[i]]).T
        data_point = Data(x= x_train, edge_index=edge_index, edge_attr=edge_attr,y=y_train[i,:])
        #print(data_point)
        # list of graph elements
        graph_train.append(data_point)

    f1_val = np.hstack((f_val,np.zeros(np.shape(b_val))))
    b1_val = np.hstack((np.zeros(np.shape(f_val)),b_val))
    eq1_val = np.hstack((np.zeros(np.shape(f_val)),(np.zeros(np.shape(b_val)))))
    #print(f1_val.shape,b1_val.shape,eq1_val.shape)

    # val graph
    x_val = torch.tensor([])
    for i in range(iter_val):
        features = np.array([f1_val[i], b1_val[i], eq1_val[i]]).T
        x_val = torch.tensor(features, dtype=torch.float32)
        #x_val = torch.tensor(np.array([f1_val[i],b1_val[i], eq1_val[i]])).T
        data_point = Data(x= x_val, edge_index=edge_index, edge_attr=edge_attr,y=y_val[i,:])
        # list of graph elements
        graph_val.append(data_point)
        
        
    return graph_train, graph_val, H, A

def generate_qp_graphs_test_data_only(n,m,nth,seed,number_of_graphs):
    np.random.seed(seed)
    #spit generated problems into train, test, val
    iter_test = int(np.rint(0.1*number_of_graphs))
    data = np.load("generated_qp_data.npz", allow_pickle=True)
    
    if n == data["H"].shape[0] and m == data["A"].shape[0]:
        H = data["H"]
        f = data["f"]
        F = data["F"]
        A = data["A"]
        b = data["b"]
        B = data["B"]
    else:
        H,f,F,A,b,B = generate_qp(n,m,seed)
        print(H.shape, f.shape,F.shape,A.shape,b.shape,B.shape)
    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])

    # Generate test set
    np.random.seed(seed+2)
    x_test = np.zeros((iter_test,n))
    lambda_test = np.zeros((iter_test,m))
    test_iterations = np.zeros((iter_test))
    test_time = np.zeros((iter_test))
    f_test = np.zeros((iter_test,n))
    b_test = np.zeros((iter_test,m))
    for i in range(iter_test):
        theta = np.random.randn(nth)
        btot = b + B @ theta
        ftot = f + F @ theta
        b_test[i,:]=btot
        f_test[i,:]=ftot
        x,_,_,info = daqp.solve(H,ftot,A,btot,blower,sense)
        x_test[i,:]= x
        lambda_test[i,:]= list(info.values())[4]
        test_iterations[i] = list(info.values())[2]
        test_time[i] = list(info.values())[0]
        
    # get optimal active set (y)
    test_active_set = (lambda_test != 0).astype(int)
    y_test = torch.tensor((np.hstack((np.zeros((iter_test,n)),test_active_set)))) 
    
    
    # Generate the graph from the training data
    graph_test = []


    # graph structure does not change, only vertex features
    #combine H and A
    edge_matrix = np.block([[H,A.T],[A,np.zeros((np.shape(A)[0],np.shape(A)[0]))]])
    #print("edge matrix shape",edge_matrix.shape)

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
        
    f1_test = np.hstack((f_test,np.zeros(np.shape(b_test))))
    b1_test = np.hstack((np.zeros(np.shape(f_test)),b_test))
    eq1_test = np.hstack((np.zeros(np.shape(f_test)),(np.zeros(np.shape(b_test)))))
    #print(f1_test.shape,b1_test.shape,eq1_test.shape)

    # test graph
    x_test = torch.tensor([])
    for i in range(iter_test):
        features = np.array([f1_test[i], b1_test[i], eq1_test[i]]).T
        x_test = torch.tensor(features, dtype=torch.float32)
        #x_test = torch.tensor(np.array([f1_test[i],b1_test[i], eq1_test[i]])).T
        data_point = Data(x= x_test, edge_index=edge_index, edge_attr=edge_attr,y=y_test[i,:])
        # list of graph elements
        graph_test.append(data_point)
        
    return graph_test, test_iterations,test_time, H,f_test,A,b_test,blower,sense



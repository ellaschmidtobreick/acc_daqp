import numpy as np
import daqp
from ctypes import * 
import os
from generate_mpqp_v2 import generate_qp

import torch
from torch_geometric.data import Data


# n - number of variables
# m - number of constraints
# nth - dimension of theta

#default train - test - val split: 80%, 10%, 10%

# generate train and val data
def generate_qp_graphs_train_val(n,m,nth,seed,number_of_graphs, H_flexible=False, A_flexible = False):

    #spit generated problems into train, test, val
    iter_train = int(np.rint(0.8*number_of_graphs))
    iter_val = int(np.rint(0.1*number_of_graphs))
    
    np.random.seed(seed)
    H,f,F,A,b,B,T = generate_qp(n,m,seed)
    print(H.shape,f.shape,F.shape,A.shape,b.shape,B.shape,T.shape)
    np.savez(f"data/generated_qp_data_{n}v_{m}c.npz", H=H, f=f, F=F, A=A, b=b, B=B,T=T)
    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])
    
    # Generate training set - only change theta and A
    x_train = np.zeros((iter_train,n))
    lambda_train = np.zeros((iter_train,m))
    train_iterations = np.zeros((iter_train))
    train_time= np.zeros((iter_train))
    
    # Generate the graph from the training data
    graph_train = []
    
    for i in range(iter_train):
        theta = np.random.randn(nth)
        
        if A_flexible == True:
            A = np.random.randn(m,n)
            B = A @ (-T)
            
        btot = b + B @ theta
        ftot = f + F @ theta
        
        if H_flexible == True:
            M = np.random.randn(n,n)
            H = M @ M.T 

        _,_,_,info = daqp.solve(H,ftot,A,btot,blower,sense)
        lambda_train[i,:]= list(info.values())[4]
        train_iterations[i] = list(info.values())[2]
        train_time[i]= list(info.values())[0]
    
        # get optimal active set (y)
        train_active_set = (lambda_train != 0).astype(int)
        y_train = torch.tensor((np.hstack((np.zeros((iter_train,n)),train_active_set)))) 
        
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
        f1_train = np.hstack((ftot,np.zeros(np.shape(btot))))
        b1_train = np.hstack((np.zeros(np.shape(ftot)),btot))
        eq1_train = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
        node_type_train = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))

        #print(f1_train.shape,b1_train.shape,eq1_train.shape)

        features = np.array([f1_train, b1_train, eq1_train,node_type_train]).T
        x_train = torch.tensor(features, dtype=torch.float32)
        data_point = Data(x= x_train, edge_index=edge_index, edge_attr=edge_attr,y=y_train[i,:])
        #print(data_point)
        # list of graph elements
        graph_train.append(data_point)

    # Generate val set
    np.random.seed(seed+1)
    x_val = np.zeros((iter_val,n))
    lambda_val = np.zeros((iter_val,m))
    val_iterations = np.zeros((iter_val))
    val_time = np.zeros((iter_val))
    
    graph_val = []
    
    for i in range(iter_val):
        theta = np.random.randn(nth)
        
        if A_flexible == True:
            A = np.random.randn(m,n)
            B = A @ (-T)
            
        btot = b + B @ theta
        ftot = f + F @ theta
        
        if H_flexible == True:
            M = np.random.randn(n,n)
            H = M @ M.T 
        
        _,_,_,info = daqp.solve(H,ftot,A,btot,blower,sense)
        lambda_val[i,:]= list(info.values())[4]
        val_iterations[i] = list(info.values())[2]
        val_time[i] = list(info.values())[0]
    

        val_active_set = (lambda_val != 0).astype(int)
        y_val = torch.tensor((np.hstack((np.zeros((iter_val,n)),val_active_set))))
        
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

        f1_val = np.hstack((ftot,np.zeros(np.shape(btot))))
        b1_val = np.hstack((np.zeros(np.shape(ftot)),btot))
        eq1_val = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
        node_type_val = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))

        # val graph
        x_val = torch.tensor([])
        features = np.array([f1_val, b1_val, eq1_val,node_type_val]).T
        x_val = torch.tensor(features, dtype=torch.float32)
        data_point = Data(x= x_val, edge_index=edge_index, edge_attr=edge_attr,y=y_val[i,:],index = i)
        # list of graph elements
        graph_val.append(data_point)
        
        
    return graph_train, graph_val

def generate_qp_graphs_test_data_only(n,m,nth,seed,number_of_graphs,H_flexible = False,A_flexible = False):
    np.random.seed(seed)
    #spit generated problems into train, test, val
    iter_test = int(np.rint(0.1*number_of_graphs))

    file_path = f"data/generated_qp_data_{n}v_{m}c_flex_H.npz"
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        H = data["H"]
        f = data["f"]
        F = data["F"]
        A = data["A"]
        b = data["b"]
        B = data["B"]
        T = data["T"]
    else:
        H,f,F,A,b,B,T = generate_qp(n,m,seed)
        print(H.shape, f.shape,F.shape,A.shape,b.shape,B.shape)
    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])

    # Generate test set
    np.random.seed(seed+2)
    x_test = np.zeros((iter_test,n))
    lambda_test = np.zeros((iter_test,m))
    test_iterations = []
    test_time = []
    f_test = []
    b_test = []
    H_test = []
    A_test = []
    blower_test = []
    sense_test = []
    # Generate the graph from the training data
    graph_test = []
    
    for i in range(iter_test):
        theta = np.random.randn(nth)
        
        if A_flexible == True:
            A = np.random.randn(m,n)
            B = A @ (-T)
            
        btot = b + B @ theta
        ftot = f + F @ theta
        
        if H_flexible == True:
            M = np.random.randn(n,n)
            H = M @ M.T
        
        b_test.append(btot)
        f_test.append(ftot)
        H_test.append(H)
        A_test.append(A)
        blower_test.append(blower)
        sense_test.append(sense)
        
        _,_,_,info = daqp.solve(H,ftot,A,btot,blower,sense)
        lambda_test[i,:]= list(info.values())[4]
        test_iterations.append(list(info.values())[2])
        test_time.append(list(info.values())[0])
        
        # get optimal active set (y)
        test_active_set = (lambda_test != 0).astype(int)
        y_test = torch.tensor((np.hstack((np.zeros((iter_test,n)),test_active_set)))) 
        
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
            
        f1_test = np.hstack((ftot,np.zeros(np.shape(btot))))
        b1_test = np.hstack((np.zeros(np.shape(ftot)),btot))
        eq1_test = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
        node_type = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))

        # test graph
        features = np.array([f1_test, b1_test, eq1_test,node_type]).T
        x_test = torch.tensor(features, dtype=torch.float32)
        data_point = Data(x=x_test, edge_index=edge_index, edge_attr=edge_attr,y=y_test[i,:])
        # list of graph elements
        graph_test.append(data_point)
        
    return graph_test, test_iterations,test_time, H_test,f_test,A_test,b_test,blower_test,sense_test,n,m

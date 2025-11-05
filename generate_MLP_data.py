import numpy as np
import daqp
from ctypes import * 
from generate_mpqp_v2 import generate_qp, generate_banded_qp, generate_sparse_qp
import torch
from torch.utils.data import TensorDataset
import os
      
               
def generate_qp_MLP_train_val(n,m,nth,seed,number_of_graphs, H_flexible=False, A_flexible = False,dataset_type="standard"):

    #spit generated problems into train, test, val
    iter_train = int(np.rint(0.8*number_of_graphs))
    iter_val = int(np.rint(0.1*number_of_graphs))
    
    np.random.seed(seed)

    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])

    if dataset_type == "standard":
        # Generate general matrices
        H,f,F,A,b,B,T = generate_qp(n,m,seed,nth) #generate_banded_qp(n, m, seed, bandwidth=10, nth = nth)# generate_qp_block_sparse(n, m, num_blocks=4, inter_block_prob=0.05, given_seed=seed, nth=nth)#generate_sparse_qp(n, m, seed, density=0.1, nth=nth)# #generate_qp(n,m,seed,nth)
        np.savez(f"data/generated_MLP_data_{n}v_{m}c.npz", H=H, f=f, F=F, A=A, b=b, B=B,T=T)
    elif dataset_type == "lmpc":
        # Load given lmpc data
        data = np.load(f'data/mpc_mpqp_N{n}.npz')
        H,f,F,A,b,B = data["H"], data["f"], data["f_theta"], data["A"], data["b"], data["W"]
        # dimension reduction for daqp solver
        f = f.squeeze()
        b = b.squeeze()


    # Generate training data
    x_train = np.zeros((iter_train,n*n+m*n+n+m))
    y_train = np.zeros((iter_train,n+m))
    
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
        lambda_train= list(info.values())[4]
    
        # save data
        x_train[i,:] = torch.cat([torch.tensor(H.flatten(), dtype=torch.float32),torch.tensor(A.flatten(), dtype=torch.float32),torch.tensor(ftot, dtype=torch.float32),torch.tensor(btot, dtype=torch.float32)], dim=0)
        #y_train[i,:] = (lambda_train != 0).astype(int)
        train_active_set = (lambda_train != 0).astype(int)
        y_train[i,:] = torch.tensor((np.hstack((np.zeros((n)),train_active_set)))) 

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    QP_train = TensorDataset(x_train, y_train)
        
    # Generate val set
    np.random.seed(seed+1)
    x_val = np.zeros((iter_val,n*n+m*n+n+m))
    y_val = np.zeros((iter_val,n+m))
    
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
        lambda_val= list(info.values())[4]
    
        # save data
        x_val[i,:] = torch.cat([torch.tensor(H.flatten(), dtype=torch.float32),torch.tensor(A.flatten(), dtype=torch.float32),torch.tensor(ftot, dtype=torch.float32),torch.tensor(btot, dtype=torch.float32)], dim=0)
        #y_val[i,:] = (lambda_val != 0).astype(int)
        val_active_set = (lambda_val != 0).astype(int)
        y_val[i,:] = torch.tensor((np.hstack((np.zeros((n)),val_active_set)))) 
        #print(y_val[i,:])
        
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    QP_val = TensorDataset(x_val, y_val)
    return QP_train, QP_val

def generate_MLP_test_data_only(n,m,nth,seed,number_of_graphs,H_flexible = False,A_flexible = False,dataset_type="standard"):
    np.random.seed(seed)
    #spit generated problems into train, test, val
    iter_test = int(np.rint(0.1*number_of_graphs))

    if dataset_type == "standard":
        file_path = f"data/generated_MLP_data_{n}v_{m}c.npz"
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
            H,f,F,A,b,B,T = generate_qp(n,m,seed,nth) #generate_banded_qp(n, m, seed, bandwidth=10, nth = nth)# generate_qp_block_sparse(n, m, num_blocks=4, inter_block_prob=0.05, given_seed=seed, nth=nth)# generate_sparse_qp(n, m, seed, density=0.1, nth=nth)#generate_qp(n,m,seed,nth)
            print(H.shape, f.shape,F.shape,A.shape,b.shape,B.shape)
    elif dataset_type == "lmpc":
        # Load given lmpc data
        data = np.load(f'data/mpc_mpqp_N{n}.npz')
        H,f,F,A,b,B = data["H"], data["f"], data["f_theta"], data["A"], data["b"], data["W"]
        # dimension reduction for daqp solver
        f = f.squeeze()
        b = b.squeeze()
        
    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])

    # Generate test set
    np.random.seed(seed+2)
    x_test = np.zeros((iter_test,n))
    #lambda_test = np.zeros((iter_test,m))
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
    
    # Generate test set
    x_test = np.zeros((iter_test,n*n+m*n+n+m))
    y_test = np.zeros((iter_test,n+m))
    
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
        lambda_test= list(info.values())[4]
        test_iterations.append(list(info.values())[2])
        test_time.append(list(info.values())[0])
                
        # save data
        x_test[i,:] = torch.cat([torch.tensor(H.flatten(), dtype=torch.float32),torch.tensor(A.flatten(), dtype=torch.float32),torch.tensor(ftot, dtype=torch.float32),torch.tensor(btot, dtype=torch.float32)], dim=0)
        test_active_set = (lambda_test != 0).astype(int)
        y_test[i,:] = torch.tensor((np.hstack((np.zeros((n)),test_active_set)))) 
        #print(y_test[i,:])
        #y_test[i,:] = (lambda_test[i,:] != 0).astype(int)
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    QP_test = TensorDataset(x_test, y_test)
    return QP_test, test_iterations,test_time, H_test,f_test,A_test,b_test,blower,_,n,m
        
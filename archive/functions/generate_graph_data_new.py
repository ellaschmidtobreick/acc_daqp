import numpy as np
import daqp
from ctypes import * 
import os
from generate_mpqp_v2 import generate_qp
from collections import Counter
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
    np.savez(f"data/generated_qp_data_{n}v_{m}c.npz", H=H, f=f, F=F, A=A, b=b, B=B)
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
    
    A_train = [0 for i in range(iter_train)] # only change A
    #H_train = [0 for i in range(iter_train)] # only change H

    for i in range(iter_train):
        theta = np.random.randn(nth)
        theta_train[i,:]=theta
        
        A = np.random.randn(m,n)  # only change A
        A_train[i] = A # only change A
        T = np.random.randn(n,nth) # A transformation such that x = T*th is primal feasible
        B = A @ (-T)
        
        btot = b + B @ theta
        ftot = f + F @ theta
        b_train[i,:]= btot
        f_train[i,:]= ftot
        
        # M = np.random.randn(n,n)
        # H = M @ M.T 
        # H_train[i] = H # only change H
        
        x,fval,exitflag,info = daqp.solve(H,ftot,A,btot,blower,sense)
        x_train[i,:]= x
        lambda_train[i,:]= list(info.values())[4]
        train_iterations[i] = list(info.values())[2]
        train_time[i]= list(info.values())[0] + list(info.values())[1]

    # Generate val set
    np.random.seed(seed+1)
    x_val = np.zeros((iter_val,n))
    lambda_val = np.zeros((iter_val,m))
    val_iterations = np.zeros((iter_val))
    val_time = np.zeros((iter_val))
    f_val = np.zeros((iter_val,n))
    b_val = np.zeros((iter_val,m))
    
    A_val = [0 for i in range(iter_val)] # only change A
    #H_val = [0 for i in range(iter_val)] # only change H

    for i in range(iter_val):
        theta = np.random.randn(nth)
        
        A = np.random.randn(m,n)  # only change A
        A_val[i] = A # only change A
        T = np.random.randn(n,nth) # A transformation such that x = T*th is primal feasible
        B = A @ (-T)
        
        btot = b + B @ theta
        ftot = f + F @ theta
        b_val[i,:]=btot
        f_val[i,:]=ftot
        
        # M = np.random.randn(n,n)
        # H = M @ M.T 
        # H_val[i] = H # only change H
        
        x,fval,exitflag,info = daqp.solve(H,ftot,A,btot,blower,sense)
        x_val[i,:]= x
        lambda_val[i,:]= list(info.values())[4]
        val_iterations[i] = list(info.values())[2]
        val_time[i] = list(info.values())[0] + list(info.values())[1]
        
    # get optimal active set (y)
    train_active_set = (lambda_train != 0).astype(int)
    y_train = torch.tensor((np.hstack((np.zeros((iter_train,n)),train_active_set)))) 
    val_active_set = (lambda_val != 0).astype(int)
    y_val = torch.tensor((np.hstack((np.zeros((iter_val,n)),val_active_set))))
    
    
    # Generate the graph from the training data
    graph_train = []
    graph_val = []

    # use if identical H and A for all data points
    # # graph structure does not change, only vertex features
    # #combine H and A
    # edge_matrix = np.block([[H,A.T],[A,np.zeros((np.shape(A)[0],np.shape(A)[0]))]])
    # #print("edge matrix shape",edge_matrix.shape)

    # # create edge_index and edge_attributes
    # edge_index = torch.tensor([])
    # edge_attr = torch.tensor([])
    # for j in range(np.shape(edge_matrix)[0]):
    #     for k in range(np.shape(edge_matrix)[1]):
    #         # add edge
    #         if edge_matrix[j,k] != 0:
    #             edge_index = torch.cat((edge_index,torch.tensor([[j,k]])),0)
    #             edge_attr = torch.cat((edge_attr,torch.tensor([edge_matrix[j,k]])),0)
    # edge_index = edge_index.long().T

    # create new vectors filled with zeros to capture vertex features better
    f1_train = np.hstack((f_train,np.zeros(np.shape(b_train))))
    b1_train = np.hstack((np.zeros(np.shape(f_train)),b_train))
    eq1_train = np.hstack((np.zeros(np.shape(f_train)),(np.zeros(np.shape(b_train)))))
    node_type_train = np.hstack((np.zeros(np.shape(f_train)),(np.ones(np.shape(b_train)))))
    #print(f1_train.shape,b1_train.shape,eq1_train.shape)

    # create matrix with vertex features
    x_train = torch.tensor([])
    for i in range(iter_train):
        # graph structure does not change, only vertex features
        #combine H and A
        A = A_train[i]
        #H = H_train[i]
        
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
            
        features = np.array([f1_train[i], b1_train[i], eq1_train[i],node_type_train[i]]).T
        x_train = torch.tensor(features, dtype=torch.float32)
        #x_train = torch.tensor([f1_train[i],b1_train[i], eq1_train[i]]).T
        data_point = Data(x= x_train, edge_index=edge_index, edge_attr=edge_attr,y=y_train[i,:])
        #print(data_point)
        # list of graph elements
        graph_train.append(data_point)

    f1_val = np.hstack((f_val,np.zeros(np.shape(b_val))))
    b1_val = np.hstack((np.zeros(np.shape(f_val)),b_val))
    eq1_val = np.hstack((np.zeros(np.shape(f_val)),(np.zeros(np.shape(b_val)))))
    node_type_val = np.hstack((np.zeros(np.shape(f_val)),(np.ones(np.shape(b_val)))))
    #print(f1_val.shape,b1_val.shape,eq1_val.shape)

    # val graph
    x_val = torch.tensor([])
    for i in range(iter_val):
                # graph structure does not change, only vertex features
        #combine H and A
        A = A_val[i]
        #H = H_val[i]
        
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
            
        features = np.array([f1_val[i], b1_val[i], eq1_val[i],node_type_val[i]]).T
        x_val = torch.tensor(features, dtype=torch.float32)
        #x_val = torch.tensor(np.array([f1_val[i],b1_val[i], eq1_val[i]])).T
        data_point = Data(x= x_val, edge_index=edge_index, edge_attr=edge_attr,y=y_val[i,:])
        # list of graph elements
        graph_val.append(data_point)
        
    return graph_train, graph_val

# function to create data sets with variable sized problems
def generate_qp_graphs_different_sizes(n_min,n_max,m_min,m_max,nth,seed,number_of_graphs,mode,H= None,A = None):

    # mode: train / test / val - detects size and seed of to generate dataset
    print(f"{mode} data")
    if mode == "train":
        iter = int(np.rint(0.8*number_of_graphs))
        np.random.seed(seed)
    elif mode =="val":
        iter = int(np.rint(0.1*number_of_graphs))
        np.random.seed(seed+1)
    elif mode == "test":
        iter = int(np.rint(0.1*number_of_graphs))
        np.random.seed(seed+2)
    else:
        print("No suitable mode was given as input")
        return [], []
    print(f"H: {H}")
    print(f"A: {A}")
    # sample n,m for different sized graphs
    graph = []
    n_vector = np.random.randint(n_min,n_max+1,size = iter)
    m_vector = np.random.randint(m_min,m_max+1, size = iter)
    theta_vector = np.random.randn(nth,iter)
    
    # generate variables only dependent on n (M,f,F,T)
    n_unique, n_counts = np.unique(n_vector, return_counts=True)
    n_dict = dict(zip(n_unique, n_counts))
    #print(n_unique, n_counts)
    M_array = [0 for i in range(iter)]
    f_array = [0 for i in range(iter)]
    F_array = [0 for i in range(iter)]
    T_array = [0 for i in range(iter)]
    
    print(n_dict)
    
    # sample for each n 
    for n in n_dict:
        n_count = n_dict[n]
        #print(n,n_count)
        indices, = np.where(n_vector == n)
        #print(indices)
        M = np.random.randn(n,n,n_count)
        f = np.random.randn(n)
        F = np.random.randn(n,nth)
        T = np.random.randn(n,nth)   #transformation matrix

        # save the sampled variables
        for position, i in enumerate(indices):
            M_array[i] = M[:,:,position]
            f_array[i] = f
            F_array[i] = F
            T_array[i] = T

    # generate variables only dependent on m (b,sense, blower)
    m_unique, m_counts = np.unique(m_vector, return_counts=True)
    m_dict = dict(zip(m_unique, m_counts))
    b_array = [0 for i in range(iter)]
    sense_array = [0 for i in range(iter)]
    blower_array = [0 for i in range(iter)]
    
    # sample for each m
    for m in m_dict:
        indices, = np.where(m_vector == m)
        b = np.random.rand(m)
        #save variables
        for i in indices:
            b_array[i] = b
            sense_array[i] = np.zeros(m, dtype=np.int32)
            blower_array[i] = np.array([-np.inf for i in range(m)])
        
    # generate variables dependent on n and m (A)
    A_array = [0 for i in range(iter)]     
    A_dict = {}
    # pair the combinations of n and m
    pairs = Counter(list(zip(n_vector, m_vector)))       # returns dict with pairs as key and count as value
    for (n_val, m_val), count in pairs.items():       
        A_dict[(n_val, m_val)] = np.random.randn(m_val,n_val,count)

    # save values in A_array
    pair_usage_counter = {key: 0 for key in pairs} # track how many times each pair has been assigned to A_array to find correct place to save it
    for i in range(iter):
        n = n_vector[i]
        m = m_vector[i]
        number_pair_used = pair_usage_counter[(n, m)]
        A_array[i] = A_dict[(n, m)][:,:,number_pair_used]
        pair_usage_counter[(n, m)] += 1

    print("All QPs generated - now transfer problems into graphs")

    # express problem as graph
    for i in range(iter):
        # generate H, btot, ftot
        if A is None:
            A = A_array[i]
        if H is None:
            H = M_array[i] @ M_array[i].T
        B = A @ (-T_array[i])
        btot = b_array[i] + B @ theta_vector[:,i]
        ftot = f_array[i] + F_array[i] @ theta_vector[:,i]

        # solve problem to get the active set 
        _,_,exitflag,info = daqp.solve(H,ftot,A,btot,blower_array[i],sense_array[i])
        lambda_dual= list(info.values())[4]
        active_set = (lambda_dual != 0).astype(int)
        y = torch.tensor((np.hstack((np.zeros(n_vector[i]),active_set)))) 
        
        # create edge matrix
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
        f1 = np.hstack((ftot,np.zeros(np.shape(btot))))
        b1= np.hstack((np.zeros(np.shape(ftot)),btot))
        eq1 = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
        node_type = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))
        # create matrix with vertex features
        x = torch.tensor([])
        features = np.array([f1, b1, eq1, node_type]).T
        x= torch.tensor(features, dtype=torch.float32)
        data_point = Data(x = x, edge_index=edge_index, edge_attr=edge_attr,y=y)
        
        # save graph
        graph.append(data_point)
    
    print("All graphs saved")   
    return graph,n_vector, m_vector

def generate_qp_graphs_test_data_only(n,m,nth,seed,number_of_graphs):
    np.random.seed(seed)
    #spit generated problems into train, test, val
    iter_test = int(np.rint(0.1*number_of_graphs))
    file_path = f"data/generated_qp_data_{n}v_{m}c.npz"
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        H = data["H"]
        f = data["f"]
        F = data["F"]
        A = data["A"]
        b = data["b"]
        B = data["B"]
        
    else:
        H,f,F,A,b,B = generate_qp(n,m,seed)
        np.savez(f"data/generated_qp_data_{n}v_{m}c.npz", H=H, f=f, F=F, A=A, b=b, B=B)
        print(H.shape, f.shape,F.shape,A.shape,b.shape,B.shape)
    sense = np.zeros(m, dtype=np.int32)
    blower = np.array([-np.inf for i in range(m)])

    # Generate test set
    np.random.seed(seed+2)
    x_test = np.zeros((iter_test,n))
    lambda_test = np.zeros((iter_test,m))
    test_iterations = np.zeros((iter_test))
    test_time = np.zeros((iter_test))
    test_time_solve = np.zeros((iter_test))
    test_time_setup = np.zeros((iter_test))
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
        test_time_solve[i] = list(info.values())[0]
        test_time_setup = list(info.values())[1]
        
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
    node_type = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))
    node_type = np.tile(node_type, (500, 1))
    #print(f1_test.shape,b1_test.shape,eq1_test.shape)

    # test graph
    x_test = torch.tensor([])
    for i in range(iter_test):
        features = np.array([f1_test[i], b1_test[i], eq1_test[i],node_type[i]]).T
        x_test = torch.tensor(features, dtype=torch.float32)
        #x_test = torch.tensor(np.array([f1_test[i],b1_test[i], eq1_test[i]])).T
        data_point = Data(x= x_test, edge_index=edge_index, edge_attr=edge_attr,y=y_test[i,:])
        # list of graph elements
        graph_test.append(data_point)
    
    test_time = test_time_setup + test_time_solve
    #test_time_solve,test_time_setup,
    return graph_test, test_iterations, test_time,H,f_test,A,b_test,blower,sense


##########################################
# Generate data with flexible size (newest version)
def generate_qp_graphs_different_sizes(n_min,n_max,m_min,m_max,nth,seed,number_of_graphs,mode,H= None,A = None):

    # mode: train / test / val - detects size and seed of to generate dataset
    print(f"{mode} data")
    # if mode == "train":
    iter = int(np.rint(0.8*number_of_graphs))
    current_seed = seed
    np.random.seed(seed)
    print(seed)
    # elif mode =="val":
    #     iter = int(np.rint(0.1*number_of_graphs))
    #     current_seed = seed
    #     np.random.seed(current_seed)
    #     print(seed+1)
    # elif mode == "test":
    #     iter = int(np.rint(0.1*number_of_graphs))
    #     current_seed = seed+2
    #     np.random.seed(current_seed)
    # else:
    #     print("No suitable mode was given as input")
    #     return [], []
    
    if H is not None:
        print(f"H: {H}")
        H_given =True
    else:
        H_given = False
        
    if A is not None:
        print(f"A: {A}")
        A_given =True
    else:
        A_given = False    
        
    # sample n,m for different sized graphs
    graph = []
    n_vector = np.random.randint(n_min,n_max+1,size = iter)
    m_vector = np.random.randint(m_min,m_max+1, size = iter)
    theta_vector = np.random.randn(nth,iter)
    
    # generate variables only dependent on n (M,f,F,T)
    n_unique, n_counts = np.unique(n_vector, return_counts=True)
    n_dict = dict(zip(n_unique, n_counts))
    #print(n_unique, n_counts)
    if H_given== False:
        M_array = [0 for i in range(iter)]
    f_array = [0 for i in range(iter)]
    F_array = [0 for i in range(iter)]
    T_array = [0 for i in range(iter)]
    
    print(n_dict)
    
    # sample for each n 
    for n in n_dict:
        n_count = n_dict[n]
        #print(n,n_count)
        indices, = np.where(n_vector == n)
        #print(indices)
        if H_given==False:
            M = np.random.randn(n,n,n_count)
        
        f = np.random.randn(n)
        F = np.random.randn(n,nth)
        T = np.random.randn(n,nth)   #transformation matrix
        
        # linear transformation to generate A and H
        #M_base = np.random.randn(n,n)
        #A_base = M = np.random.randn(m,n)

        # save the sampled variables
        for position, i in enumerate(indices):
            if H_given == False:
                M_array[i] = M[:,:,position]
            f_array[i] = f
            F_array[i] = F
            T_array[i] = T

    # generate variables only dependent on m (b,sense, blower)
    m_unique, m_counts = np.unique(m_vector, return_counts=True)
    m_dict = dict(zip(m_unique, m_counts))
    b_array = [0 for i in range(iter)]
    sense_array = [0 for i in range(iter)]
    blower_array = [0 for i in range(iter)]
    
    # sample for each m
    for m in m_dict:
        indices, = np.where(m_vector == m)
        b = np.random.rand(m)
        #save variables
        for i in indices:
            b_array[i] = b
            sense_array[i] = np.zeros(m, dtype=np.int32)
            blower_array[i] = np.array([-np.inf for i in range(m)])
    
    print(f"b {b_array[:2]}")
    # generate variables dependent on n and m (A)
    if A_given == False:
        A_array = [0 for i in range(iter)]     
        A_dict = {}
        # pair the combinations of n and m
        pairs = Counter(list(zip(n_vector, m_vector)))       # returns dict with pairs as key and count as value
        for (n_val, m_val), count in pairs.items():       
            A_dict[(n_val, m_val)] = np.random.randn(m_val,n_val,count)

        # save values in A_array
        pair_usage_counter = {key: 0 for key in pairs} # track how many times each pair has been assigned to A_array to find correct place to save it
        for i in range(iter):
            n = n_vector[i]
            m = m_vector[i]
            number_pair_used = pair_usage_counter[(n, m)]
            A_array[i] = A_dict[(n, m)][:,:,number_pair_used]
            pair_usage_counter[(n, m)] += 1

    print("All QPs generated - now transfer problems into graphs")

    # express problem as graph
    for i in range(iter):
        # generate H, btot, ftot
        if A_given == False:
            A = A_array[i]
        if H_given == False:
            H = M_array[i] @ M_array[i].T
        B = A @ (-T_array[i])
        btot = b_array[i] + B @ theta_vector[:,i]
        ftot = f_array[i] + F_array[i] @ theta_vector[:,i]
        
        # solve problem to get the active set 
        _,_,exitflag,info = daqp.solve(H,ftot,A,btot,blower_array[i],sense_array[i])
        lambda_dual= list(info.values())[4]
        active_set = (lambda_dual != 0).astype(int)
        y = torch.tensor((np.hstack((np.zeros(n_vector[i]),active_set)))) 

        # create edge matrix
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
        f1 = np.hstack((ftot,np.zeros(np.shape(btot))))
        b1= np.hstack((np.zeros(np.shape(ftot)),btot))
        eq1 = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
        node_type = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))
        # create matrix with vertex features
        x = torch.tensor([])
        features = np.array([f1, b1, eq1, node_type]).T
        x= torch.tensor(features, dtype=torch.float32)
        data_point = Data(x = x, edge_index=edge_index, edge_attr=edge_attr,y=y)
        
        # save graph
        graph.append(data_point)
        graph_train = graph
    print("All graphs saved")   
    
    
    iter = int(np.rint(0.1*number_of_graphs))
    current_seed = seed
    np.random.seed(current_seed)
    print(seed+1)
    # sample n,m for different sized graphs
    graph = []
    n_vector = np.random.randint(n_min,n_max+1,size = iter)
    m_vector = np.random.randint(m_min,m_max+1, size = iter)
    theta_vector = np.random.randn(nth,iter)
    
    # generate variables only dependent on n (M,f,F,T)
    n_unique, n_counts = np.unique(n_vector, return_counts=True)
    n_dict = dict(zip(n_unique, n_counts))
    #print(n_unique, n_counts)
    if H_given== False:
        M_array = [0 for i in range(iter)]
    f_array = [0 for i in range(iter)]
    F_array = [0 for i in range(iter)]
    T_array = [0 for i in range(iter)]
    
    print(n_dict)
    
    # sample for each n 
    for n in n_dict:
        n_count = n_dict[n]
        #print(n,n_count)
        indices, = np.where(n_vector == n)
        #print(indices)
        if H_given==False:
            M = np.random.randn(n,n,n_count)
        
        f = np.random.randn(n)
        F = np.random.randn(n,nth)
        T = np.random.randn(n,nth)   #transformation matrix
        
        # linear transformation to generate A and H
        #M_base = np.random.randn(n,n)
        #A_base = M = np.random.randn(m,n)

        # save the sampled variables
        for position, i in enumerate(indices):
            if H_given == False:
                M_array[i] = M[:,:,position]
            f_array[i] = f
            F_array[i] = F
            T_array[i] = T

    # generate variables only dependent on m (b,sense, blower)
    m_unique, m_counts = np.unique(m_vector, return_counts=True)
    m_dict = dict(zip(m_unique, m_counts))
    b_array = [0 for i in range(iter)]
    sense_array = [0 for i in range(iter)]
    blower_array = [0 for i in range(iter)]
    
    # sample for each m
    for m in m_dict:
        indices, = np.where(m_vector == m)
        b = np.random.rand(m)
        #save variables
        for i in indices:
            b_array[i] = b
            sense_array[i] = np.zeros(m, dtype=np.int32)
            blower_array[i] = np.array([-np.inf for i in range(m)])
    
    print(f"b {b_array[:2]}")
    # generate variables dependent on n and m (A)
    if A_given == False:
        A_array = [0 for i in range(iter)]     
        A_dict = {}
        # pair the combinations of n and m
        pairs = Counter(list(zip(n_vector, m_vector)))       # returns dict with pairs as key and count as value
        for (n_val, m_val), count in pairs.items():       
            A_dict[(n_val, m_val)] = np.random.randn(m_val,n_val,count)

        # save values in A_array
        pair_usage_counter = {key: 0 for key in pairs} # track how many times each pair has been assigned to A_array to find correct place to save it
        for i in range(iter):
            n = n_vector[i]
            m = m_vector[i]
            number_pair_used = pair_usage_counter[(n, m)]
            A_array[i] = A_dict[(n, m)][:,:,number_pair_used]
            pair_usage_counter[(n, m)] += 1

    print("All QPs generated - now transfer problems into graphs")

    # express problem as graph
    for i in range(iter):
        # generate H, btot, ftot
        if A_given == False:
            A = A_array[i]
        if H_given == False:
            H = M_array[i] @ M_array[i].T
        B = A @ (-T_array[i])
        btot = b_array[i] + B @ theta_vector[:,i]
        ftot = f_array[i] + F_array[i] @ theta_vector[:,i]
        
        # solve problem to get the active set 
        _,_,exitflag,info = daqp.solve(H,ftot,A,btot,blower_array[i],sense_array[i])
        lambda_dual= list(info.values())[4]
        active_set = (lambda_dual != 0).astype(int)
        y = torch.tensor((np.hstack((np.zeros(n_vector[i]),active_set)))) 

        # create edge matrix
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
        f1 = np.hstack((ftot,np.zeros(np.shape(btot))))
        b1= np.hstack((np.zeros(np.shape(ftot)),btot))
        eq1 = np.hstack((np.zeros(np.shape(ftot)),(np.zeros(np.shape(btot)))))
        node_type = np.hstack((np.zeros(np.shape(ftot)),(np.ones(np.shape(btot)))))
        # create matrix with vertex features
        x = torch.tensor([])
        features = np.array([f1, b1, eq1, node_type]).T
        x= torch.tensor(features, dtype=torch.float32)
        data_point = Data(x = x, edge_index=edge_index, edge_attr=edge_attr,y=y)
        
        # save graph
        graph.append(data_point)
        graph_val = graph
    print("All graphs saved")   
    return graph_train,graph_val
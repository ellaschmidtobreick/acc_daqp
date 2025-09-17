import numpy as np
from ctypes import * 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_curve, auc

import daqp
import torch
from torch.utils.data import DataLoader as MLPDataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
import time

from generate_graph_data import generate_qp_graphs_test_data_only, generate_qp_graphs_test_data_only_lmpc
from utils import barplot_iterations, histogram_time, histogram_prediction_time
from model import GNN, MLP
from naive_model import naive_model

# Generate test problems and the corresponding graphs
def test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, H_flexible,A_flexible,model_name):

    
    # Initialization for data generation
    graph_test = []
    H_test = []
    test_iterations_before = []
    test_time_before = []
    H_test = []
    f_test = []
    A_test = []
    b_test = []
    blower = []
    n_vector = []
    m_vector = []

    # Generate test data
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        #graph_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_test_i,b_test_i,blower_i,_,n_i,m_i = generate_qp_graphs_test_data_only(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)
        graph_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_test_i,b_test_i,blower_i,_,n_i,m_i = generate_qp_graphs_test_data_only_lmpc(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)

        graph_test = graph_test + graph_test_i
        test_iterations_before = test_iterations_before + test_iterations_before_i
        test_time_before = test_time_before + test_time_before_i
        H_test = H_test + H_test_i
        f_test = f_test + f_test_i
        A_test = A_test + A_test_i
        b_test = b_test + b_test_i
        blower = blower + blower_i

        n_vector = n_vector + [n_i for i in range(len(test_iterations_before_i))]
        m_vector= m_vector + [m_i for i in range(len(test_iterations_before_i))]

    # Load Data
    test_loader = GraphDataLoader(graph_test, batch_size = 1, shuffle = False)

    # Load model
    model = GNN(input_dim=4, output_dim=1,layer_width = layer_width) 
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth",weights_only=True))
    model.eval()
    
    # Initialization for testing 
    test_all_labels = []
    test_preds = []
    test_time_after = np.zeros(len(test_loader))
    test_iterations_after = np.zeros(len(test_loader))
    test_iterations_difference = np.zeros(len(test_loader))
    prediction_time = np.zeros(len(test_loader))
    graph_pred = []
    num_wrongly_pred_nodes_per_graph = []
    perc_wrongly_pred_nodes_per_graph = []
    
    # Test on data 
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            n = int(n_vector[i])
            m = int(m_vector[i])
            
            # Prediction on test data
            start_time = time.perf_counter()
            output = model(batch,number_of_layers)
            preds = (output.squeeze() > t).long()
            end_time = time.perf_counter()
            prediction_time[i] = end_time - start_time
            
            # Store predictions and labels
            test_preds.extend(preds.numpy())
            test_all_labels.extend(batch.y.numpy())

            # Compute graph metrics
            preds = preds.reshape(-1,n+m)
            preds_numpy = preds.numpy().reshape(-1,n+m)
            all_labels = batch.y.numpy().reshape(-1,n+m)
            graph_pred.extend(np.all(preds_numpy == all_labels, axis=1))
            
            num_wrongly_pred_nodes_per_graph.extend(np.abs((n+m) - np.sum(all_labels == preds_numpy, axis=1)))
            perc_wrongly_pred_nodes_per_graph.extend([(x / (n + m)) for x in num_wrongly_pred_nodes_per_graph])

            #if i<5:
            #    W_true = (batch.y.numpy()[n:] != 0).astype(int).nonzero()[0]
            #    W_pred = (preds_numpy[0][n:] != 0).astype(int).nonzero()[0]
            #    print(f"W_true: {W_true}")
            #    print(f"W_pred: {W_pred}")


            # Solve QPs with predicted active sets
            sense_active = preds.flatten().numpy().astype(np.int32)[n:]
            exitflag = -6
            blower_i = np.array(blower[i], copy=True)

            # solve system until it is solvable
            while exitflag == -6:   # system not solvable
                _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower_i,sense_active)
                lambda_after= list(info.values())[4]
                test_iterations_after[i] = list(info.values())[2]
                test_time_after[i]= list(info.values())[0]

                # remove one active constraint per iteration until problem is solvable
                last_one_index = np.where(sense_active == 1)[-1]
                if last_one_index is not None:
                    sense_active[last_one_index] = 0
                    
            # Solve system one more time without inactive constraints to make sure no active constraints are in there
            # sense_new = (lambda_after != 0).astype(np.int32)
            # _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower_i,sense_new)
            # test_iterations_after[i] += list(info.values())[2]
            # test_time_after[i] += list(info.values())[0]   # only consider solve time, since the daqp solver could be optimized such that the set-up only needs to be done once
            test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
            
    # Compute metrics
    test_acc = accuracy_score(test_all_labels, test_preds)
    test_acc_graph = np.mean(graph_pred)
    test_prec = precision_score(test_all_labels, test_preds)
    test_rec = recall_score(test_all_labels, test_preds)
    test_f1 = f1_score(test_all_labels, test_preds)

    precision, recall, thresholds = precision_recall_curve(test_all_labels,test_preds)
    pr_auc = auc(recall, precision)

    # Compute naive metrics
    naive_acc, naive_prec, naive_rec, naive_f1, naive_perc_wrongly_pred_nodes_per_graph, correctly_predicted_graphs,pr_auc_naive = naive_model(n_vector,m_vector,test_all_labels) 
    # Compute average over graphs
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")
    print(f"Accuracy (graph level) of the model on the test data: {test_acc_graph}")
    print(f"Perc num_wrongly_pred_nodes_per_graph: {np.mean(perc_wrongly_pred_nodes_per_graph)}")
    # Print the PR AUC
    print(f'PR AUC: {pr_auc}')
    print()
    print(f"NAIVE MODEL: acc = {naive_acc}, prec = {naive_prec}, rec = {naive_rec}, f1 = {naive_f1}")
    print(f"Naive model: perc num_wrongly_pred_nodes_per_graph: {np.mean(naive_perc_wrongly_pred_nodes_per_graph)}")
    print(f"Correctly predicted graphs: {correctly_predicted_graphs} out of {len(graph_pred)}")
    print(f'Naive PR AUC: {pr_auc_naive}')
    print()
    print(f'Number of graphs: {len(graph_pred)}, Correctly predicted graphs: {np.sum(graph_pred)}')
    print(f'Mean num_wrongly_pred_nodes_per_graph: {np.mean(num_wrongly_pred_nodes_per_graph)}')
    print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
    print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
    print(f'Prediction time: mean {np.mean(prediction_time)}, min {np.min(prediction_time)}, max {np.max(prediction_time)}')
    print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
    print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
    print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')


    #Plots to vizualize iterations and time
    histogram_time(test_time_before, test_time_after,model_name, save= True)
    histogram_prediction_time(prediction_time,model_name, save = True)
    barplot_iterations(test_iterations_before,test_iterations_after,model_name,save = True)


# Generate test problems and the corresponding graphs
def test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t,  H_flexible,A_flexible,model_name):

    # Initialization for data generation
    data_test = []
    H_test = []
    test_iterations_before = []
    test_time_before = []
    H_test = []
    f_test = []
    A_test = []
    b_test = []
    blower = []
    n_vector = []
    m_vector = []

    # Generate test data
    #data_test, test_iterations_before,test_time_before, H_test,f_test,A_test,b_test,blower,_,n,m = generate_MLP_test_data_only(n,m,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)

    # Generate test data
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        data_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_test_i,b_test_i,blower_i,_,n_i,m_i = generate_MLP_test_data_only(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)

        data_test.extend(data_test_i)
        test_iterations_before = test_iterations_before + test_iterations_before_i
        test_time_before = test_time_before + test_time_before_i
        H_test = H_test + H_test_i
        f_test = f_test + f_test_i
        A_test = A_test + A_test_i
        b_test = b_test + b_test_i
        blower = blower_i
        
        n_vector = n_vector + [n_i for i in range(len(test_iterations_before_i))]
        m_vector= m_vector + [m_i for i in range(len(test_iterations_before_i))]

    # Load Data
    test_loader = MLPDataLoader(data_test, batch_size = 1, shuffle = False)

    # Load model
    input_dimension = n[0]*n[0]+m[0]*n[0]+n[0]+m[0]
    output_dimension = n[0] + m[0]
    model = MLP(input_dim=input_dimension, output_dim=output_dimension,layer_width = layer_width) 
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth",weights_only=True))
    model.eval()
    
    # Initialization for testing 
    test_all_labels = []
    test_preds = []
    test_time_after = np.zeros(len(test_loader))
    test_iterations_after = np.zeros(len(test_loader))
    test_iterations_difference = np.zeros(len(test_loader))
    prediction_time = np.zeros(len(test_loader))
    graph_pred = []
    num_wrongly_pred_nodes_per_graph = []
    perc_wrongly_pred_nodes_per_graph = []
    
    # Test on data 
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            n = int(n_vector[i])
            m = int(m_vector[i])
            
            # Prediction on test data
            start_time = time.perf_counter()
            output = model(batch[0])
            preds = (output.squeeze() > t).long()
            end_time = time.perf_counter()
            prediction_time[i] = end_time - start_time
            
            # Store predictions and labels
            test_preds.extend(preds.numpy().flatten())
            test_all_labels.extend(batch[1].numpy().flatten())

            # Compute grph metrics
            preds = preds.reshape(-1,n+m)
            preds_numpy = preds.numpy().reshape(-1,n+m)
            all_labels = batch[1].numpy().reshape(-1,n+m)
            graph_pred.extend(np.all(preds_numpy == all_labels, axis=1))
            
            num_wrongly_pred_nodes_per_graph.extend(np.abs((n+m) - np.sum(all_labels == preds_numpy, axis=1)))
            perc_wrongly_pred_nodes_per_graph.extend([(x / (n + m)) for x in num_wrongly_pred_nodes_per_graph])

            # if i<5:
            #    W_true = (batch[1].numpy().flatten()[n:]!=0).astype(int).nonzero()[0]
            #    W_pred = (preds_numpy[0][n:] != 0).astype(int).nonzero()[0]
            #    print(f"W_true: {W_true}")
            #    print(f"W_pred: {W_pred}")


            # Solve QPs with predicted active sets
            sense_active = preds.flatten().numpy().astype(np.int32)[n:]
            exitflag = -6
            #blower_i = np.array(blower[i], copy=True)

            #print(H_test[i].size,f_test[i].size,A_test[i].size,b_test[i].size,blower.size,sense_active.size)
            while exitflag == -6:   # system not solvable
                _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower,sense_active)
                lambda_after= list(info.values())[4]
                test_iterations_after[i] = list(info.values())[2]
                test_time_after[i]= list(info.values())[0]

                # remove one active constraint per iteration until problem is solvable
                last_one_index = np.where(sense_active == 1)[-1]
                if last_one_index is not None:
                    sense_active[last_one_index] = 0
                    
            # Solve system one more time without inactive constraints to make sure no active constraints are in there
            sense_new = (lambda_after != 0).astype(np.int32)
            _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower,sense_new)
            test_iterations_after[i] += list(info.values())[2]
            test_time_after[i] += list(info.values())[0]   # only consider solve time, since the daqp solver could be optimized such that the set-up only needs to be done once
            test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
            
    # Compute metrics
    test_acc = accuracy_score(test_all_labels, test_preds)
    test_acc_graph = np.mean(graph_pred)
    test_prec = precision_score(test_all_labels, test_preds)
    test_rec = recall_score(test_all_labels, test_preds)
    test_f1 = f1_score(test_all_labels, test_preds)

    # Compute naive metrics
    n_vector = [n for i in range(len(test_loader))]
    m_vector = [m for i in range(len(test_loader))]
    naive_acc, naive_prec, naive_rec, naive_f1, naive_perc_wrongly_pred_nodes_per_graph = naive_model(n_vector,m_vector,test_all_labels) 
    # Compute average over graphs
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")
    print(f"Accuracy (graph level) of the model on the test data: {test_acc_graph}")
    print(f"Perc num_CORRECTLY_pred_nodes_per_graph: {1-np.mean(perc_wrongly_pred_nodes_per_graph)}")
    print()
    print(f"NAIVE MODEL: acc = {naive_acc}, prec = {naive_prec}, rec = {naive_rec}, f1 = {naive_f1}")
    print(f"Naive model: perc num_wrongly_pred_nodes_per_graph: {np.mean(naive_perc_wrongly_pred_nodes_per_graph)}")

    print()
    print(f'Number of graphs: {len(graph_pred)}, Correctly predicted graphs: {np.sum(graph_pred)}')
    print(f'Mean num_wrongly_pred_nodes_per_graph: {np.mean(num_wrongly_pred_nodes_per_graph)}')
    print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
    print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
    print(f'Prediction time: mean {np.mean(prediction_time)}, min {np.min(prediction_time)}, max {np.max(prediction_time)}')
    print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
    print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
    print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')

    # Plots to vizualize iterations and time
    histogram_time(test_time_before, test_time_after,model_name, save= True)
    histogram_prediction_time(prediction_time,model_name, save = True)
    barplot_iterations(test_iterations_before,test_iterations_after,model_name,save = True)

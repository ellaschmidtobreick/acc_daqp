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
from generate_MLP_data import generate_MLP_test_data_only
from utils import barplot_iterations, histogram_time, histogram_prediction_time, barplot_iter_reduction
from model import GNN, MLP
from naive_model import naive_model

# Generate test problems and the corresponding graphs
def test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, H_flexible,A_flexible,model_name,dataset_type="standard",conv_type="LEConv",two_sided = False,cuda =0):
    # Initialization for data generation
    graph_test = []
    H_test = []
    test_iterations_before = []
    test_time_before = []
    H_test = []
    f_test = []
    A_current = []
    bupper = []
    blower = []
    n_vector = []
    m_vector = []
    m_half= int(m[0]/2)

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate test data
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        if dataset_type == "standard":
            graph_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_current_i,bupper_i,blower_i,_,n_i,m_i = generate_qp_graphs_test_data_only(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible)
            # print("graph_test:", type(graph_test_i))
            # print("test_iterations:", type(test_iterations_before_i))
            # print("test_time:", type(test_time_before_i))
            # print("H_test:", type(i))
            # print("f_test:", type(f_test_i))
            # print("A_test:", type(A_current_i))
            # print("bupper_test:", type(bupper_i))
            # print("blower_test:", type(blower_i))
 
            bupper = bupper + bupper_i
            blower = blower + blower_i
        elif dataset_type == "lmpc":
            graph_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_current_i,bupper_i,blower_i,_,n_i,m_i = generate_qp_graphs_test_data_only_lmpc(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible,two_sided=two_sided)
            bupper.extend([np.array(b, dtype=np.float64) for b in bupper_i])
            blower.extend([np.array(b, dtype=np.float64) for b in blower_i])
        graph_test = graph_test + graph_test_i
        test_iterations_before = test_iterations_before + test_iterations_before_i
        test_time_before = test_time_before + test_time_before_i
        H_test = H_test + H_test_i
        f_test = f_test + f_test_i
        A_current = A_current + A_current_i



        n_vector = n_vector + [n_i for j in range(len(test_iterations_before_i))]
        m_vector= m_vector + [m_i for j in range(len(test_iterations_before_i))]
        # print(len(n_vector))
        # print(len(m_vector))
        # print(blower)
        # print(bupper)
        # print(len(graph_test),len(test_iterations_before),len(test_time_before),len(H_test),len(f_test),len(A_current),len(bupper),len(blower))
    # Load Data
    test_loader = GraphDataLoader(graph_test, batch_size = 1, shuffle = False)

    # Load model
    if dataset_type == "standard" or two_sided == False:
        input_size = 4
    else: #dataset_type == "lmpc":
        input_size = 6

    model = GNN(input_dim=input_size, output_dim=1,layer_width = layer_width,conv_type=conv_type)
    #model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth",weights_only=True))
    model.eval()
    
    # Initialization for testing 
    # test_all_labels = []
    # test_preds = []
    test_time_after = np.zeros(len(test_loader))
    test_iterations_after = np.zeros(len(test_loader))
    test_iterations_difference = np.zeros(len(test_loader))
    prediction_time = np.zeros(len(test_loader))
    # graph_pred = []
    # num_wrongly_pred_nodes_per_graph = []
    # perc_wrongly_pred_nodes_per_graph = []
    # test_all_label_graph = []

    test_loss = 0
    test_correct = 0
    test_total = 0
    test_TP = 0
    test_FP = 0
    test_FN = 0
    test_num_wrong_nodes = 0
    test_total_nodes = 0

    # warm-up model
    warmup_batch = next(iter(test_loader))[0].to(device)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(warmup_batch,number_of_layers,conv_type)
            if device == "cuda":
                torch.cuda.synchronize()
    print("warm-up done")

    # Test on data 
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            batch = batch.to(device)
            n = int(n_vector[i])
            m = int(m_vector[i])
            # print(f"Batch {i}: batch.y shape: {batch.y.shape}, n: {n}")
            # Prediction on test data
            if device == "cuda":
                torch.cuda.synchronize()
            # print(batch.x.dtype)
            start_time = time.perf_counter()
            output = model(batch,number_of_layers,conv_type)
            preds = (output.squeeze() > t).long()
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            prediction_time[i] = end_time - start_time

            # Node-level metrics
            labels = batch.y[n:]         
            preds_constraints = preds[n:]
            test_correct += (preds_constraints == labels).sum().item()
            test_total += labels.numel()
            test_TP += ((preds_constraints == 1) & (labels == 1)).sum().item()
            test_FP += ((preds_constraints == 1) & (labels == 0)).sum().item()
            test_FN += ((preds_constraints == 0) & (labels == 1)).sum().item()
            test_num_wrong_nodes += (preds_constraints != labels).sum().item()
            test_total_nodes += labels.numel()

            # if i % 20 == 0:
            #     # Move only the needed parts to CPU, avoid full .numpy() conversions
            y = batch.y[n:].detach().cpu()
            preds_print = (output.squeeze() > t).long()[n:].detach().cpu()

            W_true = (y != 0).nonzero(as_tuple=True)[0]
            W_pred = (preds_print != 0).nonzero(as_tuple=True)[0]
            # print()
            # print(f"W_true: {W_true.tolist()}")
            # print(f"W_pred: {W_pred.tolist()}")
            #     # print("len W_true, W_pred",y.shape,preds_print.shape)
            #     # Print prediction values for the predicted indices
            pred_vals = output.detach().cpu().squeeze()[W_pred + n]
            # print(f"% pred: {pred_vals.tolist()}")


            # Solve QPs with predicted active sets
            sense_active = preds.flatten().cpu().numpy().astype(np.int32)[n:]   # maybe two instead of one since only one side of constraints in active
            #sense_active = (preds_print != 0).int().cpu().numpy()
            # print("sense_active shape",sense_active.shape)

            exitflag = -6
            counter = 0
            
            # Prepare arrays safely for DAQP
            H_i = np.ascontiguousarray(H_test[i], dtype=np.float64)
            f_i = np.ascontiguousarray(f_test[i], dtype=np.float64)
            A_i = np.ascontiguousarray(A_current[i], dtype=np.float64)
            bupper_i = np.ascontiguousarray(bupper[i].flatten(), dtype=np.float64)
            blower_i = np.ascontiguousarray(blower[i].flatten(), dtype=np.float64)
            sense_i = np.ascontiguousarray(sense_active, dtype=np.int32)

            counter = 0
            max_removals = 10


            while exitflag == -6:   # system not solvable
                _,_,exitflag,info = daqp.solve(H_i, f_i, A_i, bupper_i, blower_i, sense_i)
                lambda_after= list(info.values())[4]
                test_iterations_after[i] = list(info.values())[2]
                # solve and set-up time
                test_time_after[i]= list(info.values())[0] + list(info.values())[1]

                # remove one active constraint per iteration until problem is solvable
                last_one_index = np.where(sense_active == 1)[-1]
                if last_one_index is not None:
                    sense_active[last_one_index] = 0
                   

            # while exitflag < 0 and counter <= max_removals:
            #     # print(f"Evaluate test sample {i}, attempt {counter+1}")

            #     # If exceeded max removals, solve with empty active set
            #     if counter >= max_removals:
            #         # print("Max removals reached, trying with empty active set")
            #         sense_i = np.zeros_like(sense_i, dtype=np.int32)
            #         # print("sense_i",sense_i)
            #     x, _, exitflag, info = daqp.solve(H_i, f_i, A_i, bupper_i, blower_i, sense_i)
            #     counter += 1

            #     # print("exitflag:", exitflag)

            #     # lambda_after = list(info.values())[4]
            #     test_iterations_after[i] = list(info.values())[2]
            #     test_time_after[i] = list(info.values())[0] + list(info.values())[1]

            #     # Stop if feasible
            #     if exitflag >= 0:
            #         # print("Problem solved successfully")

            #         sense_compare = np.zeros_like(sense_i, dtype=np.int32)
            #         # x_compare, _, exitflag, info = daqp.solve(H_i, f_i, A_i, bupper_i, blower_i, sense_compare)

            #         # diff_mask =  ~np.isclose(x, x_compare, rtol=1e-9, atol=1e-12)
            #         # if (np.where(diff_mask)[0]).size>0 :
            #         #     print("Incorrectly solved")
            #         # else:
            #         #     print("Correctly solved")
            #         # break

            #     # Remove last nonzero constraint safely
            #     nonzero_idx = np.flatnonzero(sense_i)
            #     if nonzero_idx.size == 0:
            #         # print("All constraints removed, but problem still infeasible")
            #         break
            #     last_index = nonzero_idx[-1]
            #     sense_i[last_index] = 0




                # print(f"Removed constraint at index {last_index}, remaining active constraints: {nonzero_idx[:-1]}")

                # remove one active constraint per iteration until problem is solvable
                # last_one_index = np.where(sense_active == 1)[0]   # extract array of indices
                # print(last_one_index)
                # if len(last_one_index) > 0:
                #     sense_active[last_one_index[-1]] = 0
                # else:
                #     print("No active constraints to remove, but system still not solvable.")
                #     break

            # print("iter before / after:", test_iterations_before[i],"/", test_iterations_after[i])


            # print(f"test iterations before: {test_iterations_before[i]}")
            # print(f"test iterations after: {test_iterations_after[i]}")
            # print() 
            # Solve system one more time without inactive constraints to make sure no active constraints are in there
            #print(sense_active)
            # _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower_i,sense_active)
            # print(f"Info before: {info}")

            # sense_new = (lambda_after != 0).astype(np.int32)
            # #print(sense_new)
            # _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower_i,sense_new)
            # print(f"Info after: {info}")
            # test_iterations_after[i] += list(info.values())[2]
            # test_time_after[i] += list(info.values())[0]   # only consider solve time, since the daqp solver could be optimized such that the set-up only needs to be done once
            
            test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
            if device == "cuda" and i % 50 == 0:
                torch.cuda.empty_cache()
            # print(f"Finished sample {i+1} / {len(test_loader)}")
            # print()

    # print("evalution now")
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    test_prec = test_TP / (test_TP + test_FP + 1e-8)
    test_rec = test_TP / (test_TP + test_FN + 1e-8)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)

    # Compute naive metrics
    # naive_acc, naive_prec, naive_rec, naive_f1, naive_perc_wrongly_pred_nodes_per_graph, correctly_predicted_graphs,pr_auc_naive = naive_model(n_vector,m_vector,test_all_labels) 
    # Compute average over graphs
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")
    # print()
    # print(f"NAIVE MODEL: acc = {naive_acc}, prec = {naive_prec}, rec = {naive_rec}, f1 = {naive_f1}")
    # print(f"Naive model: perc num_wrongly_pred_nodes_per_graph: {np.mean(naive_perc_wrongly_pred_nodes_per_graph)}")
    # # print(f"Correctly predicted graphs: {correctly_predicted_graphs} out of {len(graph_pred)}")
    # print(f'Naive PR AUC: {pr_auc_naive}')
    print()
    # print(f'Number of graphs: {len(graph_pred)}, Correctly predicted graphs: {np.sum(graph_pred)}')
    # print(f'Mean num_wrongly_pred_nodes_per_graph: {np.mean(num_wrongly_pred_nodes_per_graph)}')
    # print(len(test_all_label_graph))
    # print(f"Number of graph without an active constraint: {count_all_zero} / {count_all_zero / len(test_all_label_graph)*100}%")
    print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
    print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
    print(f'Test time reduction: mean {np.mean(np.array(test_time_before)-np.array(test_time_after))}, min {np.min(np.array(test_time_before)-np.array(test_time_after))}, max {np.max(np.array(test_time_before)-np.array(test_time_after))}')
    print(f'Prediction time: mean {np.mean(prediction_time)}, min {np.min(prediction_time)}, max {np.max(prediction_time)}')
    print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
    print(f"Test iter before: quantiles {np.percentile(test_iterations_before, [10,25, 50, 75,90])}")
    print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
    print(f"Test iter after: quantiles {np.percentile(test_iterations_after, [10,25, 50, 75,90])}")
    print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')
    print(f"Test iter after: quantiles {np.percentile(test_iterations_difference, [5,10,20,30,40, 50, 60,70,80,90,95])}")


    #Plots to vizualize iterations and time
    # histogram_time(test_time_before, test_time_after,model_name, save= True)
    # #histogram_prediction_time(prediction_time,model_name, save = True)
    # barplot_iter_reduction(test_iterations_difference,model_name, save = True)
    # barplot_iterations(test_iterations_before,test_iterations_after,model_name,save = True)

    #return np.mean(test_time_before), np.mean(test_time_after),np.mean(np.array(test_time_before)-np.array(test_time_after)), np.mean(prediction_time)
    #return test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference
    #return test_acc, test_prec, test_rec, test_f1
    return prediction_time, test_time_after, test_iterations_after

# Generate test problems and the corresponding graphs
def test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t,  H_flexible,A_flexible,model_name,dataset_type="standard",cuda = 0):

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

    device = torch.device(f"cuda{cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate test data
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        data_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_test_i,b_test_i,blower_i,_,n_i,m_i = generate_MLP_test_data_only(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible,dataset_type=dataset_type)

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
    model = model.to(device)
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth",weights_only=True))
    model.eval()
    
    # Initialization for testing 

    test_time_after = np.zeros(len(test_loader))
    test_iterations_after = np.zeros(len(test_loader))
    test_iterations_difference = np.zeros(len(test_loader))
    prediction_time = np.zeros(len(test_loader))
 
    counter = 0
    test_loss = 0
    test_correct = 0
    test_total = 0
    test_TP = 0
    test_FP = 0
    test_FN = 0
    test_num_wrong_nodes = 0
    test_total_nodes = 0

    # Warm-up once
    warmup_batch = next(iter(test_loader))[0].to(device)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(warmup_batch)
            if device == "cuda":
                torch.cuda.synchronize()

    # Test on data 
    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            batch = [b.to(device) for b in batch]
            inputs, labels = batch
            n = int(n_vector[i])
            m = int(m_vector[i])
            
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = model(inputs)
            
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            prediction_time[i] = end_time - start_time
            preds = (output.squeeze() > t).long()

            
            # Node-level metrics
            print(labels.shape, preds.shape)
            labels_constraints = labels.squeeze()[n:]
            preds_constraints = preds[n:]
            print(labels_constraints.shape,preds_constraints.shape)
            test_correct += (preds_constraints == labels_constraints).sum().item()
            test_total += labels_constraints.numel()
            test_TP += ((preds_constraints == 1) & (labels_constraints == 1)).sum().item()
            test_FP += ((preds_constraints == 1) & (labels_constraints == 0)).sum().item()
            test_FN += ((preds_constraints == 0) & (labels_constraints == 1)).sum().item()
            test_num_wrong_nodes += (preds_constraints != labels_constraints).sum().item()
            test_total_nodes += labels_constraints.numel()

            # Solve QPs with predicted active sets
            sense_active = preds.flatten().cpu().numpy().astype(np.int32)[n:]
            exitflag = -6
            #blower_i = np.array(blower[i], copy=True)

            #print(H_test[i].size,f_test[i].size,A_test[i].size,b_test[i].size,blower.size,sense_active.size)
            while exitflag == -6:   # system not solvable
                _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower,sense_active)
                lambda_after= list(info.values())[4]
                test_iterations_after[i] = list(info.values())[2]
                # solve and set-up time
                test_time_after[i]= list(info.values())[0] + list(info.values())[1]

                # remove one active constraint per iteration until problem is solvable
                last_one_index = np.where(sense_active == 1)[-1]
                if last_one_index is not None:
                    sense_active[last_one_index] = 0
                    
            test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
            
    # Compute metrics

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    test_prec = test_TP / (test_TP + test_FP + 1e-8)
    test_rec = test_TP / (test_TP + test_FN + 1e-8)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)

    # test_acc = accuracy_score(test_all_labels, test_preds)
    # test_prec = precision_score(test_all_labels, test_preds)
    # test_rec = recall_score(test_all_labels, test_preds)
    # test_f1 = f1_score(test_all_labels, test_preds)

    # # Compute naive metrics
    # n_vector = [n for i in range(len(test_loader))]
    # m_vector = [m for i in range(len(test_loader))]
    # naive_acc, naive_prec, naive_rec, naive_f1, naive_perc_wrongly_pred_nodes_per_graph, _, pr_auc= naive_model(n_vector,m_vector,test_all_labels) 
    
    # Compute average over graphs
    print(counter)
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")
    print()
    # print(f"NAIVE MODEL: acc = {naive_acc}, prec = {naive_prec}, rec = {naive_rec}, f1 = {naive_f1}")
    # print(f"Naive model: perc num_wrongly_pred_nodes_per_graph: {np.mean(naive_perc_wrongly_pred_nodes_per_graph)}")
    # print(f"PR AUC: {pr_auc}")
    print()
    # print(f'Number of graphs: {len(graph_pred)}, Correctly predicted graphs: {np.sum(graph_pred)}')
    # print(f'Mean num_wrongly_pred_nodes_per_graph: {np.mean(num_wrongly_pred_nodes_per_graph)}')
    print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
    print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
    print(f"Time reduction: mean {np.mean(np.array(test_time_before)-np.array(test_time_after))}, min {np.min(np.array(test_time_before)-np.array(test_time_after))}, max {np.max(np.array(test_time_before)-np.array(test_time_after))}")
    print(f'Prediction time: mean {np.mean(prediction_time)}, min {np.min(prediction_time)}, max {np.max(prediction_time)}')
    print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
    print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
    print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')

    # Plots to vizualize iterations and time
    # histogram_time(test_time_before, test_time_after,model_name, save= True)
    # histogram_prediction_time(prediction_time,model_name, save = True)
    # barplot_iterations(test_iterations_before,test_iterations_after,model_name,save = True)
    
    #return np.mean(test_time_before), np.mean(test_time_after),np.mean(np.array(test_time_before)-np.array(test_time_after)), np.mean(prediction_time)
    return prediction_time, test_time_after, test_iterations_after #np.mean(prediction_time), np.mean(test_time_after), np.mean(test_iterations_after)

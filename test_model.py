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
def test_GNN(n,m,nth, seed, data_points,layer_width,number_of_layers,t, H_flexible,A_flexible,model_name,dataset_type="standard",conv_type="LEConv",two_sided = False,cuda =0,sparsity ="dense",relu_slope = 0.1):
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
            graph_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_current_i,bupper_i,blower_i,_,n_i,m_i = generate_qp_graphs_test_data_only(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible,sparsity =sparsity)
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

    # Load Data
    test_loader = GraphDataLoader(graph_test, batch_size = 1, shuffle = False)

    # Load model
    if dataset_type == "standard" or two_sided == False:
        input_size = 4
    else: #dataset_type == "lmpc":
        input_size = 6

    model = GNN(input_dim=input_size, output_dim=1,layer_width = layer_width,conv_type=conv_type)
    model = model.to(device)
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth",weights_only=True))
    model.eval()
    
    # Initialization for testing 
    test_time_after = np.zeros(len(test_loader))
    test_iterations_after = np.zeros(len(test_loader))
    test_iterations_difference = np.zeros(len(test_loader))
    prediction_time = np.zeros(len(test_loader))

    test_loss = 0
    test_correct = 0
    test_total = 0
    test_TP = 0
    test_FP = 0
    test_FN = 0
    test_num_wrong_nodes = 0
    test_total_nodes = 0

    # Warm-up model
    warmup_batch = next(iter(test_loader))[0].to(device)
    with torch.inference_mode():
        for _ in range(5):
            _ = model(warmup_batch,number_of_layers,conv_type,relu_slope)
            if device == "cuda":
                torch.cuda.synchronize()
    print("warm-up done")

    # Test on data 
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            batch = batch.to(device)
            n = int(n_vector[i])
            m = int(m_vector[i])
            # Prediction on test data
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = model(batch,number_of_layers,conv_type,relu_slope)
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

            # Print predictions vs. true labels
            # y = batch.y[n:].detach().cpu()
            # preds_print = (output.squeeze() > t).long()[n:].detach().cpu()
            # W_true = (y != 0).nonzero(as_tuple=True)[0]
            # W_pred = (preds_print != 0).nonzero(as_tuple=True)[0]
            # print()
            # print(f"W_true: {W_true.tolist()}")
            # print(f"W_pred: {W_pred.tolist()}")

            pred_vals = output.detach().cpu().squeeze()[n:]

            # Solve QPs with predicted active sets
            sense_active = preds.flatten().cpu().numpy().astype(np.int32)[n:]

            # Prepare arrays safely for DAQP
            H_i = np.ascontiguousarray(H_test[i], dtype=np.float64)
            f_i = np.ascontiguousarray(f_test[i], dtype=np.float64)
            A_i = np.ascontiguousarray(A_current[i], dtype=np.float64)
            bupper_i = np.ascontiguousarray(bupper[i].flatten(), dtype=np.float64)
            blower_i = np.ascontiguousarray(blower[i].flatten(), dtype=np.float64)
            sense_i = np.ascontiguousarray(sense_active, dtype=np.int32)

            # print(H_i.shape,f_i.shape,A_i.shape,bupper_i.shape,blower_i.shape,sense_i.shape)
            exitflag = -6           

            # system overdetermined
            while exitflag == -6:
                _,_,exitflag,info = daqp.solve(H_i, f_i, A_i, bupper_i, blower_i, sense_i)
                lambda_after= list(info.values())[4]
                test_iterations_after[i] = list(info.values())[2]
                # solve and set-up time
                test_time_after[i]= list(info.values())[0] + list(info.values())[1]
                   
                if exitflag == -6:
                    active_indices = np.where(sense_i == 1)[0]
                    if len(active_indices) == 0:
                        print("All constraints removed; still infeasible.")
                        break

                    # Remove active constraint with lowest conficence
                    probs = pred_vals[active_indices]
                    lowest_prob_index = active_indices[np.argmin(probs)]
                    sense_i[lowest_prob_index] = 0

            test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
            if device == "cuda" and i % 50 == 0:
                torch.cuda.empty_cache()

    # Evaluation
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    test_prec = test_TP / (test_TP + test_FP + 1e-8)
    test_rec = test_TP / (test_TP + test_FN + 1e-8)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)

    # Print results
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")
    print()
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
    # histogram_prediction_time(prediction_time,model_name, save = True)
    # barplot_iter_reduction(test_iterations_difference,model_name, save = True)
    # barplot_iterations(test_iterations_before,test_iterations_after,model_name,save = True)

    return prediction_time, test_time_before, test_time_after, test_iterations_before,test_iterations_after, test_iterations_difference
    #return np.mean(test_time_before), np.mean(test_time_after),np.mean(np.array(test_time_before)-np.array(test_time_after)), np.mean(prediction_time)
    #return test_acc, test_prec, test_rec, test_f1

# Generate test problems and the corresponding graphs
def test_MLP(n,m,nth, seed, data_points,layer_width,number_of_layers,t,  H_flexible,A_flexible,model_name,dataset_type="standard",cuda = 0,sparsity ="dense",relu_slope = 0.1):

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

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Generate test data
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        data_test_i, test_iterations_before_i,test_time_before_i, H_test_i,f_test_i,A_test_i,b_test_i,blower_i,_,n_i,m_i = generate_MLP_test_data_only(n_i,m_i,nth,seed,data_points,H_flexible=H_flexible,A_flexible=A_flexible,dataset_type=dataset_type,sparsity=sparsity)

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
            _ = model(warmup_batch,relu_slope)
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
            output = model(inputs,relu_slope)
            
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            prediction_time[i] = end_time - start_time
            preds = (output.squeeze() > t).long()
            pred_vals = output.detach().cpu().squeeze()[n:]

            # Node-level metrics
            labels_constraints = labels.squeeze()[n:]
            preds_constraints = preds[n:]
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

            while exitflag == -6:   # system not solvable
                _,_,exitflag,info = daqp.solve(H_test[i],f_test[i],A_test[i],b_test[i],blower,sense_active)
                lambda_after= list(info.values())[4]
                test_iterations_after[i] = list(info.values())[2]
                # solve and set-up time
                test_time_after[i]= list(info.values())[0] + list(info.values())[1]

                if exitflag == -6:
                    active_indices = np.where(sense_active == 1)[0]
                    if len(active_indices) == 0:
                        print("All constraints removed; still infeasible.")
                        break

                    # Remove active constraint with lowest conficence
                    probs = pred_vals[active_indices]
                    lowest_prob_index = active_indices[np.argmin(probs)]
                    sense_active[lowest_prob_index] = 0
                    
            test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
            
    # Compute metrics

    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    test_prec = test_TP / (test_TP + test_FP + 1e-8)
    test_rec = test_TP / (test_TP + test_FN + 1e-8)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)

    # Compute average over graphs
    print("TESTING")
    print(f"Accuracy (node level) of the model on the test data: {test_acc}")
    print(f"Precision of the model on the test data: {test_prec}")
    print(f"Recall of the model on the test data: {test_rec}")
    print(f"F1-Score of the model on the test data: {test_f1}")
    print()
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

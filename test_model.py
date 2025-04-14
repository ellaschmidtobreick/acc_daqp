import numpy as np
from ctypes import * 
import matplotlib.pyplot as plt
import daqp
import torch
from torch_geometric.loader import DataLoader
import time
import os

from generate_graph_data import generate_qp_graphs_test_data_only
from utils import barplot_iterations, boxplot_time, histogram_time, histogram_prediction_time, histogram_time_no_ignoed_constraints, barplot_iterations_no_ignored_constraints
import config
from model import GNN

# to scale a bigger n,m can be used in testing than in training
n = 10 #config.n
m= 40 #config.m
model_path = "saved_models/model_10v_40c.pth"

# Generate test problems and the corresponding graphs
graph_test, test_iterations_before,test_time_before_solve,test_time_before_setup, H,f_test,A,b_test,blower,sense = generate_qp_graphs_test_data_only(n,m,config.nth,config.seed,config.data_points)

# Load Data
test_loader = DataLoader(graph_test, batch_size = 1, shuffle = False)

# Define and load model
model = GNN(input_dim=3, output_dim=1,layer_width = 128) 
optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define variables to analyze performance 
correct = 0
total = 0
test_num_wrongly_pred_nodes_per_graph = 0
test_all_labels = []
test_preds = []
test_time_after = np.zeros(len(test_loader))
test_time_after_solve = np.zeros(len(test_loader))
test_time_after_setup = np.zeros(len(test_loader))
test_time_after_no_ignored_constraints = np.zeros(len(test_loader))
test_time_after_no_ignored_constraints_solve = np.zeros(len(test_loader))
test_time_after_no_ignored_constraints_setup = np.zeros(len(test_loader))
test_iterations_after = np.zeros(len(test_loader))
test_iterations_after_no_ignored_constraints = np.zeros(len(test_loader))
test_iterations_difference = np.zeros(len(test_loader))
W_diff_FN = 0
output_FN = []
output_TN = []
output_FP = []
W_diff_list = []
prediction_time = np.zeros(len(test_loader))

#Final evaluation on test data
with torch.no_grad():
    for i,batch in enumerate(test_loader):
        # save prediction time
        start_time = time.time()
        output = model(batch,config.number_of_layers)
        # add a new class of constraints that should get ignored
        preds = torch.where(output.squeeze() < 0.001, torch.tensor(4), torch.where(output.squeeze() > config.t, torch.tensor(1),torch.tensor(0)))
        end_time = time.time()
        prediction_time[i] = end_time - start_time
        #original prediction
        preds_old = (output.squeeze() > config.t).long()

        # Varibales to save printable output
        correct += (((output.squeeze() > config.t).long()) == batch.y).sum().item()
        total += batch.y.size(0)
        
        # Store predictions and labels
        test_preds.extend(preds.numpy())
        test_all_labels.extend(batch.y.numpy())

        # reshape predictions for graph accuracy
        preds = preds.reshape(-1,n+m)

        # using my own solver to get the values for time and iteration before
        # empty working set, solving the full QP
        #W = []
        # start_time_before = time.perf_counter()
        # x_before, lambda_before, _,it_before  = self_implement_daqp.daqp_self(H,f_test[i,:],A,b_test[i,:],sense,W)
        # end_time_before = time.perf_counter()
        # W_before = [j for j, value in enumerate(lambda_before) if value != 0]
        # x,_,_,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense)
        # lambda_before =list(info.values())[4]
        # test_iterations_before[i] = list(info.values())[2]
        # test_time_before[i]= list(info.values())[0]
        # test_time_before[i] = end_time_before- start_time_before
        
        # true active set
        W_true = (batch.y.numpy()[n:] != 0).astype(int).nonzero()[0]
            
        # get the predicted active set
        preds_constraints = (preds.flatten()[n:] == 1)
        W_pred = torch.nonzero(preds_constraints, as_tuple=True)[0].numpy()
        sense_active = preds.flatten().numpy().astype(np.int32)[n:] # sense 1 if active, 4 to ignore
        
        # using my own solver to get the values for time and iteration after
        # start_time_after = time.perf_counter()
        # x_after, lambda_after, _, it_after = self_implement_daqp.daqp_self(H,f_test[i,:],A,b_test[i,:],sense,W_pred) 
        # end_time_after = time.perf_counter()
        
        # solve the QP with the predicted active set 
        x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_active)
        lambda_after = list(info.values())[4]
        sense_new = (lambda_after != 0).astype(np.int32)
        test_iterations_after[i] = list(info.values())[2]
        test_time_after_solve[i] = list(info.values())[0]   # consider solve and set-up time
        test_time_after_setup[i] = list(info.values())[1]     # consider solve and set-up time

        # if problem is overdetermined, reduce the initial working set until it is not overdetermined anymore
        if exitflag == -6:
            while exitflag == -6:
                x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_active)
                # change last constraint claimed being active in the initial working set to not being active           
                last_one_index = np.where(sense_active == 1)[0][-1]
                if last_one_index is not None:
                    sense_active[last_one_index] = 0
            
            # set all ignored constraints to zero (not ignored)
            lambda_after = list(info.values())[4]
            sense_new = (lambda_after != 0).astype(np.int32)
            test_iterations_after[i] = list(info.values())[2]
            test_time_after_solve[i] = list(info.values())[0]    # consider solve and set-up time
            test_time_after_setup[i] = list(info.values())[1]     # consider solve and set-up time
            # solve once again without any ignored constraints, to make sure no active constraints were ignored 
            # if no active constraints war ignored only one iteration
            # if active constraint was ignored, multiple iterations required
            x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_new)
            #print(x,fval,exitflag,info)
            #lambda_after = list(info.values())[4]
            test_iterations_after[i] += list(info.values())[2]
            # only consider solve time, set-up could be optimized and only done once
            test_time_after_solve[i] += list(info.values())[0]
           
        # Test new method against not using any ignored constraints
        # set all ignored constraints, to not being ignored anymore
        sense_compare = preds_old.flatten().numpy().astype(np.int32)[n:]
        x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_compare)
        test_iterations_after_no_ignored_constraints[i] = list(info.values())[2]
        # only consider solve time, set-up could be optimized such that it only needs to be done once
        test_time_after_no_ignored_constraints_solve[i] = list(info.values())[0]
        test_time_after_no_ignored_constraints_setup[i] = list(info.values())[1]

        # if problem is overdetermined, reduce the initial working set until it is not overdetermined anymore
        if exitflag == -6:
            while exitflag == -6:
                x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_active)
                # change last constraint claimed being active in the initial working set to not being active           
                last_one_index = np.where(sense_active == 1)[0][-1]
                if last_one_index is not None:
                    sense_active[last_one_index] = 0
            
            # set all ignored constraints to zero (not ignored)
            lambda_after = list(info.values())[4]
            sense_new = (lambda_after != 0).astype(np.int32)
            test_iterations_after_no_ignored_constraints[i] = list(info.values())[2]
            test_time_after_no_ignored_constraints_solve[i] = list(info.values())[0]    # consider solve and set-up time
            test_time_after_no_ignored_constraints_setup[i] = list(info.values())[1]     # consider solve and set-up time

            # solve once again without any ignored constraints, to make sure no active constraints were ignored 
            # if no active constraints war ignored only one iteration
            # if active constraint was ignored, multiple iterations required
            x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_new)
            #print(x,fval,exitflag,info)
            #lambda_after = list(info.values())[4]
            test_iterations_after_no_ignored_constraints[i] += list(info.values())[2]
            # only consider solve time, set-up could be optimized and only done once
            test_time_after_no_ignored_constraints_solve[i] += list(info.values())[0]
        
        # analyze FN
        output_constraints = output.flatten()[n:]
        W_pred_set = set(W_pred)
        W_true_set = set(W_true)
        W_diff = sorted(W_true_set.symmetric_difference(W_pred_set))
        W_diff_FN = sorted(W_true_set.difference(W_pred_set))
        output_FN.extend(output.flatten().numpy()[n:][W_diff_FN])

        # analyze TN
        all_indices = set(range(m))
        W_pred_complement = all_indices - W_pred_set
        W_true_complement = all_indices - W_true_set
        W_common_zero_elements = sorted(W_pred_complement.intersection(W_true_complement))
        output_TN.extend(output.flatten().numpy()[n:][W_common_zero_elements])
        
        # analyze FP
        W_diff_FP = sorted(W_pred_set.difference(W_true_set))
        W_diff_list.extend(W_diff_FP)
        output_FP.extend(output.flatten().numpy()[n:][W_diff_FP])
        # print(f"Difference in W {len(set(W_true)^set(W_pred))}, It before {it_before}, If after {it_after}, It diff {it_before - it_after}")
        
        #print(f" W true {W_true} \n W pred {W_pred}")
        #print(output.flatten().numpy()[n:][W_diff_FP])
        #print(output.flatten().numpy()[n:][W_diff])
        
        #print(output_constraints[W_pred].numpy())
        #print(output.flatten().numpy()[n:])
        # test_iterations_after[i] = it_after
        # test_time_after[i] = end_time_after - start_time_after
        test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
        
# Compute average over graph metrices
test_all_label_graph = np.array(test_all_labels).reshape(-1,n+m)
test_preds_graph = np.array(test_preds).reshape(-1,n+m)
test_preds_graph[test_preds_graph == 4] = 0
acc_graph_test = np.mean(np.all(test_all_label_graph == test_preds_graph, axis=1))
test_num_wrongly_pred_nodes_per_graph = np.abs((n+m) - np.sum(test_all_label_graph == test_preds_graph, axis=1))

test_time_after_no_ignored_constraints = test_time_after_no_ignored_constraints_solve + test_time_after_no_ignored_constraints_setup
test_time_after = test_time_after_setup + test_time_after_solve
test_time_before = test_time_before_setup + test_time_before_solve

# print output
print(f'correct:{correct}, total: {total}')
print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')
print(f'Number of graphs: {test_all_label_graph.shape[0]}, Correctly predicted graphs: {np.sum(np.all(test_all_label_graph == test_preds_graph, axis=1))}')
print(f'Graph accuracy of the model on the test data: {100 * acc_graph_test:.2f}%')
print(test_num_wrongly_pred_nodes_per_graph)
print(f'Mean: {np.mean(test_num_wrongly_pred_nodes_per_graph)}')
print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
print(f'Test time before setup: mean {np.mean(test_time_before_setup)}, min {np.min(test_time_before_setup)}, max {np.max(test_time_before_setup)}')
print(f'Test time before solve: mean {np.mean(test_time_before_solve)}, min {np.min(test_time_before_solve)}, max {np.max(test_time_before_solve)}')
print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
print(f'Test time after setup: mean {np.mean(test_time_after_setup)}, min {np.min(test_time_after_setup)}, max {np.max(test_time_after_setup)}')
print(f'Test time after solve: mean {np.mean(test_time_after_solve)}, min {np.min(test_time_after_solve)}, max {np.max(test_time_after_solve)}')
print(f'Test time after, no ignored constraints: mean {np.mean(test_time_after_no_ignored_constraints)}, min {np.min(test_time_after_no_ignored_constraints)}, max {np.max(test_time_after_no_ignored_constraints)}')
print(f'Test time after, no ignored constraints setup: mean {np.mean(test_time_after_no_ignored_constraints_setup)}, min {np.min(test_time_after_no_ignored_constraints_setup)}, max {np.max(test_time_after_no_ignored_constraints_setup)}')
print(f'Test time after, no ignored constraints solve: mean {np.mean(test_time_after_no_ignored_constraints_solve)}, min {np.min(test_time_after_no_ignored_constraints_solve)}, max {np.max(test_time_after_no_ignored_constraints_solve)}')
print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')
#print(output_FN)
#print(output_FP)
#print([val for val in output_FN if 1e-5 < val < 0.1])
#print(np.min(output_FP))

# plot which output values the FP have
# plt.hist(output_FP, bins=50,range=(np.min(output_FP),np.max(output_FP)), alpha=0.7, label='FN', color='blue')
# plt.show()

# # plot which constraints are most often FP
# all_iterations = sorted(set(int(i) for i in W_diff_list).union(int(i) for i in W_diff_list))
# # fill up the ticks that do not have values
# all_iterations = range(0,np.max(all_iterations)+1,1)
# # Prepare values
# from collections import Counter
# count_dict = Counter(W_diff_list)
# values = [count_dict.get(it, 0) for it in all_iterations]
# # Bar width and positions
# x = np.arange(len(all_iterations))
# width = 0.4
# # Plot bars
# plt.bar(x - width/2, values, width=width, label="Without GNN", color='blue')
# plt.show()

# threshold tuning
# print(best_threshold)
# print(best_mean)


#Plots to show reduction in time and iterations
#boxplot_time(test_time_before,test_time_after,"time",save = False)
histogram_time(test_time_before, test_time_after, save= False)
# histogram_time(test_time_before_solve,test_time_after_solve, save= False)
# histogram_time(test_time_before_solve,test_time_after_no_ignored_constraints_solve, save= False)
# histogram_time(test_time_before_setup,test_time_after_setup, save= False)
# histogram_time(test_time_before_setup,test_time_after_no_ignored_constraints_setup, save= False)

histogram_time_no_ignoed_constraints(test_time_after_no_ignored_constraints,test_time_after, save= False)
histogram_time_no_ignoed_constraints(test_time_after_no_ignored_constraints_setup,test_time_after_setup, save= False)
histogram_time_no_ignoed_constraints(test_time_after_no_ignored_constraints_solve,test_time_after_solve, save= False)

# histogram_prediction_time(prediction_time,save=False)
# #boxplot_time(test_iterations_before,test_iterations_after, "iterations",save = False)
# barplot_iterations(test_iterations_before,test_iterations_after, "iterations",save = False)
# barplot_iterations_no_ignored_constraints(test_iterations_after_no_ignored_constraints,test_iterations_after, "iterations",save = False)

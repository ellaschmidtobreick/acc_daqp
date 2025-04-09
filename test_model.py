import numpy as np
from ctypes import * 
import matplotlib.pyplot as plt
import daqp
import torch
from torch_geometric.loader import DataLoader

from generate_graph_data import generate_qp_graphs_test_data_only
from utils import barplot_iterations, boxplot_time, histogram_time
import config
from model import GNN

# to scale a bigger n,m can be used in testing than in training
n = 10 # config.n
m= 40 # config.m

# Generate test problems and the corresponding graphs
graph_test, test_iterations_before,test_time_before, H,f_test,A,b_test,blower,sense = generate_qp_graphs_test_data_only(n,m,config.nth,config.seed,config.data_points)

# Load Data
test_loader = DataLoader(graph_test, batch_size = 1, shuffle = False)

model = GNN(input_dim=3, output_dim=1,layer_width = 128) 
optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
   
#Final evaluation on test data
model.load_state_dict(torch.load("current_model.pth"))
model.eval()
correct = 0
total = 0
test_num_wrongly_pred_nodes_per_graph = 0

test_all_labels = []
test_preds = []
#test_time_before = np.zeros(len(test_loader))
test_time_after = np.zeros(len(test_loader))
#test_iterations_before = np.zeros(len(test_loader))
test_iterations_after = np.zeros(len(test_loader))
test_iterations_difference = np.zeros(len(test_loader))
W_diff_FN = 0
output_FN = []
    
with torch.no_grad():
    for i,batch in enumerate(test_loader):
        output = model(batch,config.number_of_layers)
        preds = (output.squeeze() > config.t).long()
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)
        # Store predictions and labels
        test_preds.extend(preds.numpy())
        test_all_labels.extend(batch.y.numpy())

        # reshape predictions for graph accuracy
        preds = preds.reshape(-1,n+m)
        
        # solve full QP
        #W = []
        # start_time_before = time.perf_counter()
        # x_before, lambda_before, _,it_before  = self_implement_daqp.daqp_self(H,f_test[i,:],A,b_test[i,:],sense,W)
        # end_time_before = time.perf_counter()
        # W_before = [j for j, value in enumerate(lambda_before) if value != 0]
        # x,_,_,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense)
        # lambda_before =list(info.values())[4]
        W_true = (batch.y.numpy()[n:] != 0).astype(int).nonzero()[0]
        # test_iterations_before[i] = list(info.values())[2]
        # test_time_before[i]= list(info.values())[0]
        # test_time_before[i] = end_time_before- start_time_before
                
        # solve the reduced QPs to see the reduction
        preds_constraints = preds.flatten()[n:]
        W_pred = torch.nonzero(preds_constraints, as_tuple=True)[0].numpy()
        # start_time_after = time.perf_counter()
        # x_after, lambda_after, _, it_after = self_implement_daqp.daqp_self(H,f_test[i,:],A,b_test[i,:],sense,W_pred) # sense flag 1 if active, 4 if ignore
        # end_time_after = time.perf_counter()
        sense_active = preds.flatten().numpy().astype(np.int32)[n:]

        x,fval,exitflag,info = daqp.solve(H,f_test[i,:],A,b_test[i,:],blower,sense_active)
        lambda_after= list(info.values())[4]
        test_iterations_after[i] = list(info.values())[2]
        test_time_after[i]= list(info.values())[0]

        output_constraints = output.flatten()[n:]
        W_pred_set = set(W_pred)
        W_true_set = set(W_true)
        W_diff = sorted(W_true_set.symmetric_difference(W_pred_set))
        W_diff_FN = sorted(W_true_set.difference(W_pred_set))
        output_FN.extend(output.flatten().numpy()[n:][W_diff_FN])
        # print(f"Difference in W {len(set(W_true)^set(W_pred))}, It before {it_before}, If after {it_after}, It diff {it_before - it_after}")
        
        # print(f" W true {W_true} \n W pred {W_pred}")
        # print(output.flatten().numpy()[n:][W_diff_FN])
        # print(output.flatten().numpy()[n:][W_diff])
        
        #print(output_constraints[W_pred].numpy())
        #print(output.flatten().numpy()[n:])
        # test_iterations_after[i] = it_after
        # test_time_after[i] = end_time_after - start_time_after
        test_iterations_difference[i] = test_iterations_before[i]-test_iterations_after[i]
        
# over graph metrices
test_all_label_graph = np.array(test_all_labels).reshape(-1,n+m)
test_preds_graph = np.array(test_preds).reshape(-1,n+m)

# Compute average over graphs
acc_graph_test = np.mean(np.all(test_all_label_graph == test_preds_graph, axis=1))
test_num_wrongly_pred_nodes_per_graph = np.abs((n+m) - np.sum(test_all_label_graph == test_preds_graph, axis=1)) ## CHECK THIS
print(f'correct:{correct}, total: {total}')
print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')
print(f'Number of graphs: {test_all_label_graph.shape[0]}, Correctly predicted graphs: {np.sum(np.all(test_all_label_graph == test_preds_graph, axis=1))}')
print(f'Graph accuracy of the model on the test data: {100 * acc_graph_test:.2f}%')
print(test_num_wrongly_pred_nodes_per_graph)
print(f'Mean: {np.mean(test_num_wrongly_pred_nodes_per_graph)}')
print(f'Test time before: mean {np.mean(test_time_before)}, min {np.min(test_time_before)}, max {np.max(test_time_before)}')
print(f'Test time after: mean {np.mean(test_time_after)}, min {np.min(test_time_after)}, max {np.max(test_time_after)}')
print(f'Test iter before: mean {np.mean(test_iterations_before)}, min {np.min(test_iterations_before)}, max {np.max(test_iterations_before)}')
print(f'Test iter after: mean {np.mean(test_iterations_after)}, min {np.min(test_iterations_after)}, max {np.max(test_iterations_after)}')
print(f'Test iter reduction: mean {np.mean(test_iterations_difference)}, min {np.min(test_iterations_difference)}, max {np.max(test_iterations_difference)}')
#print(output_FN)
# threshold tuning
# print(best_threshold)
# print(best_mean)

#Boxplot to show reduction
boxplot_time(test_time_before,test_time_after,"time",save = False)
histogram_time(test_time_before, test_time_after,save= False)

boxplot_time(test_iterations_before,test_iterations_after, "iterations",save = False)

barplot_iterations(test_iterations_before,test_iterations_after, "iterations",save = False)

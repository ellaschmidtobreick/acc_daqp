import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def boxplot_time(time_before,time_after, label, save):
    plt.boxplot([time_before,time_after],showfliers=False)
    plt.ylabel(label)
    plt.xticks([1, 2], ['without GNN', 'with GNN'])
    if save == True:
        plt.savefig(f"plots/boxplot_{label}.png")
    plt.show()

        
def barplot_iterations(iterations_before, iterations_after, model_name, save):
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]
    
    # Count occurrences of each iteration count
    before_counts = Counter(iterations_before)
    after_counts = Counter(iterations_after)
    #after_no_ignored_constraints_counts = Counter(iterations_after_no_ignored_constraints)

    # Get all unique iteration numbers
    all_iterations = sorted(set(int(i) for i in before_counts.keys()).union(int(i) for i in after_counts.keys()))
    # fill up the ticks that do not have values
    all_iterations = range(0,np.max(all_iterations)+1,1)
    
    # Prepare values
    before_values = [before_counts.get(it, 0) for it in all_iterations]
    after_values = [after_counts.get(it, 0) for it in all_iterations]

    # Bar width and positions
    x = np.arange(len(all_iterations))
    width = 0.4 

    # Plot bars
    plt.bar(x - width/2, before_values, width=width, label="Without GNN",alpha = 0.7, color=colors[0])
    plt.bar(x + width/2, after_values, width=width, label="With GNN",alpha = 0.7, color=colors[2])

    # Labels and legend
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    
    # xticks
    tick_value = int(np.around((all_iterations[-1]/20)/5, decimals=0)*5)
    #plt.xticks(x, all_iterations,fontsize=7)
    plt.xticks(ticks=x[::5], labels=all_iterations[::5], fontsize=7)
    plt.xlim(x[0] - width, x[-1] + width)
    plt.legend()
    plt.title("Iterations")

    # Save and show
    if save == True:
        plt.savefig(f"plots/bar_it_{model_name}.png")
    plt.show()


# def barplot_iterations_no_ignored_constraints(iterations_after_no_ignored_constraints, iterations_after, label, save):
#     # Count occurrences of each iteration count
#     before_counts = Counter(iterations_after_no_ignored_constraints)
#     after_counts = Counter(iterations_after)
#     #after_no_ignored_constraints_counts = Counter(iterations_after_no_ignored_constraints)

#     # Get all unique iteration numbers
#     all_iterations = sorted(set(int(i) for i in before_counts.keys()).union(int(i) for i in after_counts.keys()))
#     # fill up the ticks that do not have values
#     all_iterations = range(0,np.max(all_iterations)+1,1)
    
#     # Prepare values
#     before_values = [before_counts.get(it, 0) for it in all_iterations]
#     after_values = [after_counts.get(it, 0) for it in all_iterations]
#     #after_values_no_ignored_constraints = [after_no_ignored_constraints_counts.get(it, 0) for it in all_iterations]
#     # Bar width and positions
#     x = np.arange(len(all_iterations))
#     width = 0.4 # 0.25

#     # Plot bars
#     plt.bar(x - width/2, before_values, width=width, label="With GNN, no ignored constraints", color='green')
#     plt.bar(x + width/2, after_values, width=width, label="With GNN", color='orange')
#     # plt.bar(x - width, before_values, width=width, label="Without GNN", color='blue')
#     # plt.bar(x , after_values, width=width, label="With GNN", color='orange')
#     # plt.bar(x + width, after_values_no_ignored_constraints, width=width, label="With GNN, no ignored constraints", color='green')

#     # Labels and legend
#     plt.xlabel("Number of Iterations")
#     plt.ylabel("Frequency")
    
#     # xticks
#     tick_value = int(np.around((all_iterations[-1]/20)/5, decimals=0)*5)
#     #plt.xticks(x, all_iterations,fontsize=7)
#     plt.xticks(ticks=x[::5], labels=all_iterations[::5], fontsize=7)
#     plt.xlim(x[0] - width, x[-1] + width)
#     plt.legend()
#     plt.title(label)

    # # Save and show
    # if save == True:
    #     plt.savefig(f"barplot_{model_name}_no_ignored_constraints.png")
    # plt.show()


def histogram_time(time_before, time_after, model_name,save):
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]
    # Find common bin edges based on the data range
    min_val = min(np.min(time_before), np.min(time_after))
    max_val = 0.0003 # 10v40c: 0.00005 #25v50c: 0.0003
    n_bins = 50  # Set the desired number of bins

    plt.hist(time_before, bins=50,range=(0,max_val),  alpha=0.7, label='without GNN', color=colors[0]) #0.0003 range=(0,max_val),
    plt.hist(time_after, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN', color=colors[2]) #0.00005 range=(0,max_val,),

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time without GNN vs with GNN')
    plt.legend()
    if save == True:
        plt.savefig(f"plots/histo_time_{model_name}.png")
    plt.show()
        
# def histogram_time_no_ignoed_constraints(time_after_no_ignored_constraints, time_after, save):
#     # Find common bin edges based on the data range
#     min_val = min(np.min(time_after_no_ignored_constraints), np.min(time_after))#,np.min(prediction_time))
#     max_val = max(np.max(time_after_no_ignored_constraints), np.max(time_after))#,np.max(prediction_time))
#     n_bins = 50  # Set the desired number of bins
#     bin_edges = np.linspace(min_val, max_val, n_bins+1)

#     plt.hist(time_after_no_ignored_constraints, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN,no ignored components', color='green') #0.0003 range=(0,max_val),
#     plt.hist(time_after, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN', color='orange') #0.00005 range=(0,max_val,),
#     #plt.hist(time_after_no_ignored_constraints, bins=50,range=(0,0.00005),  alpha=0.3, label='with GNN, no ignored constraints', color='green') #0.00005 range=(0,max_val,),

#     #plt.hist(prediction_time, bins = 50, range=(0,max_val), alpha = 0.7, label ='prediction time',color ='green')
#     #plt.hist(prediction_time+time_after, bins = 50, range=(0,max_val), alpha = 0.7, label ='with GNN and prediction time',color ='red')

#     plt.xlabel('Time')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Time with GNN without ignored constraints vs with ignored constraints')
#     plt.legend()
#     if save == True:
#         plt.savefig(f"histogram_time__no_ignored_constraints.png")
#     plt.show()
        
# def hist_output_vs_true_label(output, true_label, save = False):    
#     # Separate predictions based on true label (0 = inactive, 1 = active)
#     output = np.array(output)
#     true_label = np.array(true_label)

#     active_preds = output[true_label == 1]
#     inactive_preds = output[true_label == 0]

#     # Plot histograms
#     plt.figure(figsize=(8, 5))
#     plt.hist(inactive_preds, bins=100, alpha=0.6, label='True inactive nodes', color='blue')
#     plt.hist(active_preds, bins=100, alpha=0.6, label='Ture active nodes', color='orange')
#     plt.xlabel('Predicted output')
#     plt.ylabel('Count')
#     plt.legend()
#     plt.title('Distribution of Output with regards to True Label')
#     plt.yscale('log')
#     plt.tight_layout()
#     if save == True:
#         plt.savefig(f"plots/histogram_output_vs_true_label.png")
#     plt.show()
    
def map_train_acc(train_acc_fixedHA,train_acc_fixedH,train_acc_fixedA,train_acc_flex,figure_name,save = False):
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]
        
    plt.plot(train_acc_fixedHA,color=colors[0], label='fixed H, fixed A')
    plt.plot(train_acc_fixedH,color=colors[1], label='fixed H, flexible A')
    plt.plot(train_acc_fixedA,color=colors[2], label='flexible H, fixed A')
    plt.plot(train_acc_flex,color=colors[3], label='flexible H, flexible A')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy for differently generated H and A')
    plt.legend()
    
    if save == True:
        plt.savefig(f"plots/train_acc_{figure_name}.png")
    plt.show()
    
def histogram_prediction_time(prediction_time,model_name, save = False):
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]
    
    max = 0.01 # 10v40c:0.007 # 25v50c: 0.01
    
    plt.hist(prediction_time, bins=70,range=(np.min(prediction_time),max), alpha=0.7, label='prediction time', color=colors[1])

    plt.xlabel('Prediction time')
    plt.ylabel('Count')
    plt.title('Time of active-set prediction')
    plt.legend()
    
    if save == True:
        plt.savefig(f"plots/histo_pred_time_{model_name}.png")
    plt.show()

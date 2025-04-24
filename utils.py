import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def boxplot_time(time_before,time_after, label, save):
    plt.boxplot([time_before,time_after],showfliers=False)
    plt.ylabel(label)
    plt.xticks([1, 2], ['without GNN', 'with GNN'])
    plt.show()
    if save == True:
        plt.savefig(f"boxplot_{label}.png")
        
def barplot_iterations(iterations_before, iterations_after, label, save):
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
    #after_values_no_ignored_constraints = [after_no_ignored_constraints_counts.get(it, 0) for it in all_iterations]
    # Bar width and positions
    x = np.arange(len(all_iterations))
    width = 0.4 # 0.25

    # Plot bars
    plt.bar(x - width/2, before_values, width=width, label="Without GNN", color='blue')
    plt.bar(x + width/2, after_values, width=width, label="With GNN", color='orange')
    # plt.bar(x - width, before_values, width=width, label="Without GNN", color='blue')
    # plt.bar(x , after_values, width=width, label="With GNN", color='orange')
    # plt.bar(x + width, after_values_no_ignored_constraints, width=width, label="With GNN, no ignored constraints", color='green')

    # Labels and legend
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    
    # xticks
    tick_value = int(np.around((all_iterations[-1]/20)/5, decimals=0)*5)
    #plt.xticks(x, all_iterations,fontsize=7)
    plt.xticks(ticks=x[::5], labels=all_iterations[::5], fontsize=7)
    plt.xlim(x[0] - width, x[-1] + width)
    plt.legend()
    plt.title(label)

    # Save and show
    plt.show()
    if save == True:
        plt.savefig(f"barplot_{label}.png")

def barplot_iterations_no_ignored_constraints(iterations_after_no_ignored_constraints, iterations_after, label, save):
    # Count occurrences of each iteration count
    before_counts = Counter(iterations_after_no_ignored_constraints)
    after_counts = Counter(iterations_after)
    #after_no_ignored_constraints_counts = Counter(iterations_after_no_ignored_constraints)

    # Get all unique iteration numbers
    all_iterations = sorted(set(int(i) for i in before_counts.keys()).union(int(i) for i in after_counts.keys()))
    # fill up the ticks that do not have values
    all_iterations = range(0,np.max(all_iterations)+1,1)
    
    # Prepare values
    before_values = [before_counts.get(it, 0) for it in all_iterations]
    after_values = [after_counts.get(it, 0) for it in all_iterations]
    #after_values_no_ignored_constraints = [after_no_ignored_constraints_counts.get(it, 0) for it in all_iterations]
    # Bar width and positions
    x = np.arange(len(all_iterations))
    width = 0.4 # 0.25

    # Plot bars
    plt.bar(x - width/2, before_values, width=width, label="With GNN, no ignored constraints", color='green')
    plt.bar(x + width/2, after_values, width=width, label="With GNN", color='orange')
    # plt.bar(x - width, before_values, width=width, label="Without GNN", color='blue')
    # plt.bar(x , after_values, width=width, label="With GNN", color='orange')
    # plt.bar(x + width, after_values_no_ignored_constraints, width=width, label="With GNN, no ignored constraints", color='green')

    # Labels and legend
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    
    # xticks
    tick_value = int(np.around((all_iterations[-1]/20)/5, decimals=0)*5)
    #plt.xticks(x, all_iterations,fontsize=7)
    plt.xticks(ticks=x[::5], labels=all_iterations[::5], fontsize=7)
    plt.xlim(x[0] - width, x[-1] + width)
    plt.legend()
    plt.title(label)

    # Save and show
    plt.show()
    if save == True:
        plt.savefig(f"barplot_{label}_no_ignored_constraints.png")

def histogram_time(time_before, time_after, save):
    # Find common bin edges based on the data range
    min_val = min(np.min(time_before), np.min(time_after))#,np.min(prediction_time))
    max_val = max(np.max(time_before), np.max(time_after))#,np.max(prediction_time))
    n_bins = 50  # Set the desired number of bins
    bin_edges = np.linspace(min_val, max_val, n_bins+1)

    plt.hist(time_before, bins=50,range=(0,max_val),  alpha=0.7, label='without GNN', color='blue') #0.0003 range=(0,max_val),
    plt.hist(time_after, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN', color='orange') #0.00005 range=(0,max_val,),
    #plt.hist(time_after_no_ignored_constraints, bins=50,range=(0,0.00005),  alpha=0.3, label='with GNN, no ignored constraints', color='green') #0.00005 range=(0,max_val,),

    #plt.hist(prediction_time, bins = 50, range=(0,max_val), alpha = 0.7, label ='prediction time',color ='green')
    #plt.hist(prediction_time+time_after, bins = 50, range=(0,max_val), alpha = 0.7, label ='with GNN and prediction time',color ='red')

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time without GNN vs with GNN')
    plt.legend()
    plt.show()
    if save == True:
        plt.savefig(f"histogram_time.png")
        
def histogram_time_no_ignoed_constraints(time_after_no_ignored_constraints, time_after, save):
    # Find common bin edges based on the data range
    min_val = min(np.min(time_after_no_ignored_constraints), np.min(time_after))#,np.min(prediction_time))
    max_val = max(np.max(time_after_no_ignored_constraints), np.max(time_after))#,np.max(prediction_time))
    n_bins = 50  # Set the desired number of bins
    bin_edges = np.linspace(min_val, max_val, n_bins+1)

    plt.hist(time_after_no_ignored_constraints, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN,no ignored components', color='green') #0.0003 range=(0,max_val),
    plt.hist(time_after, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN', color='orange') #0.00005 range=(0,max_val,),
    #plt.hist(time_after_no_ignored_constraints, bins=50,range=(0,0.00005),  alpha=0.3, label='with GNN, no ignored constraints', color='green') #0.00005 range=(0,max_val,),

    #plt.hist(prediction_time, bins = 50, range=(0,max_val), alpha = 0.7, label ='prediction time',color ='green')
    #plt.hist(prediction_time+time_after, bins = 50, range=(0,max_val), alpha = 0.7, label ='with GNN and prediction time',color ='red')

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time with GNN without ignored constraints vs with ignored constraints')
    plt.legend()
    plt.show()
    if save == True:
        plt.savefig(f"histogram_time__no_ignored_constraints.png")
        

def histogram_prediction_time(prediction_time,save=False):
    plt.hist(prediction_time, bins=50, alpha=0.7, label='prediction time', color='green')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of prediction time')
    plt.legend()
    plt.show()
    if save == True:
        plt.savefig(f"histogram_prediction_time.png")
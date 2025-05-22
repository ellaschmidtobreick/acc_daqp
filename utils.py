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
    
    plt.xticks(ticks=x[::5], labels=all_iterations[::5], fontsize=7)
    plt.xlim(x[0] - width, x[-1] + width)
    plt.legend()
    plt.title("Iterations")

    # Save and show
    if save == True:
        plt.savefig(f"plots/bar_it_{model_name}.png")
    plt.show()


def histogram_time(time_before, time_after, model_name,save):
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]

    max_val = 0.00005 # 10v40c: 0.00005 #25v50c: 0.0003

    plt.hist(time_before, bins=50,range=(0,max_val),  alpha=0.7, label='without GNN', color=colors[0])
    plt.hist(time_after, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN', color=colors[2])

    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time without GNN vs with GNN')
    plt.legend()
    if save == True:
        plt.savefig(f"plots/histo_time_{model_name}.png")
    plt.show()
        
        
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
    
    max = 0.007 # 10v40c:0.007 # 25v50c: 0.01
    
    plt.hist(prediction_time, bins=70,range=(np.min(prediction_time),max), alpha=0.7, label='prediction time', color=colors[1])

    plt.xlabel('Prediction time in seconds')
    plt.ylabel('Frequency')
    plt.title('Time of active-set prediction')
    plt.legend()
    
    if save == True:
        plt.savefig(f"plots/histo_pred_time_{model_name}.png")
    plt.show()

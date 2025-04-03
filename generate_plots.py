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

    # Get all unique iteration numbers
    all_iterations = sorted(set(int(i) for i in before_counts.keys()).union(int(i) for i in after_counts.keys()))
    
    # Prepare values
    before_values = [before_counts.get(it, 0) for it in all_iterations]
    after_values = [after_counts.get(it, 0) for it in all_iterations]

    # Bar width and positions
    x = np.arange(len(all_iterations))
    width = 0.4

    # Plot bars
    plt.bar(x - width/2, before_values, width=width, label="Without GNN", color='blue')
    plt.bar(x + width/2, after_values, width=width, label="With GNN", color='orange')

    # Labels and legend
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    plt.xticks(x, all_iterations,fontsize=7)
    plt.legend()
    plt.title(label)

    # Save and show
    plt.show()
    if save == True:
        plt.savefig(f"barplot_{label}.png")

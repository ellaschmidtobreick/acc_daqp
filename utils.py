import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib

def boxplot_time(time_before,time_after, label, save):
    plt.rcParams.update({'font.size': 12})
    plt.boxplot([time_before,time_after],showfliers=False)
    plt.ylabel(label)
    plt.xticks([1, 2], ['without GNN', 'with GNN'])
    if save == True:
        plt.savefig(f"plots/boxplot_{label}.png")
    plt.show()

        
def barplot_iterations(iterations_before, iterations_after, model_name, save):
    plt.rcParams.update({'font.size': 12})
    
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
    
    plt.xticks(ticks=x[::5], labels=all_iterations[::5])
    plt.xlim(x[0] - width, x[-1] + width)
    plt.legend()
    #plt.title("Iterations")

    # Save and show
    if save == True:
        plt.savefig(f"plots/bar_it_{model_name}.pdf")
    plt.show()


def histogram_time(time_before, time_after, model_name,save):
    plt.rcParams.update({'font.size': 12})
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]

    max_val = 0.00002 # 10v40c: 0.00005 #25v100c: 0.0003

    plt.hist(time_before, bins=50,range=(0,max_val),  alpha=0.7, label='without GNN', color=colors[0])
    plt.hist(time_after, bins=50,range=(0,max_val),  alpha=0.7, label='with GNN', color=colors[2])

    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency')
    #plt.title('Histogram of Time without GNN vs with GNN')
    plt.legend()
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    plt.gca().xaxis.set_major_formatter(formatter)

    if save == True:
        plt.savefig(f"plots/histo_time_{model_name}.pdf")
    plt.show()
        
        
def map_train_acc(train_acc_fixedHA,train_acc_fixedH,train_acc_fixedA,train_acc_flex,figure_name,save = False):
    plt.rcParams.update({'font.size': 12})
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]
        
    plt.plot(train_acc_fixedHA,color=colors[0], label='fixed H, fixed A')
    plt.plot(train_acc_fixedH,color=colors[1], label='fixed H, flexible A')
    plt.plot(train_acc_fixedA,color=colors[2], label='flexible H, fixed A')
    plt.plot(train_acc_flex,color=colors[3], label='flexible H, flexible A')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.title('Training accuracy for differently generated H and A')
    plt.legend()
    
    if save == True:
        plt.savefig(f"plots/train_acc_{figure_name}.pdf")
    plt.show()
    
def histogram_prediction_time(prediction_time,model_name, save = False):
    plt.rcParams.update({'font.size': 12})
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, 4)]
    
    max = 0.01 # 10v40c:0.007 # 25v50c: 0.01
    
    plt.hist(prediction_time, bins=70,range=(np.min(prediction_time),max), alpha=0.7, label='prediction time', color=colors[1])

    plt.xlabel('Prediction time in seconds')
    plt.ylabel('Frequency')
    #plt.title('Time of active-set prediction')
    plt.legend()
    
    if save == True:
        plt.savefig(f"plots/histo_pred_time_{model_name}.pdf")
    plt.show()


def pareto_front(points):
    """
    points: list of (x, y) pairs
    returns the indices of the Pareto-optimal points
    """
    points = np.array(points)
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, (x, y) in enumerate(points):
        if is_pareto[i]:
            # dominated if another point is <= in both and strictly < in one
            is_pareto[is_pareto] = np.any(points[is_pareto] < [x, y], axis=1) | np.all(points[is_pareto] == [x, y], axis=1)
            is_pareto[i] = True  # keep current
    return np.where(is_pareto)[0]

def plot_pareto(points, labels, file_name="plots/pareto_plot_test.pdf"):
    plt.figure(figsize=(10, 6))  # increase figure size (width x height in inches)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for the legend
    points = np.array(points)
    idx = pareto_front(points)

    # Extract model_type and variant from the label tuples
    model_types = [lbl[0] for lbl in labels]
    variants = [lbl[1] for lbl in labels]

    # Assign colors per model type
    unique_types = sorted(set(model_types))
    cmap = plt.cm.tab10
    type_to_color = {t: cmap(i % 10) for i, t in enumerate(unique_types)}

    # Assign markers per variant
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]
    unique_variants = sorted(set(variants))
    variant_to_marker = {v: markers[i % len(markers)] for i, v in enumerate(unique_variants)}

    # Plot all points
    for (x, y), t, v in zip(points, model_types, variants):
        plt.scatter(
            x, y,
            color=type_to_color[t],
            marker=variant_to_marker[v],
            alpha=0.8, s=70, edgecolor="k",
            label=f"{t}-{v}"
        )

    # Plot all points
    # for (x, y), label in zip(points, labels):
    #     plt.scatter(x, y, color=label_to_color[label], label=label, alpha=0.7, s=70, edgecolor="k")

    # Plot Pareto front (sorted for line plotting)
    pareto_points = points[idx]
    pareto_points = pareto_points[np.argsort(pareto_points[:,0])]
    plt.plot(pareto_points[:,0], pareto_points[:,1], "r--", linewidth=2, label="Pareto front")

    # Create legend handles for model types (colors)
    type_handles = [mpatches.Patch(color=type_to_color[t], label=t) for t in unique_types]
    # Create legend handles for variants (markers)
    variant_handles = [mlines.Line2D([], [], color='k', marker=variant_to_marker[v], linestyle='None', markersize=8, label=str(v)) for v in unique_variants]

    # Combine legends
    plt.legend(handles=type_handles + variant_handles, loc='upper left', title="Model Type & Layer Width")

    plt.yscale("log")
    plt.xscale("log")
    # plt.xlim([np.min(points)*0.9, np.max(points)*1.1])
    # plt.ylim([np.min(points)*0.9, np.max(points)*1.1])
    plt.ylabel("Prediction time")
    plt.xlabel("Solve time")
    plt.title("Prediction vs Solve Time (Pareto Front)")
    plt.grid(True)
    plt.savefig(file_name)
    plt.savefig(file_name.replace(".pdf", ".png"))
    plt.show()

def plot_scaling(points, labels, file_name="plots/scaling_plot_test.pdf"):
    plt.figure(figsize=(10, 6))  # increase figure size (width x height in inches)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for the legend
    points = np.array(points)
    idx = pareto_front(points)

    # Extract model_type and variant from the label tuples
    model_types = [lbl[0] for lbl in labels]
    variants = [lbl[1] for lbl in labels]

    # Assign colors per model type
    unique_types = sorted(set(model_types))
    cmap = plt.cm.tab10
    type_to_color = {t: cmap(i % 10) for i, t in enumerate(unique_types)}

    # Plot all points
    for (x, y), t in zip(points, model_types):
        plt.scatter(
            x, y,
            color=type_to_color[t],
            alpha=0.8, s=70, edgecolor="k",
            label=f"{t}"
        )

    # Plot all points
    # for (x, y), label in zip(points, labels):
    #     plt.scatter(x, y, color=label_to_color[label], label=label, alpha=0.7, s=70, edgecolor="k")

    # Create legend handles for model types (colors)
    type_handles = [mpatches.Patch(color=type_to_color[t], label=t) for t in unique_types]
    # Combine legends
    plt.legend(handles=type_handles, loc='upper left', title="Model Type")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([np.min(points)*0.9, np.max(points)*1.1])
    plt.ylim([np.min(points)*0.9, np.max(points)*1.1])
    plt.ylabel("Prediction time")
    plt.xlabel("Solve time")
    plt.title("Prediction vs Solve Time with scaling size")
    plt.grid(True)
    plt.savefig(file_name)
    plt.savefig(file_name.replace(".pdf", ".png"))
    plt.show()

def plot_scaling2(points, labels, file_name="plots/scaling_plot_test.pdf"):
    plt.rcParams.update({'font.size': 12})

    points = np.array(points)
    labels = np.array(labels)[:, 0]

    model_types = np.unique(labels)
    colors = ["blue", "orange"]

    fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
    if len(model_types) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_types):
        ax.set_title(model)
        ax.set_xlabel("Total graph size")
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

        # use ax, not plt
        ax.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        ax.grid(which="major", linestyle="-", linewidth=0.7)
        ax.grid(which="minor", visible=False)

        model_mask = labels == model
        model_points = points[model_mask]

        ax.scatter(np.arange(len(model_points)) * 5, model_points[:, 0], color=colors[1], label="Solving Time")
        if model != "Non-learned":
            ax.scatter(np.arange(len(model_points)) * 5, model_points[:, 1], color=colors[0], label="Prediction Time")

    axes[0].set_ylabel("Time (seconds)")
    axes[0].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(file_name)
    plt.savefig(file_name.replace(".pdf", ".png"))
    # plt.show()

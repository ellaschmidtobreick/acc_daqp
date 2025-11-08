import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib

colors = [
    "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C",
    "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00",
    "#CAB2D6", "#6A3D9A"
]


def boxplot_time(time_before,time_after, label, save):
    plt.rcParams.update({'font.size': 12})
    plt.boxplot([time_before,time_after],showfliers=False)
    plt.ylabel(label)
    plt.xticks([1, 2], ['Cold-starting', 'Warm-starting with GNN'])
    if save == True:
        plt.savefig(f"plots/boxplot_{label}.png")
    plt.show()

        
def barplot_iterations(iterations_before, iterations_after, model_name, save):
    matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18})
    
    # cmap = plt.get_cmap("viridis")
    # colors = [cmap(i) for i in np.linspace(0, 1, 4)]
    
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
    plt.bar(x - width/2, before_values, width=width, label="Cold-starting",alpha = 0.9, color="#6A3D9A")
    plt.bar(x + width/2, after_values, width=width, label="Warm-starting with GNN",alpha = 0.7, color="#33A02C")

    
    # Calculate quartiles
    # q2_bef = np.percentile(iterations_before, 50)
    # q2_aft = np.percentile(iterations_after, 50)
    q2_bef = np.mean(iterations_before)
    q2_aft = np.mean(iterations_after)
    q3_bef = np.percentile(iterations_before, 90)
    q3_aft = np.percentile(iterations_after, 90)
    q1_bef = np.percentile(iterations_before, 10)
    q1_aft = np.percentile(iterations_after, 10)


    # Quartiles and percentiles
    plt.axvline(x=q2_bef, color="#CAB2D6", linestyle='--')
    plt.axvline(x=q2_aft, color="#B2DF8A", linestyle='--')
    # Quartiles and percentiles
    plt.axvline(x=q1_bef, color="#CAB2D6", linestyle='-.')
    plt.axvline(x=q1_aft, color="#B2DF8A", linestyle='-.')
    plt.axvline(x=q3_bef, color="#CAB2D6", linestyle='-.')
    plt.axvline(x=q3_aft, color="#B2DF8A", linestyle='-.')

    # Labels and legend
    plt.xlabel("Number of Iterations")
    plt.ylabel("Frequency")
    plt.ylim(0, 40)

    #plt.xticks(ticks=x[::5], labels=all_iterations[::5])
    plt.xlim(x[0] - width, x[-1] + width)
    plt.legend()
    #plt.title("Iterations")
    plt.tight_layout()
    # Save and show
    if save == True:
        plt.savefig(f"plots/bar_it_{model_name}.pdf")
    plt.show()


def barplot_iter_reduction(iterations_reduction, model_name, save):
    matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18    })

    # Count occurrences of each iteration count
    red_counts = Counter(iterations_reduction)

    # Get all unique iteration numbers
    all_iterations = sorted(set(int(i) for i in red_counts.keys()))
    
    # fill up the ticks that do not have values
    all_iterations = range(np.min(all_iterations)-1,np.max(all_iterations)+1,1)
    # Prepare values
    red_values = [red_counts.get(it, 0) for it in all_iterations]

    # Bar width and positions
    width = 0.9
    # Plot bars
    plt.bar(all_iterations , red_values, width=width,alpha = 0.9, color="#1F78B4")

    # Labels and legend
    plt.xlabel("Iterations")
    plt.ylabel("Frequency")
    
    #plt.xticks(ticks=x[::5], labels=all_iterations[::5])
    plt.xlim(all_iterations[0] - width, all_iterations[-1] + width)

    # Save and show
    if save == True:
        plt.savefig(f"plots/bar_it_red_{model_name}.pdf")
    plt.show()

def histogram_time(time_before, time_after, model_name,save):
    matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18    })
    # cmap = plt.get_cmap("viridis")
    # colors = [cmap(i) for i in np.linspace(0, 1, 4)]

    max_val = np.max([np.max(time_before), np.max(time_after)]) + 0.000001 # 10v40c: 0.00005 #25v100c: 0.0003
    min_val = np.min([np.min(time_before), np.min(time_after)]) - 0.000001
    print(f"max val: {max_val}, min val: {min_val}") # max = 0.003 for 50v
    plt.hist(time_before, bins=30,range=(0.000007,0.00003),  alpha=0.9, label='Cold-starting', color="#6A3D9A")
    plt.hist(time_after, bins=30,range=(0.000007,0.00003),  alpha=0.7, label='Warm-starting with GNN', color="#33A02C")

    # Calculate quartiles
    q2_bef = np.percentile(time_before, 50)
    q2_aft = np.percentile(time_after, 50)
    q3_bef = np.percentile(time_before, 90)
    q3_aft = np.percentile(time_after, 90)

    # Quartiles and percentiles
    # plt.axvline(x=q2_bef, color="#CAB2D6", linestyle='--')
    # plt.axvline(x=q2_aft, color="#B2DF8A", linestyle='--')
  
    plt.axvline(x=q3_bef, color="#CAB2D6", linestyle='-.')
    plt.axvline(x=q3_aft, color="#B2DF8A", linestyle='-.')
  
    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency')
    plt.ylim(0, 40)
    #plt.title('Histogram of Time without GNN vs with GNN')
    plt.legend()
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()

    if save == True:
        plt.savefig(f"plots/histo_time_{model_name}.pdf")
    plt.show()

def histogram_iterations(iterations_before, iterations_after, model_name, save):
    matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18    })

    max_val = np.max([np.max(iterations_before), np.max(iterations_after)]) + 2
    min_val = np.min([np.min(iterations_before), np.min(iterations_after)])
    print(f"max val: {max_val}, min val: {min_val}")
    plt.hist(iterations_before, bins=20,range=(min_val,max_val),  alpha=0.9, label='Cold-starting', color="#6A3D9A")
    plt.hist(iterations_after, bins=20,range=(min_val,max_val),  alpha=0.7, label='Warm-starting with GNN', color="#33A02C")

    # Calculate quartiles
    q2_bef = np.mean(iterations_before)
    q2_aft = np.mean(iterations_after)
    q3_bef = np.percentile(iterations_before, 90)
    q3_aft = np.percentile(iterations_after, 90)

    # Quartiles and percentiles
    # plt.axvline(x=q2_bef, color="#CAB2D6", linestyle='--')
    # plt.axvline(x=q2_aft, color="#B2DF8A", linestyle='--')
    plt.axvline(x=q3_bef, color="#CAB2D6", linestyle='-.')
    plt.axvline(x=q3_aft, color="#B2DF8A", linestyle='-.')

    plt.xlabel('Number of iterations')
    plt.ylabel('Frequency')
    plt.ylim(0, 40)
    #plt.title('Histogram of Time without GNN vs with GNN')
    plt.legend()
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()

    if save == True:
        plt.savefig(f"plots/histo_iter_{model_name}.pdf")
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
    
    max = 18 # 10v40c:0.007 # 25v50c: 0.01
    
    plt.hist(prediction_time, bins=70,range=(np.min(prediction_time),max), alpha=0.7, label='prediction time', color=colors[1])

    plt.xlabel('Prediction time in seconds')
    plt.ylabel('Frequency')
    #plt.title('Time of active-set prediction')
    plt.legend()
    plt.tight_layout()

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


def plot_scaling(points, labels, file_name="plots/scaling_plot_test"):
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 22,
        "font.size": 22,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18    })

    points = np.array(points)
    labels = np.array(labels)[:, 0]

    model_types = np.unique(labels)
    model_types = [model_types[i] for i in [1, 2, 0]]
    # colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    colors = ['#e31a1c','#fb9a99','#6a3d9a','#cab2d6']

    fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
    if len(model_types) == 1:
        axes = [axes]

    enum = ["(a)","(b)","(c)"]
    for i, model in enumerate(model_types):
        ax = axes[i]
        model_description = model
        model_description = model_description.replace("Non-learned","Cold-started")

        #ax.set_title(model)
        ax.text(0.5,-0.35, f"{enum[i]} {model}",ha="center", transform=ax.transAxes)

        model_mask = labels == model
        model_points = points[model_mask]
        x_points = np.arange(len(model_points)+1)[1:] * 50

        # Solving time
        ax.plot(x_points, model_points[:, 0], color=colors[0], label="Solving Time")
        ax.fill_between(x_points, model_points[:, 0] - model_points[:, 2], model_points[:, 0] + model_points[:, 2], color=colors[1], alpha=0.3)

        # Prediction time 
        if model != "Cold-started":
            ax.plot(x_points, model_points[:, 1], color=colors[2], label="Prediction Time")
            ax.fill_between(x_points, model_points[:, 1] - model_points[:, 3], model_points[:, 1] + model_points[:, 3], color=colors[3], alpha=0.3)
        
        ax.set_xlabel("Total graph size")
        ax.set_yscale("log")
        ax.set_yticks([ 1e-5, 1e-4, 1e-3, 1e-2,1e-1])

        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(which="major", linestyle="-", linewidth=0.7)
        ax.grid(which="minor", visible=False)

        if i == 0:
            ax.set_ylabel("Time (seconds)")
            ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf")
    plt.savefig(f"{file_name}.png")
    plt.show()

def plot_scaling_iterations(iterations, labels, file_name="plots/scaling_plot_iterations_test"):
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 20,
        "font.size": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16    })

    iterations = np.array(iterations)
    labels = np.array(labels)[:, 0]
    model_types = np.unique(labels)
    # colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
    colors = ['#1f78b4','#33a02c','#ff7f00','#a6cee3','#b2df8a','#fdbf6f']
    fig, ax = plt.subplots(1, 1,figsize=(6.4, 4.8),  sharey=True)

    #ax.set_title("Iterations after prediction")
    ax.set_xlabel("Total graph size")
    ax.set_yscale("log")
    
    ax.grid(which="major", linestyle="-", linewidth=0.7)
    ax.grid(which="minor", visible=False)
    mapping = {
        'Cold-started': 'Cold-start',
        'GNN': 'Warm-start GNN',
        'MLP': 'Warm-start MLP'
    }

    legend_labels = [mapping[m] for m in model_types]
    for i, model in enumerate(model_types):
        model_mask = labels == model
        model_iterations = iterations[model_mask]
        
        x_points = np.arange(len(model_iterations)+1)[1:] * 50
        #ax.scatter(np.arange(len(model_iterations)+1)[1:] * 50, model_iterations[:], color=colors[i], label=model)
        ax.plot(x_points, model_iterations[:, 0], color=colors[i], label=legend_labels[i])
        ax.fill_between(x_points, model_iterations[:, 0] - model_iterations[:, 1], model_iterations[:, 0] + model_iterations[:, 1], color=colors[i+3], alpha=0.3)
        
    ax.set_ylabel("Iterations")
    ax.legend(loc="lower right",)
    ax.set_yticks([1e1,1e2,1e3])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    plt.tight_layout()
    plt.savefig(f"{file_name}.pdf")
    plt.savefig(f"{file_name}.png")
    plt.show()

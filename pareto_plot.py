import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from train_model import train_GNN, train_MLP
from test_model import test_GNN, test_MLP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

n = [10]
m = [40]
nth = 7
seed = 123
data_points = 2000 #5000
lr = 0.001
number_of_max_epochs = 100
layer_width = 128 # vary
number_of_layers = 3     # vary
track_on_wandb = False #True
t = 0.9 #0.6 # vary


conv_types = ["GAT", "LEConv"]
layer_width = [64, 128,256]
number_of_layers = [3, 4,5]


prediction_time_vector , solving_time_vector, label_vector = [], [], []


for j in layer_width:
    for k in number_of_layers:
        for i in conv_types:
            print(f"--- GNN, Conv: {i}, Layer width: {j}, Number of layers: {k} ---")
            train_GNN(n,m,nth, seed, data_points,lr,number_of_max_epochs,j,k, track_on_wandb,t, False,False,"model_10v_40c_pareto",dataset_type="standard", conv_type=i)
            prediction_time, test_time_after = test_GNN(n,m,nth, seed, data_points,j,k,t, False,False,"model_10v_40c_pareto",dataset_type="standard",conv_type=i) 
            prediction_time_vector.append(prediction_time)
            solving_time_vector.append(test_time_after)
            label_vector.append((f"GNN - {i}", f"{k} layers"))
            print()

        print(f"--- MLP, Layer width: {j}, Number of layers: {k} ---")
        train_MLP(n,m,nth, seed, data_points,lr,number_of_max_epochs,j,k, track_on_wandb,t, False,False,"MLP_model_10v_40c_pareto",dataset_type="standard")
        prediction_time, test_time_after = test_MLP(n,m,nth, seed, data_points,j,k,t, False,False,"MLP_model_10v_40c_pareto",dataset_type="standard")
        prediction_time_vector.append(prediction_time)
        solving_time_vector.append(test_time_after)
        label_vector.append(("MLP",f"{k} layers"))
        print()

points = list(zip(solving_time_vector,prediction_time_vector))
labels = label_vector
colors = ["blue"] * (len(points)-1) + ["green"]


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

def plot_pareto(points, labels, colors=None):
    plt.figure(figsize=(10, 6))  # increase figure size (width x height in inches)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space on the right for the legend
    points = np.array(points)
    idx = pareto_front(points)

    # Extract model_type and variant from the label tuples
    model_types = [lbl[0] for lbl in label_vector]
    variants = [lbl[1] for lbl in label_vector]

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


    plt.ylabel("Prediction time")
    plt.xlabel("Solve time")
    plt.title("Prediction vs Solve Time (Pareto Front)")
    plt.grid(True)
    plt.savefig(f"plots/pareto_plot_dense2.png")
    plt.savefig(f"plots/pareto_plot_dense2.pdf")
    plt.show()


# --- Example usage ---
# points = [(0.1, 1.2), (0.2, 0.9), (0.15, 1.0), (0.25, 0.8), (0.3, 0.95)]
# labels = ["MLP-1", "MLP-2", "GNN-1", "GNN-2", "GNN-3"]
# colors = ["blue", "blue", "orange", "orange", "orange"]

plot_pareto(points, labels)

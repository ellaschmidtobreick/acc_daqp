import numpy as np
import pickle
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib

# Load data
with open("./data/scaling_data_std.pkl", "rb") as f:
    points_loaded, labels_loaded,iterations_after_loaded = pickle.load(f) # ,iterations_after_loaded


matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16    })

iterations = np.array(iterations_after_loaded)
labels = np.array(labels_loaded)[:, 0]

model_types = np.unique(labels)
# colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
colors = ['#1f78b4','#33a02c','#ff7f00','#a6cee3','#b2df8a','#fdbf6f']
fig, ax = plt.subplots(1, 1,figsize=(6.4, 4.8),  sharey=True)

#ax.set_title("Iterations after prediction")
ax.set_xlabel("Total graph size")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Iterations")
ax.legend(loc="lower right",)
ax.set_yticks([1e1,1e2,1e3])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

ax.grid(which="major", linestyle="-", linewidth=0.7)
ax.grid(which="minor", visible=False)

legend_labels = ["Warm-start GNN","Warm-start MLP","Cold-start"]
x = np.arange(0,201,10)[1:]*5

for i, model in enumerate(model_types):
    model_mask = labels == model
    model_iterations = iterations[model_mask]

    x_points = np.arange(len(model_iterations)+1)[1:] * 50
    ax.plot(x_points, model_iterations[:, 0], color=colors[i], label=legend_labels[i])
    ax.fill_between(x_points, model_iterations[:, 0] - model_iterations[:, 1], model_iterations[:, 0] + model_iterations[:, 1], color=colors[i+3], alpha=0.3)
    

    ax.plot(x, x,color = "green", label ="x")
    ax.plot(x, 0.01*x**2,color = "orange", label ="x^2")
    ax.plot(x,x**0.5,color = "red", label ="x^1/2")
    ax.plot(x,x**0.66,color = "pink", label ="x^2/3")





plt.tight_layout()
plt.savefig("./plots/scaling_plot_slopes_iter_log_log_server.png")
plt.savefig("./plots/scaling_plot_slopes_iter_log_log_server.pdf")
plt.show() 
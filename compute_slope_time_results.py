import numpy as np
import pickle
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load data
with open("./data/scaling_data.pkl", "rb") as f:
    points_loaded, label_vector_loaded = pickle.load(f)


points = np.array(points_loaded)
labels = np.array(label_vector_loaded)[:, 0]
lengths = np.unique(np.array(label_vector_loaded)[:, 1])[1:]
#length_vector =  np.unique(lengths)[1:]
model_types = np.unique(labels)

# Sum variables and constraints to get graph size
def sum_numbers(s):
    numbers = list(map(int, re.findall(r'\d+', s)))
    return sum(numbers)

graph_sizes = np.sort(np.array([sum_numbers(s) for s in lengths]))

def linear(x, a, b):
    return a*x + b

def exp_func(x, a, b):
    return a*np.exp(b*x.astype(np.float64))


fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
x_scaled = graph_sizes/ np.max(graph_sizes)  # scales x to [0, 1]
colors = ["blue", "orange"]

for ax, model in zip(axes, model_types):

    model_mask = labels == model
    model_points = points[model_mask]

    print(model)

    if len(model_types) == 1:
        axes = [axes]


    ax.set_title(model)
    ax.set_xlabel("Total graph size")

    # params_l, covariance_l = curve_fit(linear, x_scaled, model_points[:,0])
    # print("Fitted parameters (a, b):", params_l)
    params_e, covariance_e = curve_fit(exp_func, x_scaled, model_points[:,0])
    print("Parameters fitted to scaled x")
    print("Fitted exponential parameters for solving time (a, b):", params_e)

    ax.scatter(graph_sizes, model_points[:,0], label="Solving time", color =colors[0])
    ax.plot(graph_sizes, exp_func(x_scaled, *params_e), label="Fitted solving time", color="green")

    if model != "Non-learned":

        # Fit linear model
        params_pred_l, covariance_pred_l = curve_fit(linear, x_scaled, model_points[:,1])
        print("Fitted linear parameters for prediction time (a, b):", params_pred_l)
        # params_pred_e, covariance_pred_e = curve_fit(exp_func, x_scaled, model_points[:,1])
        # print("Fitted parameters (a, b):", params_pred_e)

        ax.scatter(graph_sizes, model_points[:,1], label="Prediction time",color =colors[1])
        ax.plot(graph_sizes, linear(x_scaled, *params_pred_l), label="Fitted prediction time", color="red")
    print()
    ax.legend()



axes[0].set_ylabel("Time (seconds)")
axes[0].legend(loc="upper left")

plt.tight_layout()
plt.savefig("./plots/scaling_plot_slopes.png")
plt.savefig("./plots/scaling_plot_slopes.pdf")
plt.show() 

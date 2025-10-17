import numpy as np
import pickle
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from utils import plot_scaling2 
import matplotlib

# Load data
with open("./data/scaling_data_big_test.pkl", "rb") as f:
    points_loaded, label_vector_loaded = pickle.load(f)


# points = np.array(points_loaded)
# labels = np.array(label_vector_loaded)[:, 0]
# lengths = np.unique(np.array(label_vector_loaded)[:, 1])[1:]
# #length_vector =  np.unique(lengths)[1:]
# model_types = np.unique(labels)

# # Sum variables and constraints to get graph size
# def sum_numbers(s):
#     numbers = list(map(int, re.findall(r'\d+', s)))
#     return sum(numbers)

# graph_sizes = np.sort(np.array([sum_numbers(s) for s in lengths]))

# def linear(x, a, b):
#     return a*x + b

# def exp_func(x, a, b):
#     return a*np.exp(b*x.astype(np.float64))


# fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
# x_scaled = graph_sizes/ np.max(graph_sizes)  # scales x to [0, 1]
# colors = ["blue", "orange"]

# for ax, model in zip(axes, model_types):

#     model_mask = labels == model
#     model_points = points[model_mask]

#     print(model)

#     if len(model_types) == 1:
#         axes = [axes]


#     ax.set_title(model)
#     ax.set_xlabel("Total graph size")

#     # params_l, covariance_l = curve_fit(linear, x_scaled, model_points[:,0])
#     # print("Fitted parameters (a, b):", params_l)
#     params_e, covariance_e = curve_fit(exp_func, x_scaled, model_points[:,0])
#     print("Parameters fitted to scaled x")
#     print("Fitted exponential parameters for solving time (a, b):", params_e)

#     ax.scatter(graph_sizes, model_points[:,0], label="Solving time", color =colors[1])
#     ax.plot(graph_sizes, exp_func(x_scaled, *params_e), label="Fitted solving time", color="red")

#     if model != "Non-learned":

#         # Fit linear model
#         params_pred_l, covariance_pred_l = curve_fit(linear, x_scaled, model_points[:,1])
#         print("Fitted linear parameters for prediction time (a, b):", params_pred_l)
#         # params_pred_e, covariance_pred_e = curve_fit(exp_func, x_scaled, model_points[:,1])
#         # print("Fitted parameters (a, b):", params_pred_e)

#         ax.scatter(graph_sizes, model_points[:,1], label="Prediction time",color =colors[0])
#         ax.plot(graph_sizes, linear(x_scaled, *params_pred_l), label="Fitted prediction time", color="green")
#     print()
#     ax.legend()



# axes[0].set_ylabel("Time (seconds)")
# axes[0].legend(loc="upper left")

# plt.tight_layout()
# plt.savefig("./plots/scaling_plot_slopes.png")
# plt.savefig("./plots/scaling_plot_slopes.pdf")
# plt.show() 

plt.rcParams.update({'font.size': 14})

points = np.array(points_loaded)
labels = np.array(label_vector_loaded)[:, 0]

model_types = np.unique(labels)
colors = ["blue", "orange"]

fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
if len(model_types) == 1:
    axes = [axes]

x = np.arange(0,201,10)[1:]*5

for ax, model in zip(axes, model_types):
    ax.set_title(model)
    ax.set_xlabel("Total graph size")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # use ax, not plt
    ax.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    ax.grid(which="major", linestyle="-", linewidth=0.7)
    ax.grid(which="minor", visible=False)

    model_mask = labels == model
    model_points = points[model_mask]

    ax.scatter(np.arange(len(model_points)) * 50, model_points[:, 0], color=colors[1], label="Solving Time")
    if model != "Non-learned":
        ax.scatter(np.arange(len(model_points)) * 50, model_points[:, 1], color=colors[0], label="Prediction Time")
    if model =="GNN":
        ax.plot(x,[0.001 for i in range(len(x))],color = "blue", label ="x^0")
        ax.plot(x,0.00000001 * x**2,color = "yellow", label ="x^2")
        ax.plot(x,0.0000000001 * x**3,color = "red", label ="x^3")

    if model == "MLP":
        ax.plot(x,[0.0001 for i in range(len(x))],color = "green", label ="x^0")
        ax.plot(x,0.0000000001 * x**3,color = "red", label ="x^3")
        ax.plot(x,0.00000001 * x**2,color = "yellow", label ="x^2")

    if model == "Non-learned":
        ax.plot(x,0.00000001 * x**2,color = "yellow", label ="x^2")
        ax.plot(x,0.0000000001 * x**3,color = "red", label ="x^3")
        ax.plot(x,0.0000000001 * x**(2.5),color = "orange", label ="x^2.5")
    # ax.plot(x,x**20,color = "red", label ="x^20")
    # ax.plot(x,0.0001 * x,color = "purple", label ="x")
    # ax.plot(x,0.0000001 * x**2,color = "yellow", label ="x^2")
    # ax.plot(x,0.0001 * x**5,color = "orange", label ="x^4")
    # ax.plot(x,0.0001 * x**3,color = "red", label ="x^3")
    # ax.plot(x,0.001 * x**(1/2),color = "green", label ="x^(1/2)")
    # ax.plot(x,0.0001 * x**(1/2),color = "green", label ="x^(1/2)")
    # ax.plot(x,0.0001 * x**(1/3),color = "blue", label ="x^(1/3)")
    # ax.plot(x,0.001 * x**(1/3),color = "blue", label ="x^(1/3)")
axes[0].set_ylabel("Time (seconds)")
axes[0].legend(loc="upper left")

plt.tight_layout()
plt.savefig("./plots/scaling_plot_slopes_log_log_server.png")
plt.savefig("./plots/scaling_plot_slopes_log_log_server.pdf")
plt.show() 
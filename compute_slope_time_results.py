import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

# Load data
with open(".\data\scaling_data_std_server_sparse.pkl", "rb") as f:
    points_loaded, labels_loaded,iterations_after_loaded = pickle.load(f)

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18    })

points = np.array(points_loaded)
labels = np.array(labels_loaded)[:, 0]

model_types = np.unique(labels)
model_types = [model_types[i] for i in [1, 2, 0]]
colors = ['#1f78b4','#33a02c','#ff7f00','#a6cee3','#b2df8a','#fdbf6f']
#fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
fig, axes = plt.subplots(1, 2, figsize=(5 * 2, 5), sharey=True)

if len(model_types) == 1:
    axes = [axes]

x = np.arange(0,201,10)[1:]*5

enum = ["(a)","(b)","(c)"]
for i, model in enumerate(model_types):
    #ax = axes[i]
    model_description = model
    model_description = model_description.replace("Non-learned","Cold-started")

    #ax.set_title(model)

    model_mask = labels == model
    model_points = points[model_mask]
    x_points = np.arange(len(model_points)+1)[1:] * 50

    # ax.plot(x,model_points[0, 1]*(x/x_points[0]),color = "yellow", label ="$x^2$")
    # ax.plot(x,[model_points[0, 1] for b in range(len(x))],color = "cyan", label ="$x$")

    #ax.plot(x,model_points[0, 0]*(x/x_points[0]),color = "red", label ="x^3")
    #ax.plot(x,model_points[0, 0]*(x/x_points[0]),color = "orange", label ="x^2.5")

    # ax.plot(x, model_points[0, 0]*(x/x_points[0]),color = "green", label ="$x$")
    # ax.plot(x, model_points[0, 0]*(x/x_points[0])**2,color = "orange", label ="$x^2$")
    # ax.plot(x,model_points[0, 0]*(x/x_points[0])**3,color = "blue", label ="$x^3$")
    # ax.plot(x,model_points[0, 0]*(x/x_points[0])**2.5,color = "pink", label ="$x^2.5$")

    if i == 0:
        axes[0].set_ylabel("Time (seconds)")
        axes[0].text(0.5,-0.35, f"{enum[0]} Solve Time",ha="center", transform=axes[0].transAxes)          
        axes[1].text(0.5,-0.35, f"{enum[1]} Prediction Time",ha="center", transform=axes[1].transAxes)

    for j,ax in enumerate(axes):
        if j==0:
            # Solve time
            axes[j].plot(x_points, model_points[:, 0], color=colors[0+i], label="Solve Time")
            axes[j].fill_between(x_points, model_points[:, 0] - model_points[:, 2], model_points[:, 0] + model_points[:, 2], color=colors[3+i], alpha=0.3)


        if j==1:
            # Prediction time 
            if model != "Cold-started":
                axes[j].plot(x_points, model_points[:, 1],'--', color=colors[0+i], label="Prediction Time")
                axes[j].fill_between(x_points, model_points[:, 1] - model_points[:, 3], model_points[:, 1] + model_points[:, 3],color=colors[3+i], alpha=0.3)
            


        axes[j].set_xlabel("Total graph size")
        axes[j].set_yscale("log")
        #axes[j].set_xscale("log")

        axes[j].set_yticks([ 1e-5, 1e-4, 1e-3, 1e-2,1e-1])
        axes[j].set_xlim(0,1000)

        axes[j].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axes[j].grid(which="major", linestyle="-", linewidth=0.7)
        axes[j].grid(which="minor", visible=False)

plt.tight_layout()

# Add shared legend above the plots
legend_elements = [
    matplotlib.lines.Line2D([0], [0], color=colors[0], label='Graph Neural Network'),
    matplotlib.lines.Line2D([0], [0], color=colors[1], label='Multilayer Perceptron'),
    matplotlib.lines.Line2D([0], [0], color=colors[2], label='Cold-started'),

]
fig.legend(handles=legend_elements, loc='upper center', ncol=3,
           bbox_to_anchor=(0.51, 1.06), fontsize=18, framealpha=0.8)

plt.savefig("./plots/scaling_plot_std_log_camera_ready.pdf", bbox_inches='tight')
plt.savefig("./plots/scaling_plot_std_log_camera_ready.png", bbox_inches='tight')
plt.show()
# plt.savefig("./plots/scaling_plot_slopes_time_log_log_server1.pdf")
# plt.savefig("./plots/scaling_plot_slopes_time_log_log_server1.png")
plt.show()


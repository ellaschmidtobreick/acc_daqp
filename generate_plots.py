import pickle
from utils import histogram_iterations, plot_scaling, plot_scaling_iterations, plot_scaling_one_plot, histogram_time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "font.size": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})

##### Figure 4: scaling of input QP size for GNN warm-starting #####

with open("./data/scaling_data_std_one_GNN.pkl", "rb") as f:
    points_loaded, labels_loaded,iterations_after_loaded = pickle.load(f)

plot_scaling(points_loaded, labels_loaded,"plots/scaling_plot_std_test1",save = False)
plot_scaling_one_plot(points_loaded, labels_loaded,"plots/scaling_plot_std_log_log_one_plot", save=False)
plot_scaling_iterations(iterations_after_loaded, labels_loaded,"plots/scaling_plot_iterations_std_test1", save=False)

##### Figure 5 in the paper: histogram of iterations and solver time for scaling experiment #####

n_train = [20, 40, 60]
m_train = [40,80,120]
n_test = [100]
m_test = [200]
model_name = f"model_{n_train}v_{m_train}c_multi"

with open(f"data/multi_experiment_{n_train}v_{m_train}c_test_{n_test}v_{m_test}c.pkl", "rb") as f:
    parameters, test_time_before_avg,test_time_after_avg,test_iterations_before_avg,test_iterations_after_avg,test_iterations_diff_avg = pickle.load(f)

histogram_time(test_time_before_avg, test_time_after_avg, f"{model_name}_test", save=False)
histogram_iterations(test_iterations_before_avg, test_iterations_after_avg, f"{model_name}_test", save=False)


##### Figure 6 in paper: histogram of iterations before and after GNN warm-starting for 5v_206c and 50v_296c #####

# Set parameters
n = [5, 50]
m = [206, 296]

desired_ratio = 4 / 3          # width/height per subplot — adjust as needed
subplot_height = 6             # inches
subplot_width  = subplot_height * desired_ratio
fig_width = subplot_width * 2  # two subplots side by side

fig, axes = plt.subplots(1, 2, figsize=(fig_width, subplot_height), sharey=True)
model_name = f"model_{n[0]}v_{m[0]}c_lmpc_R_00001_avg"
with open("./data/lmpc_experiment_5v_206c_server.pkl", "rb") as f:
    parameters, test_time_before_avg, test_time_after_avg, \
    test_iterations_before_avg, test_iterations_after_avg, \
    test_iterations_diff_avg = pickle.load(f)

histogram_iterations(test_iterations_before_avg, test_iterations_after_avg,
                     f"{model_name}_test", save=False, ax=axes[0])

model_name = f"model_{n[1]}v_{m[1]}c_lmpc_R_00001_avg"
with open("./data/lmpc_experiment_50v_296c_t_05.pkl", "rb") as f:
    parameters, test_time_before_avg, test_time_after_avg, \
    test_iterations_before_avg, test_iterations_after_avg, \
    test_iterations_diff_avg = pickle.load(f)

histogram_iterations(test_iterations_before_avg, test_iterations_after_avg,
                     f"{model_name}_test", save=False, ax=axes[1])

# Single shared legend, deduped
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc='upper center', ncol=2, fontsize=18,
           bbox_to_anchor=(0.5, 1.05))

# Remove y-axis label from right subplot since sharey=True shares the axis
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(f"plots/histo_iter_lmpc_combined.pdf", bbox_inches='tight',save = False)
plt.show()

##### Figure 9: histogram of time before and after GNN warm-starting for 5v_206c and 50v_296c #####

# Set parameters
n = [5, 50]
m = [206, 296]

desired_ratio = 4 / 3          # width/height per subplot — adjust as needed
subplot_height = 6             # inches
subplot_width  = subplot_height * desired_ratio
fig_width = subplot_width * 2  # two subplots side by side

fig, axes = plt.subplots(1, 2, figsize=(fig_width, subplot_height), sharey=True)
model_name = f"model_{n[0]}v_{m[0]}c_lmpc_R_00001_avg"
with open("./data/lmpc_experiment_5v_206c_server.pkl", "rb") as f:
    parameters, test_time_before_avg, test_time_after_avg, \
    test_iterations_before_avg, test_iterations_after_avg, \
    test_iterations_diff_avg = pickle.load(f)

histogram_time(test_time_before_avg, test_time_after_avg,
                     f"{model_name}_test", save=False, ax=axes[0])

model_name = f"model_{n[1]}v_{m[1]}c_lmpc_R_00001_avg"
with open("./data/lmpc_experiment_50v_296c_t_05.pkl", "rb") as f:
    parameters, test_time_before_avg, test_time_after_avg, \
    test_iterations_before_avg, test_iterations_after_avg, \
    test_iterations_diff_avg = pickle.load(f)

histogram_time(test_time_before_avg, test_time_after_avg,
                     f"{model_name}_test", save=False, ax=axes[1])

# Single shared legend, deduped
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),
           loc='upper center', ncol=2, fontsize=18,
           bbox_to_anchor=(0.5, 1.05))

# Remove y-axis label from right subplot since sharey=True shares the axis
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(f"plots/histo_time_lmpc_combined.pdf", bbox_inches='tight',save = False)
plt.show()
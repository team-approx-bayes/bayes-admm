# Util functions
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
try:
    import wandb
except ModuleNotFoundError as err:
    pass


# Plot 2D visualisation for toy data
def plot_2d_visualisation(model_test_outputs_list, data_generator, worker_ind_list_const=[-1], title="",
                          wandb_log=False, path=None):

    if data_generator.num_outputs == 2:
        # Binary classification: only 2 colours
        colours = ['#A9CCE0', '#AB5735']
        marker_styles = ['o', 's', 'x', '+']
        imshow_values = [1 / 24, 19 / 24]
    else:
        # Colours and markers for classes, for plotting (same values as the colormap 'Paired')
        colours = ['#A9CCE0', '#3774B3', '#B0E28C', '#3AA23D', '#F69698', '#E50028', '#F8BF73', '#FB7D21', '#C5AFD5',
                   '#6B3897', '#FDFE9F', '#AB5735']
        marker_styles = ['o', 's', 'x', '+']

        # imshow colour values for 10 classes
        imshow_values = [1 / 24, 3 / 24, 5 / 24, 7 / 24, 9 / 24, 11 / 24, 13 / 24, 15 / 24, 17 / 24, 19 / 24]

    # Create plot
    num_plots = len(model_test_outputs_list)
    if num_plots > 1:
        plt.figure(figsize=(9 * num_plots, 9))
    else:
        plt.figure(figsize=(9, 9))

    for count, model_test_outputs in enumerate(model_test_outputs_list):

        # If there is more than one plot, then want to split up the data into each plot, else want all data in one plot
        if num_plots > 1:
            worker_ind_list = [worker_ind_list_const[count]]
        else:
            worker_ind_list = worker_ind_list_const

        # Find most likely class
        model_test_outputs, model_test_class_outputs = torch.max(model_test_outputs, dim=-1)
        model_test_class_outputs = np.array(
            [imshow_values[model_test_class_outputs[i]] for i in range(len(model_test_class_outputs))])
        model_test_class_outputs = model_test_class_outputs.reshape(data_generator.test_shape)

        axs = plt.subplot(1, num_plots, count + 1)
        axs.title.set_text(title + 'Node ' + str(worker_ind_list))

        # Plot over test data (imshow)
        plt.imshow(model_test_class_outputs, cmap='Paired',
                   extent=(data_generator.x_axis_min, data_generator.x_axis_max,
                           data_generator.y_axis_min, data_generator.y_axis_max),
                   vmin=0, vmax=1, origin='lower')

        # Plot training datapoints
        for worker_ind in worker_ind_list:
            labels_unique = np.unique(data_generator.labels[worker_ind])
            for label_ind in labels_unique:
                idx = np.where(data_generator.labels[worker_ind] == label_ind)
                plt.scatter(data_generator.inputs_plot[worker_ind][idx, 0],
                            data_generator.inputs_plot[worker_ind][idx, 1],
                            facecolors='w', edgecolors=colours[label_ind], s=30, linewidth=2,
                            marker=marker_styles[label_ind % 4], alpha=0.6)

    if path is not None:
        save_path = path + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if wandb_log:
        title_wandb = title[:title.find('comm')]
        wandb.log({title_wandb + 'Node ' + str(worker_ind_list_const): plt})

    plt.show()

# Very simple plotting function for metric
def plot_simple(metric, title="", y_label="", x_label="", xlim=None, ylim=None, horizontal_lines=[], horizontal_labels=[]):
    plt.figure(figsize=(9, 9))

    plt.plot(metric)

    # Plot horizontal lines
    for i in range(len(horizontal_lines)):
        plt.axhline(y=horizontal_lines[i], color='k', linestyle='--', label=horizontal_labels[i])

    plt.grid()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.show()

# Return model parameters as a vector
def return_model_parameters(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

# Set model parameters to a vector
def set_model_parameters(model, weight_values):
    torch.nn.utils.vector_to_parameters(copy.deepcopy(weight_values), model.parameters())

# KL(p||q), to evaluate the variational objective
# sigp, sigq are variances
def kldivergence(mp, sigp, mq, sigq):
    kl = ((mp - mq) ** 2.0) / sigq
    kl += torch.log(sigq) - torch.log(sigp) - 1.0 
    kl += sigp / sigq 

    return 0.5 * kl.sum() 
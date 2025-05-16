import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

level = 0.10
num_simulation = 50
if_saveplot = True

dirname = os.path.dirname(os.getcwd())
data_path = dirname + "/Dataset/mAb_dataset_validation.xlsx" # File name
fname = 'Data_missper' + str(int(level*100)) + '_numsim' + str(int(num_simulation)) + '/'

X0 = pd.read_excel(data_path)
X0 = X0.values
Time = X0[:, 0]
X0 = X0[:, 1:]

X0_mean = np.mean(X0, axis=0)
X0_std = np.std(X0, axis=0)
X0_norm = np.max(X0, axis=0) - np.min(X0, axis=0)

# Create lists for the plot
variables = ['Diameter', 'TotalDensity', 'ViableDensity', 'Ca', 'Gln', 'Glu', 'Gluc', 'K', 'Lac', 'Na', 'NH4', 'Osmo', 'P_CO2', 'pH']
units = ['\u03BCm', '10\u2075cells/mL', '10\u2075cells/mL', 'mmol/L', 'mmol/L', 'mmol/L', 'g/L', 'mmol/L', 'g/L', 'mmol/L', 'mmol/L', 'mOsm/kg', 'mmHg', '-']
algorithms = ['MI', 'Alternating', 'SVDImpute', 'PCADA', 'PPCA', 'PPCA-M', 'BPCA', 'SVT', 'ALM']
x_pos = np.arange(len(algorithms))
color = ['gray', 'red', 'orange', 'yellow', 'greenyellow', 'green', 'cyan', 'blue', 'purple']


feasibility_total = np.load(fname + 'feasibility_total.npy')
plausibility_total = np.load(fname + 'plausibility_total.npy')
rapidity_total = np.load(fname + 'rapidity_total.npy')
A_total = np.load(fname + 'A_total.npy')
NRMSE_total = np.load(fname + 'NRMSE_total.npy')

for miss_type in range(1, 6):
    subset = NRMSE_total[:, :, miss_type - 1, :]
    index = NRMSE_total[:, :, miss_type - 1, :] != 0
    NRMSE_mean = np.nanmean(np.where(index, subset, np.nan), axis=2)
    NRMSE_std = np.nanstd(np.where(index, subset, np.nan), axis=2)
    rapidity_mean = np.mean(rapidity_total[:, miss_type - 1, :], axis=1)
    rapidity_std = np.std(rapidity_total[:, miss_type - 1, :], axis=1)
    feasibility_mean = np.mean(feasibility_total[:, miss_type - 1, :], axis=1)
    feasibility_std = np.std(feasibility_total[:, miss_type - 1, :], axis=1)
    plausibility_mean = np.mean(plausibility_total[:, miss_type - 1, :], axis=1)
    plausibility_std = np.std(plausibility_total[:, miss_type - 1, :], axis=1)
    A_mean = np.mean(A_total[:, miss_type - 1, :], axis=1)
    A_std = np.std(A_total[:, miss_type - 1, :], axis=1)

    fname = 'Data_missper' + str(int(level*100)) + '_numsim' + str(int(num_simulation)) + '/Plots_validation_total'
    if os.path.exists(fname) == False:
        os.mkdir(fname)


    # Build the plot
    fig, ax = plt.subplots(figsize=(20,10))
    lower_error = np.minimum(feasibility_std, feasibility_mean)
    upper_error = feasibility_std
    asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
    ax.bar(x_pos, feasibility_mean, yerr=asymmetric_error, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3, edgecolor='black')
    ax.set_ylim(bottom=0, top=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=40)
    ax.yaxis.grid(True, zorder=0)
    ax.minorticks_on()
    ax.grid(which='major', axis='y', linestyle='-')
    ax.grid(which='minor', axis='y', linestyle=':')
    ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

    # Save the figure and show
    plt.ylabel('Number of elements', fontsize=60)
    fig.tight_layout()
    fig.tight_layout()
    if if_saveplot:
        plt.savefig(fname + '/Feasibility_misstype' + str(miss_type) + '.png')
    #plt.show()
    #plt.close()

    # Build the plot
    fig, ax = plt.subplots(figsize=(20,10))
    lower_error = np.minimum(plausibility_std, plausibility_mean)
    upper_error = plausibility_std
    asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
    ax.bar(x_pos, plausibility_mean, yerr=asymmetric_error, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3, edgecolor='black')
    ax.set_ylim(bottom=0, top=40)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=40)
    ax.yaxis.grid(True, zorder=0)
    ax.minorticks_on()
    ax.grid(which='major', axis='y', linestyle='-')
    ax.grid(which='minor', axis='y', linestyle=':')
    ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

    # Save the figure and show
    plt.ylabel('Number of elements', fontsize=60)
    fig.tight_layout()
    fig.tight_layout()
    if if_saveplot:
        plt.savefig(fname + '/Plausibility_misstype' + str(miss_type) + '.png')
    #plt.show()
    #plt.close()

    # Build the plot
    fig, ax = plt.subplots(figsize=(20,10))
    lower_error = np.minimum(rapidity_std, rapidity_mean)
    upper_error = rapidity_std
    asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
    ax.bar(x_pos, rapidity_mean, yerr=asymmetric_error, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3, edgecolor='black')
    ax.set_ylim(bottom=0.001, top=1000)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=40)
    ax.yaxis.grid(True, zorder=0)
    ax.minorticks_on()
    ax.grid(which='major', axis='y', linestyle='-')
    ax.grid(which='minor', axis='y', linestyle=':')
    ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

    # Save the figure and show
    plt.yscale("log")
    plt.ylabel('Computation time (s)', fontsize=60)
    fig.tight_layout()
    fig.tight_layout()
    if if_saveplot:
        plt.savefig(fname + '/Rapidity_misstype' + str(miss_type) + '.png')
    #plt.show()
    #plt.close()

    # Build the plot
    fig, ax = plt.subplots(figsize=(20,10))
    lower_error = np.minimum(A_std, A_mean)
    upper_error = A_std
    asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
    ax.bar(x_pos, A_mean, yerr=asymmetric_error, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3, edgecolor='black')
    ax.set_ylim(bottom=3, top=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=40)
    ax.yaxis.grid(True, zorder=0)
    ax.minorticks_on()
    ax.grid(which='major', axis='y', linestyle='-')
    ax.grid(which='minor', axis='y', linestyle=':')
    ax.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

    # Save the figure and show
    plt.ylabel('Number of PCs', fontsize=60)
    fig.tight_layout()
    fig.tight_layout()
    if if_saveplot:
        plt.savefig(fname + '/A_misstype' + str(miss_type) + '.png')
    #plt.show()
    #plt.close()


    if miss_type == 1:
        miss_type_string = 'Missing completely at random'
    elif miss_type == 2:
        miss_type_string = 'Sensor drop-out'
    elif miss_type == 3:
        miss_type_string = 'Multi-rate missingness'
    elif miss_type == 4:
        miss_type_string = 'Censoring'
    elif miss_type == 5:
        miss_type_string = 'Patterned missingness'

    title = miss_type_string
    numrow = 3
    numcol = math.ceil(NRMSE_total.shape[0] / numrow)
    fig, axs = plt.subplots(figsize=(20,12), nrows=numrow, ncols=numcol)
    fig.suptitle(title, fontsize=40)
    varind = 0

    for i in range(numrow):
        for j in range(numcol):
            if varind >= NRMSE_total.shape[0]:
                for k in range(j, numcol):
                    axs[-1, k].axis('off')
                break

            axs[i, j].set_xticks(x_pos)
            lower_error = np.minimum(NRMSE_std[varind,:], NRMSE_mean[varind,:])
            upper_error = NRMSE_std[varind,:]
            asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
            axs[i, j].bar(x_pos, NRMSE_mean[varind,:], yerr=asymmetric_error, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3, edgecolor='black')
            axs[i, j].set_ylim(bottom=0, top=3)
            axs[i, j].yaxis.grid(True, zorder=0)
            axs[i, j].minorticks_on()
            axs[i, j].grid(which='major', axis='y', linestyle='-')
            axs[i, j].grid(which='minor', axis='y', linestyle=':')
            axs[i, j].set_title(variables[varind] + ' (' + units[varind] + ')', fontsize=18)
            axs[i, j].tick_params(axis='y', labelsize=12)
            axs[i, j].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)

            varind += 1
    fig.tight_layout()
    fig.tight_layout()
    if if_saveplot:
        plt.savefig(fname + '/NRMSE_misstype' + str(miss_type) + '.png')
    #plt.show()
    #plt.close()

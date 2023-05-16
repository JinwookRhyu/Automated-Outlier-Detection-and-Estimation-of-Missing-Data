import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def plot_dataset(X, X_label, Time, variables, units, title, xlabel, ylim_lb, ylim_ub, if_saveplot, fname, if_nrmse, nrmse, if_showplot = True, numrow = 3):
    assert X.shape[0] == len(Time), f"The number of observations: {X.shape[0]} does not match with the length of time: {len(Time)}"
    assert X.shape[1] == len(variables), f"The number of variables: {X.shape[1]} does not match with the length of variables list: {len(variables)}"

    if if_showplot == True:

        color = ["blue", "cyan", "red", "springgreen", "olivedrab"]
        marker = ["o", "^", "*", "^", "*"]
        size = [20, 30, 70, 30, 70]
        numcol = math.ceil(X.shape[1] / numrow)

        fig, axs = plt.subplots(figsize=(20,12), nrows=numrow, ncols=numcol)
        fig.supxlabel(xlabel, fontsize=30)
        if if_saveplot == False:
            fig.suptitle(title, fontsize=40)
        varind = 0
        for i in range(numrow):
            for j in range(numcol):
                if varind >= X.shape[1]:
                    for k in range(j, numcol):
                        axs[-1, k].axis('off')
                    break
                for k in range(0, 5):
                    x = Time[X_label[:, varind] == k]
                    y = X[X_label[:,varind]==k, varind]
                    axs[i, j].scatter(x=x, y=y, marker=marker[k], color=color[k], s=size[k])
                axs[i, j].minorticks_on()
                axs[i, j].grid(which='major', linestyle='-', linewidth='0.5')
                axs[i, j].grid(which='minor', linestyle=':', linewidth='0.5')
                if if_nrmse == 1:
                    if nrmse[varind] > 0:
                        axs[i, j].set_title(variables[varind] + ' (' + units[varind] + ')\nNRMSE: ' + str(round(nrmse[varind], 4)), fontsize=18)
                    else:
                        axs[i, j].set_title(variables[varind] + ' (' + units[varind] + ')\nNRMSE: -', fontsize=18)
                else:
                    axs[i, j].set_title(variables[varind] + ' (' + units[varind] + ')', fontsize=18)
                axs[i, j].set_xlim(left=0, right=np.max(Time) + 1)
                axs[i, j].set_ylim(bottom=ylim_lb[varind], top=ylim_ub[varind])
                axs[i, j].tick_params(axis='x', labelsize=12)
                axs[i, j].tick_params(axis='y', labelsize=12)
                varind += 1
        plt.tight_layout()
        if if_saveplot == True:
            plt.savefig(fname=fname + title + '.png')
        plt.show()
        plt.close()


def plot_evaluation(feasibility, plausibility, time_final, fname, title, if_saveplot, if_showplot = True):

    if if_showplot == True:

        fig, axs = plt.subplots(figsize=(14, 6), nrows=1, ncols=3)
        fig.suptitle('Algorithm evaluation results', fontsize=40)
        algorithms = ['MI', 'Alternating', 'SVDImpute', 'PCADA', 'PPCA', 'PPCA-M', 'BPCA', 'SVT', 'ALM']
        x_pos = np.arange(len(algorithms))
        color = ['gray', 'red', 'orange', 'yellow', 'greenyellow', 'green', 'cyan', 'blue', 'purple']

        axs[0].bar(x_pos, feasibility, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3,
                   edgecolor='black')
        axs[0].set_ylim(bottom=0, top=max(np.ceil(max(feasibility) * 1.5), 1))
        axs[0].set_xticks(x_pos)
        axs[0].set_xticklabels(algorithms, fontsize=12)
        axs[0].tick_params(axis='x', labelsize=12)
        axs[0].tick_params(axis='y', labelsize=12)
        axs[0].yaxis.grid(True, zorder=0)
        axs[0].minorticks_on()
        axs[0].grid(which='major', axis='y', linestyle='-')
        axs[0].grid(which='minor', axis='y', linestyle=':')
        axs[0].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        axs[0].set_title('Feasibility', fontsize=25)
        axs[0].set_ylabel('# of imputed values outside the boundaries', fontsize=15)

        axs[1].bar(x_pos, plausibility, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3,
                   edgecolor='black')
        axs[1].set_ylim(bottom=0, top=max(np.ceil(max(plausibility) * 1.5), 1))
        axs[1].set_xticks(x_pos)
        axs[1].set_xticklabels(algorithms, fontsize=12)
        axs[1].tick_params(axis='x', labelsize=12)
        axs[1].tick_params(axis='y', labelsize=12)
        axs[1].yaxis.grid(True, zorder=0)
        axs[1].minorticks_on()
        axs[1].grid(which='major', axis='y', linestyle='-')
        axs[1].grid(which='minor', axis='y', linestyle=':')
        axs[1].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        axs[1].set_title('Plausibility', fontsize=25)
        axs[1].set_ylabel('# of imputed values considered outliers', fontsize=15)

        axs[2].bar(x_pos, time_final, align='center', alpha=1, ecolor='black', capsize=10, color=color, zorder=3,
                   edgecolor='black')
        axs[2].set_ylim(bottom=0.001, top=10**(np.ceil(max(np.log10(time_final)))))
        axs[2].set_xticks(x_pos)
        axs[2].set_xticklabels(algorithms, fontsize=12)
        axs[2].tick_params(axis='x', labelsize=12)
        axs[2].tick_params(axis='y', labelsize=12)
        axs[2].yaxis.grid(True, zorder=0)
        axs[2].minorticks_on()
        axs[2].grid(which='major', axis='y', linestyle='-')
        axs[2].grid(which='minor', axis='y', linestyle=':')
        axs[2].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        axs[2].set_yscale('log')
        axs[2].set_title('Rapidity', fontsize=25)
        axs[2].set_ylabel('Computation time (s)', fontsize=15)
        fig.tight_layout()
        fig.tight_layout()
        if if_saveplot == True:
            plt.savefig(fname=fname + title + '.png')
        plt.show()
        plt.close()

import os
import pandas as pd
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import svd
from Preprocessing import preprocessing_A0, preprocessing_A1
from Plot_dataset import plot_dataset
from Determine_A import build_pca, pca_cross_validation
from Fill_missing import fill_missing
from Addmissingness import addMissingness
import concurrent.futures

mpl.use('Agg')

num_simulation = 50

# Pre-allocating results
feasibility_total = np.zeros((9, 5, num_simulation))
plausibility_total = np.zeros((9, 5, num_simulation))
rapidity_total = np.zeros((9, 5, num_simulation))
A_total = np.zeros((9, 5, num_simulation))
NRMSE_total = np.zeros((14, 9, 5, num_simulation))
NRMSE_overall_total = np.zeros((9, 5, num_simulation))

# Calling X from the data file
# The given data should be [Observation x Variable]
dirname = os.path.dirname(os.getcwd())
data_path = dirname + "/Dataset/mAb_dataset_validation.xlsx" # File name
sheet_name = 0 # Sheet name. 0 for the first page, 1 for the second page, ...
X = pd.read_excel(data_path, engine='openpyxl')
variables = X.columns[1:].tolist()
X = X.values
Time = X[:, 0]
X = X[:, 1:]                # Exclude time column from the dataset
X0 = copy.deepcopy(X)       # X0 for true values

# Input of Addmissingness software
level = 0.10    # percent of missing data, specified by a decimal
tol = 0.001     # tolerance to specify an acceptble deviation from the desired level
fname = 'Data_missper' + str(int(level*100)) + '_numsim' + str(int(num_simulation)) + '/'
if os.path.exists(fname) == False:
    os.mkdir(fname)

num_components_ub = 14                        # Upper limit of numPC for PCA and PLS mdoels
num_repeat = 10                               # Number of repetition of MC-CV for determining numPC if if_groupCV_A = False
n_splits = 5                                # Number of splits for determining numPC if if_groupCV_A = False
SE_rule_PCA = 1

# Setting units, upper/lower bounds for plotting, and upper/lower bounds for feasibility metric per each variable.
units = ['\u03BCm', '10\u2075cells/mL', '10\u2075cells/mL', 'mmol/L', 'mmol/L', 'mmol/L', 'g/L', 'mmol/L', 'g/L', 'mmol/L', 'mmol/L', 'mOsm/kg', 'mmHg', '-']
filter_lb = np.array([0] * X.shape[1])
filter_ub = np.array([np.inf] * X.shape[1])
ylim_lb = np.array([16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 6])
ylim_ub = np.array([26, 500, 500, 0.2, 6, 8, 6, 20, 5, 200, 5, 700, 300, 8.5])
variables_mask = np.array([1] * X.shape[1])
observations_mask = Time < 30 # Only consider the datapoints collected during the first 30 days

# Parameters related to outlier detection in Step A.
maxiter_outlier = 20             # Number of maximum iterations for outlier detection
Conlim_preprocessing = 0.999     # Confidence limit used for outlier detection
Contrib = '2D-Alcala'            # Method used for calculating T^2 and Q contributions. ['2D-Alcala'/'2D-Chiang'/'simple'/'absolute']
ContribLimMethod = '2D-Alcala'   # Method used for calculating the control limits for T^2 and Q contributions. ['2D-Alcala'/'t'/'norm']
numminelem = 0                   # Threshold for determining low-quality observations (i.e. remove observations with less than numminelem elements)

def run_simulation(simulation, miss_type, variables_mask, observations_mask, units, filter_lb, filter_ub, ylim_lb, ylim_ub):
    import copy

    fname = 'Data_missper' + str(int(level * 100)) + '_numsim' + str(int(num_simulation)) + '/Plots_misstype' + str(
        miss_type) + '/'
    if os.path.exists(fname) == False:
        os.mkdir(fname)

    # Calling X from the data file
    # The given data should be [Observation x Variable]
    X = pd.read_excel(data_path, sheet_name=sheet_name)
    variables = X.columns[1:].tolist()
    X = X.values
    X0 = copy.deepcopy(X)
    Time = X[:, 0]
    X = X[:, 1:]
    X0 = X0[:, 1:]

    # Setting whether to normalize the dataset and save/show the plots
    if_normalize = True  # Do we normalize the given data before we put it into the code? (true or false)
    if simulation == 1:
        if_saveplot = True  # Do we save the plots?
        if_showplot = True  # Do we show plots during iteration of steps A-1 and A-2?
    else:
        if_saveplot = False  # Do we save the plots?
        if_showplot = False  # Do we show plots during iteration of steps A-1 and A-2?

    #################################
    X, miss_per = addMissingness(X, miss_type, level, tol)
    #################################

    ### Step A-0: Preprocessing before Step A (Only use variables_mask and observations_mask)
    variables_mask = np.array([bool(item) for item in variables_mask])
    observations_mask = np.array([bool(item) for item in observations_mask])
    X_A0 = preprocessing_A0(X=X, variables_mask=variables_mask, observations_mask=observations_mask)
    Time = Time[observations_mask]
    variables = [variables[i] for i in np.argwhere(variables_mask).flatten()]
    units = [units[i] for i in np.argwhere(variables_mask).flatten()]
    filter_lb = filter_lb[variables_mask]
    filter_ub = filter_ub[variables_mask]
    ylim_lb = ylim_lb[variables_mask]
    ylim_ub = ylim_ub[variables_mask]
    title = 'Given dataset'
    xlabel = 'Time (day)'
    X_label_A0 = pd.isnull(X_A0) * 1
    plot_dataset(X=X_A0, X_label=np.zeros(X_A0.shape), Time=Time, variables=variables, units=units, title=title,
                 xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot,
                 fname=fname, if_nrmse=0, nrmse=[])
    N = X_A0.shape[0]  # Number of observations
    V = X_A0.shape[1]  # Number of variables

    ### Step A-1: Temporary imputation of missing values
    method = 'interpolation'  # Method used for temporary imputation. ['mean', 'interpolation', 'last_observed']
    X_label_A1 = copy.deepcopy(X_label_A0)
    X_A1 = copy.deepcopy(X_A0)
    X_A1 = preprocessing_A1(X=X_A1, Time=Time, method=method)
    title = 'Temporary imputation using interpolation [Iteration 1]'
    plot_dataset(X=X_A1, X_label=X_label_A1, Time=Time, variables=variables, units=units, title=title,
                 xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot,
                 fname=fname, if_nrmse=0, nrmse=[])

    ### Step A-2: Outlier detection based on T^2 and Q contributions
    # Determine number of PCs (using cross-validation)
    PRESS_PCA = pca_cross_validation(X_A1, n_splits=n_splits, num_repeat=num_repeat,
                                     num_components_ub=num_components_ub)
    num_outliers = 0
    indmin_PCA = np.argmin(np.mean(PRESS_PCA, axis=1))
    A = np.where(np.mean(PRESS_PCA, axis=1) < np.mean(PRESS_PCA[indmin_PCA, :]) + SE_rule_PCA * np.std(
        PRESS_PCA[indmin_PCA, :], ddof=1) / np.sqrt(PRESS_PCA.shape[1]))[0][0] + 1

    # List of indices detected as outliers
    ind_normal = list([[]] * V)
    ind_outlier = list([[]] * V)
    ind_outlier_old = list([[]] * V)
    ind_outlier_new = list([[]] * V)

    ind_time = [i for i in range(len(Time)) if Time[i] > 10000 / 24 / 60]

    # Iteration of [Outlier detection / Temporary imputation / Determination of number of PCs] for outlier detection
    X_label_A2 = copy.deepcopy(X_label_A0)
    X_A2 = copy.deepcopy(X_A0)
    for numiter in range(maxiter_outlier):

        # Calculation of T^2 and Q contributions
        model = build_pca(X=X_A1, A=A, ConLim=Conlim_preprocessing, Contrib=Contrib,
                          ContribLimMethod=ContribLimMethod)
        P = model['parameters']['P']
        T = model['prediction']['T']
        T_sq_con = model['diagnostics']['T_sq_con']
        SRE_con = model['diagnostics']['SRE_con']
        lim_T_sq_con = model['estimates']['lim_T_sq_con']
        lim_SRE_con = model['estimates']['lim_SRE_con']

        # Outlier detection based on the T^2 and Q contributions
        for i in range(V):
            ind_outlier_new[i] = list(
                set(np.where((abs(T_sq_con[:, i]) > lim_T_sq_con[i]) | (abs(SRE_con[:, i]) > lim_SRE_con[i]))[
                        0]) - set(
                    ind_outlier[i]))
            ind_outlier_old[i] = ind_outlier[i]
            ind_outlier_new[i] = [x for x in ind_outlier_new[i] if x in ind_time]
            ind_outlier_old[i] = [x for x in ind_outlier_old[i] if x in ind_time]
            ind_outlier[i] = list(set(ind_outlier_new[i] + ind_outlier_old[i]))
            ind_normal[i] = list(
                set([x for x in range(X_A0.shape[0]) if X_label_A0[x, i] == 0]) - set(ind_outlier[i]))
            X_label_A1[[x for x in ind_outlier_new[i] if X_label_A0[x, i] == 0], i] = 2
            X_label_A2[[x for x in ind_outlier_new[i] if X_label_A0[x, i] == 0], i] = 2
        title = 'Outlier detection [Iteration ' + str(numiter + 1) + ']'
        plot_dataset(X=X_A1, X_label=X_label_A1, Time=Time, variables=variables, units=units, title=title,
                     xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot,
                     if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])
        num_new_outliers = np.sum(X_label_A2 == 2) - num_outliers

        if num_new_outliers == 0:  # If there is no newly detected outliers
            break
        num_outliers = np.sum(X_label_A2 == 2)

        # Convert detected outliers as NaN values
        X_A2[X_label_A2 == 2] = np.nan

        # Temporary imputation
        X_A1 = copy.deepcopy(X_A2)
        X_A1 = preprocessing_A1(X=X_A1, Time=Time, method=method)
        title = 'Temporary imputation using interpolation [Iteration ' + str(numiter + 2) + ']'
        X_label_A1[X_label_A1 == 2] = 1
        plot_dataset(X=X_A1, X_label=X_label_A1, Time=Time, variables=variables, units=units, title=title,
                     xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot,
                     if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

        # Determination of number of PCs
        PRESS_CV = pca_cross_validation(X_A1, n_splits=n_splits, num_repeat=num_repeat,
                                        num_components_ub=num_components_ub)
        indmin_CV = np.argmin(np.mean(PRESS_CV, axis=1))
        A_new = np.where(np.mean(PRESS_CV, axis=1) < np.mean(PRESS_CV[indmin_CV, :]) + SE_rule_PCA * np.std(
            PRESS_CV[indmin_CV, :], ddof=1) / np.sqrt(PRESS_CV.shape[1]))[0][0] + 1
        A = A_new

    # Plot the results after Step A-2
    title = 'Preprocessed dataset using interpolation'
    plot_dataset(X=X_A0, X_label=X_label_A2, Time=Time, variables=variables, units=units, title=title, xlabel=xlabel,
                 ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname,
                 if_nrmse=0, nrmse=[])

    if if_showplot:
        plt.figure()
        cmap = mpl.colors.ListedColormap(["blue", "cyan", "red", "black", "green", "green"])
        norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 5), cmap.N)
        im = plt.imshow(X_label_A2, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
        values = np.unique(X_label_A2.ravel())
        legend_labels = ['Observed Values', 'Missing Values', 'Detected Outliers', 'Removed Rows']
        patches = [mpatches.Patch(color=cmap.colors[values[i]], label=legend_labels[values[i]]) for i in
                   range(len(values))]
        plt.xlabel('Variables', fontsize=20)
        plt.ylabel('Observations', fontsize=20)
        plt.legend(handles=patches, borderaxespad=0., framealpha=0.5, fancybox=True)
        plt.tight_layout()
        if if_saveplot:
            plt.savefig(fname + 'preprocessed_data_indication.png')
        # plt.show()

    ### Step A-3: Elimination of low-quality observations
    X_B_old = copy.deepcopy(X_A0)
    X_B = copy.deepcopy(X_A2)
    X_label_B = copy.deepcopy(X_label_A2)
    Time_B = copy.deepcopy(Time)
    X_B_old = X_B_old[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i, :] == 0]) >= numminelem)], :]
    X_B = X_B[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i, :] == 0]) >= numminelem)], :]
    X_label_B = X_label_B[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i, :] == 0]) >= numminelem)], :]
    Time_B = np.array([Time_B[i] for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i, :] == 0]) >= numminelem)])
    X_label_A3 = copy.deepcopy(X_label_A2)
    X_label_A3[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i, :] == 0]) < numminelem)], :] = 3

    if if_showplot:
        plt.figure()
        im = plt.imshow(X_label_A3, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
        values = np.unique(X_label_A3.ravel())
        patches = [mpatches.Patch(color=cmap.colors[values[i]], label=legend_labels[values[i]]) for i in
                   range(len(values))]
        plt.xlabel('Variables', fontsize=20)
        plt.ylabel('Observations', fontsize=20)
        plt.legend(handles=patches, borderaxespad=0., framealpha=0.5, fancybox=True)
        plt.tight_layout()
        if if_saveplot:
            plt.savefig(fname + 'preprocessed_data_indication_after_removing.png')
        # plt.show()

    X_label_B_old = copy.deepcopy(X_label_B)
    X_label_B[X_label_B == 1] = 3
    X_label_B[X_label_B == 2] = 4

    X_B1 = copy.deepcopy(X_B)
    X_B1 = preprocessing_A1(X=X_B1, Time=Time, method=method)
    PRESS_PCA = pca_cross_validation(X_B1, n_splits=n_splits, num_repeat=num_repeat,
                                     num_components_ub=num_components_ub)
    indmin_PCA = np.argmin(np.mean(PRESS_PCA, axis=1))
    A = np.where(np.mean(PRESS_PCA, axis=1) < np.mean(PRESS_PCA[indmin_PCA, :]) + SE_rule_PCA * np.std(
        PRESS_PCA[indmin_PCA, :], ddof=1) / np.sqrt(PRESS_PCA.shape[1]))[0][0] + 1

    ### Step B: Imputation of missing values using different imputation algorithms
    X0_mean = np.mean(X0, axis=0)
    X0_std = np.std(X0, axis=0)
    _, _, P0 = svd(X0)

    X_MI, A_MI, time_MI = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='MI', n_splits=n_splits,
                                       num_repeat=num_repeat, num_components_ub=num_components_ub,
                                       SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_MI = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_MI[j] = np.sqrt(
                np.sum(np.multiply((np.divide(X_MI[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                    np.isnan(X_B[:, j])))
    NRMSE_overall_MI = np.sqrt(
        np.sum(np.multiply((np.divide(X_MI - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with MI'
    plot_dataset(X=X_MI, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel,
                 ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname,
                 nrmse=NRMSE_MI, if_nrmse=1)

    X_Alternating, A_Alternating, time_Alternating = fill_missing(X=X_B, Time=Time_B, method='mean', A=A,
                                                                  algorithm='Alternating', n_splits=n_splits,
                                                                  num_repeat=num_repeat,
                                                                  num_components_ub=num_components_ub,
                                                                  SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_Alternating = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_Alternating[j] = np.sqrt(np.sum(
                np.multiply((np.divide(X_Alternating[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                np.isnan(X_B[:, j])))
    NRMSE_overall_Alternating = np.sqrt(
        np.sum(np.multiply((np.divide(X_Alternating - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with Alternating'
    plot_dataset(X=X_Alternating, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title,
                 xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot,
                 fname=fname, nrmse=NRMSE_Alternating, if_nrmse=1)

    X_SVDImpute, A_SVDImpute, time_SVDImpute = fill_missing(X=X_B, Time=Time_B, method=method, A=A,
                                                            algorithm='SVDImpute', n_splits=n_splits,
                                                            num_repeat=num_repeat, num_components_ub=num_components_ub,
                                                            SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_SVDImpute = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_SVDImpute[j] = np.sqrt(np.sum(
                np.multiply((np.divide(X_SVDImpute[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                np.isnan(X_B[:, j])))
    NRMSE_overall_SVDImpute = np.sqrt(
        np.sum(np.multiply((np.divide(X_SVDImpute - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with SVDImpute'
    plot_dataset(X=X_SVDImpute, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title,
                 xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot,
                 fname=fname, nrmse=NRMSE_SVDImpute, if_nrmse=1)

    X_PCADA, A_PCADA, time_PCADA = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='PCADA',
                                                n_splits=n_splits, num_repeat=num_repeat,
                                                num_components_ub=num_components_ub, SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_PCADA = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_PCADA[j] = np.sqrt(np.sum(
                np.multiply((np.divide(X_PCADA[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                np.isnan(X_B[:, j])))
    NRMSE_overall_PCADA = np.sqrt(
        np.sum(np.multiply((np.divide(X_PCADA - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with PCADA'
    plot_dataset(X=X_PCADA, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title,
                 xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot,
                 fname=fname, nrmse=NRMSE_PCADA, if_nrmse=1)

    X_PPCA, A_PPCA, time_PPCA = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='PPCA',
                                             n_splits=n_splits, num_repeat=num_repeat,
                                             num_components_ub=num_components_ub, SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_PPCA = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_PPCA[j] = np.sqrt(
                np.sum(np.multiply((np.divide(X_PPCA[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                    np.isnan(X_B[:, j])))
    NRMSE_overall_PPCA = np.sqrt(
        np.sum(np.multiply((np.divide(X_PPCA - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with PPCA'
    plot_dataset(X=X_PPCA, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel,
                 ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname,
                 nrmse=NRMSE_PPCA, if_nrmse=1)

    X_PPCAM, A_PPCAM, time_PPCAM = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='PPCA-M',
                                                n_splits=n_splits, num_repeat=num_repeat,
                                                num_components_ub=num_components_ub, SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_PPCAM = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_PPCAM[j] = np.sqrt(np.sum(
                np.multiply((np.divide(X_PPCAM[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                np.isnan(X_B[:, j])))
    NRMSE_overall_PPCAM = np.sqrt(
        np.sum(np.multiply((np.divide(X_PPCAM - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with PPCA-M'
    plot_dataset(X=X_PPCAM, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title,
                 xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot,
                 fname=fname, nrmse=NRMSE_PPCAM, if_nrmse=1)

    X_BPCA, A_BPCA, time_BPCA = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='BPCA',
                                             n_splits=n_splits, num_repeat=num_repeat,
                                             num_components_ub=num_components_ub, SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_BPCA = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_BPCA[j] = np.sqrt(
                np.sum(np.multiply((np.divide(X_BPCA[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                    np.isnan(X_B[:, j])))
    NRMSE_overall_BPCA = np.sqrt(
        np.sum(np.multiply((np.divide(X_BPCA - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with BPCA'
    plot_dataset(X=X_BPCA, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel,
                 ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname,
                 nrmse=NRMSE_BPCA, if_nrmse=1)

    X_SVT, A_SVT, time_SVT = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='SVT', n_splits=n_splits,
                                          num_repeat=num_repeat, num_components_ub=num_components_ub,
                                          SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_SVT = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_SVT[j] = np.sqrt(
                np.sum(np.multiply((np.divide(X_SVT[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                    np.isnan(X_B[:, j])))
    NRMSE_overall_SVT = np.sqrt(
        np.sum(np.multiply((np.divide(X_MI - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with SVT'
    plot_dataset(X=X_SVT, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel,
                 ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname,
                 nrmse=NRMSE_SVT, if_nrmse=1)

    X_ALM, A_ALM, time_ALM = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='ALM', n_splits=n_splits,
                                          num_repeat=num_repeat, num_components_ub=num_components_ub,
                                          SE_rule_PCA=SE_rule_PCA, verbose_alg=False)
    NRMSE_ALM = np.zeros((X0.shape[1]))
    for j in range(X0.shape[1]):
        if np.sum(np.isnan(X[:, j])) > 0:
            NRMSE_ALM[j] = np.sqrt(
                np.sum(np.multiply((np.divide(X_ALM[:, j] - X0[:, j], X0_std[j])) ** 2, np.isnan(X_B[:, j]))) / np.sum(
                    np.isnan(X_B[:, j])))
    NRMSE_overall_ALM = np.sqrt(
        np.sum(np.multiply((np.divide(X_ALM - X0, X0_std)) ** 2, np.isnan(X_B))) / np.sum(np.isnan(X_B)))
    title = 'Filled in dataset with ALM'
    plot_dataset(X=X_ALM, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel,
                 ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname,
                 nrmse=NRMSE_ALM, if_nrmse=1)

    ### Step C: Evaluation of each algorithm
    feasibility = np.zeros(9)
    plausibility = np.zeros(9)
    X_final = np.dstack((X_MI, X_Alternating, X_SVDImpute, X_PCADA, X_PPCA, X_PPCAM, X_BPCA, X_SVT, X_ALM))
    A_final = [A_MI, A_Alternating, A_SVDImpute, A_PCADA, A_PPCA, A_PPCAM, A_BPCA, A_SVT, A_ALM]
    time_final = [time_MI, time_Alternating, time_SVDImpute, time_PCADA, time_PPCA, time_PPCAM, time_BPCA, time_SVT,
                  time_ALM]
    NRMSE_final = np.squeeze(np.dstack((NRMSE_MI, NRMSE_Alternating, NRMSE_SVDImpute, NRMSE_PCADA, NRMSE_PPCA,
                                        NRMSE_PPCAM, NRMSE_BPCA, NRMSE_SVT, NRMSE_ALM)))
    NRMSE_overall_final = [NRMSE_overall_MI, NRMSE_overall_Alternating, NRMSE_overall_SVDImpute,
                           NRMSE_overall_PCADA, NRMSE_overall_PPCA, NRMSE_overall_PPCAM, NRMSE_overall_BPCA,
                           NRMSE_overall_SVT, NRMSE_overall_ALM]

    for i in range(9):
        model = build_pca(X=X_final[:, :, i], A=A_final[i], ConLim=Conlim_preprocessing, Contrib=Contrib,
                          ContribLimMethod=ContribLimMethod)
        T_sq_con = model['diagnostics']['T_sq_con']
        SRE_con = model['diagnostics']['SRE_con']
        lim_T_sq_con = model['estimates']['lim_T_sq_con']
        lim_SRE_con = model['estimates']['lim_SRE_con']

        # feasibility = # of imputed values outside the boundaries
        feasibility[i] = np.sum((np.multiply(X_final[:, :, i], np.isnan(X_B)) < filter_lb) | (
                    np.multiply(X_final[:, :, i], np.isnan(X_B)) > filter_ub))
        # plausibility = # of imputed values considered outliers
        plausibility[i] = np.sum((abs(np.multiply(T_sq_con, np.isnan(X_B))) > lim_T_sq_con) | (
                    abs(np.multiply(SRE_con, np.isnan(X_B))) > lim_SRE_con))

    print('Simulation # ' + str(simulation) + ' for miss type # ' + str(miss_type) + ' completed.')
    # Be sure to return results
    return (simulation, miss_type,
            feasibility, plausibility, time_final, A_final, NRMSE_final, NRMSE_overall_final)



all_tasks = [(sim, mt) for sim in range(1, num_simulation+1) for mt in range(1, 6)]

results = []

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, sim, mt, variables_mask, observations_mask, units, filter_lb, filter_ub, ylim_lb, ylim_ub) for sim, mt in all_tasks]
        for f in concurrent.futures.as_completed(futures):
            simulation, miss_type, feasibility, plausibility, time_final, A_final, NRMSE_final, NRMSE_overall_final = f.result()
            feasibility_total[:, miss_type - 1, simulation - 1] = feasibility
            plausibility_total[:, miss_type - 1, simulation - 1] = plausibility
            rapidity_total[:, miss_type - 1, simulation - 1] = time_final
            A_total[:, miss_type - 1, simulation - 1] = A_final
            NRMSE_total[:, :, miss_type - 1, simulation - 1] = NRMSE_final
            NRMSE_overall_total[:, miss_type - 1, simulation - 1] = NRMSE_overall_final


# Save results
fname = 'Data_missper' + str(int(level*100)) + '_numsim' + str(int(num_simulation)) + '/'
np.save(fname + 'feasibility_total.npy', feasibility_total)
np.save(fname + 'plausibility_total.npy', plausibility_total)
np.save(fname + 'rapidity_total.npy', rapidity_total)
np.save(fname + 'A_total.npy', A_total)
np.save(fname + 'NRMSE_total.npy', NRMSE_total)

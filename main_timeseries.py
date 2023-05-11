import pandas as pd
import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Preprocessing import preprocessing_A0, preprocessing_A1
from Plot_dataset import plot_dataset, plot_evaluation
from Determine_A import build_pca, cross_validate_pca, rescale_by, scale_by
from Fill_missing import fill_missing

# Calling X from the data file
# The given data should be [Observation x Variable]
data_path = "Dataset/202111919_pre_2.xlsx" # File name
sheet_name = 0 # Sheet name. 0 for the first page, 1 for the second page, ...
X = pd.read_excel(data_path, sheet_name=sheet_name)
variables = X.columns[1:].tolist()
X = X.values
Time = X[:, 0] / 24 / 60    # Convert [min] to [day]
X = X[:, 1:]                # Exclude time column from the dataset

# Setting whether to normalize the dataset and save/show the plots
if_normalize = True   # Do we normalize the given data before we put it into the code? (true or false)
if_saveplot = False   # Do we save the plots?
fname = 'Plots/'      # File directory to save plots if if_saveplot == True
if_showplot = True    # Do we show plots during iteration of steps A-1 and A-2?

# Setting units, upper/lower bounds for plotting, and upper/lower bounds for feasibility metric per each variable.
units = ['\u03BCm', '10\u2075cells/mL', '-', '10\u2075cells/mL', 'mmol/L', 'mmol/L', 'mmol/L', 'g/L', 'mmol/L', 'g/L', 'mmol/L', 'mmol/L', 'mOsm/kg', 'mmHg', '-', 'mmHg']
filter_lb = [0] * X.shape[1]
filter_ub = [np.inf] * X.shape[1]
ylim_lb = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 6, 0]
ylim_ub = [26, 500, 150, 500, 0.2, 6, 8, 6, 20, 5, 200, 5, 700, 300, 8.5, 300]
variables_mask = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0] # 1 to include, 0 to exclude
observations_mask = Time < 30 # Only consider the datapoints collected during the first 30 days

# Parameters related to outlier detection in Step A.
maxiter_outlier = 20            # Number of maximum iterations for outlier detection
Conlim_preprocessing = 0.9999   # Confidence limit used for outlier detection
Contrib = '2D'                  # Method used for calculating T^2 and Q contributions. ['2D'/'simple'/'absolute']
numminelem = 5                  # Threshold for determining low-quality observations (i.e. remove observations with less than numminelem elements)


### Step A-0: Preprocessing before Step A (Only use variables_mask and observations_mask)
variables_mask = [bool(item) for item in variables_mask]
variables_mask_del = variables_mask
X_A0 = preprocessing_A0(X=X, variables_mask=variables_mask, observations_mask=observations_mask)
Time = Time[np.argwhere(observations_mask).flatten()]
variables = [variables[i] for i in np.argwhere(variables_mask).flatten()]
units = [units[i] for i in np.argwhere(variables_mask).flatten()]
filter_lb = [filter_lb[i] for i in np.argwhere(variables_mask).flatten()]
filter_ub = [filter_ub[i] for i in np.argwhere(variables_mask).flatten()]
ylim_lb = [ylim_lb[i] for i in np.argwhere(variables_mask).flatten()]
ylim_ub = [ylim_ub[i] for i in np.argwhere(variables_mask).flatten()]
title = 'Given dataset'
xlabel = 'Time (day)'
X_label_A0 = np.isnan(X_A0) * 1
plot_dataset(X=X_A0, X_label=np.zeros(X_A0.shape), Time=Time, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])
N = X_A0.shape[0]   # Number of observations
V = X_A0.shape[1]   # Number of variables


### Step A-1: Temporary imputation of missing values
method = 'interpolation'    # Method used for temporary imputation. ['mean', 'interpolation', 'last_observed']
X_label_A1 = copy.deepcopy(X_label_A0)
X_A1 = copy.deepcopy(X_A0)
X_A1 = preprocessing_A1(X=X_A1, Time=Time, method=method)
title = 'Temporary imputation using interpolation [Iteration 1]'
plot_dataset(X=X_A1, X_label=X_label_A1, Time=Time, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])


### Step A-2: Outlier detection based on T^2 and Q contributions
# Determine number of PCs (using cross-validation)
A_CV = int(np.round(0.8 * np.min([V, N])))
model_CV = build_pca(X=X_A1, A=A_CV, ErrBasedOn='scaled', ConLim=Conlim_preprocessing, Contrib=Contrib, Preprocessing='standardize')
Q = np.zeros((V, V, A_CV))
alpha = np.zeros((V, A_CV))
R_sq = np.zeros((V, A_CV))
for a in range(A_CV):
    Q[:, :, a] = np.matmul(model_CV['parameters']['P'][:, :a], model_CV['parameters']['P'][:, :a].T)
    alpha[:, a] = np.diag(Q[:, :, a])
    R_sq[:, a] = np.divide(np.sum(np.matmul(scale_by(X=X_A1, mu=model_CV['scaling']['mu'], sigma=model_CV['scaling']['sigma']), Q[:, :, a]) ** 2, axis=0),
                           np.sum(scale_by(X=X_A1, mu=model_CV['scaling']['mu'], sigma=model_CV['scaling']['sigma']) ** 2, axis=0))

RMSE_CV, PRESS_CV = cross_validate_pca(X=X_A1, A=A_CV, VarMethod='venetian_blind', G_obs=7, Kind='ekf_fast')
A = np.argmin(RMSE_CV) + 1
num_outliers = 0

# List of indices detected as outliers
ind_normal = list([[]]*V)
ind_outlier = list([[]]*V)
ind_outlier_old = list([[]]*V)
ind_outlier_new = list([[]]*V)

ind_time = [i for i in range(len(Time)) if Time[i] > 10000 / 24 / 60]   # Indices that are not protected from considered outliers

# Iteration of [Outlier detection / Temporary imputation / Determination of number of PCs] for outlier detection
X_label_A2 = copy.deepcopy(X_label_A0)
X_A2 = copy.deepcopy(X_A0)
for numiter in range(maxiter_outlier):

    # Calculation of T^2 and Q contributions
    model = build_pca(X=X_A1, A=A, ErrBasedOn='scaled', ConLim=Conlim_preprocessing, Contrib=Contrib)
    P = model['parameters']['P']
    T = model['prediction']['T']
    E = model['prediction']['E']
    X_rec = rescale_by(X=model['prediction']['X_rec'], mu=model['scaling']['mu'], sigma=model['scaling']['sigma'])
    EV = model['performance']['EV']
    T_sq = model['diagnostics']['T_sq']
    SRE = model['diagnostics']['SRE']
    T_sq_con = model['diagnostics']['T_sq_con']
    SRE_con = model['diagnostics']['SRE_con']
    lim_T_sq = model['estimates']['lim_T_sq']
    lim_SRE = model['estimates']['lim_SRE']
    lim_T_sq_con = model['estimates']['lim_T_sq_con']
    lim_SRE_con = model['estimates']['lim_SRE_con']
    l = model['estimates']['l']

    # Outlier detection based on the T^2 and Q contributions
    for i in range(V):
        ind_outlier_new[i] = list(set(np.where((abs(T_sq_con[:, i]) > lim_T_sq_con[i]) | (abs(SRE_con[:, i]) > lim_SRE_con[i]))[0]) - set(ind_outlier[i]))
        ind_outlier_old[i] = ind_outlier[i]
        ind_outlier_new[i] = [x for x in ind_outlier_new[i] if x in ind_time]
        ind_outlier_old[i] = [x for x in ind_outlier_old[i] if x in ind_time]
        ind_outlier[i] = list(set(ind_outlier_new[i] + ind_outlier_old[i]))
        ind_normal[i] = list(set([x for x in range(X_A0.shape[0]) if X_label_A0[x, i] == 0]) - set(ind_outlier[i]))
        X_label_A1[[x for x in ind_outlier_new[i] if X_label_A0[x, i] == 0], i] = 2
        X_label_A2[[x for x in ind_outlier_new[i] if X_label_A0[x, i] == 0], i] = 2
    title = 'Outlier detection [Iteration ' + str(numiter+1) + ']'
    plot_dataset(X=X_A1, X_label=X_label_A1, Time=Time, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

    if np.sum(X_label_A2 == 2) == num_outliers: # If there is no newly detected outliers
        break
    num_outliers = np.sum(X_label_A2 == 2)

    # Convert detected outliers as NaN values
    X_A2[X_label_A2 == 2] = np.nan

    # Temporary imputation
    X_A1 = copy.deepcopy(X_A2)
    X_A1 = preprocessing_A1(X=X_A1, Time=Time, method=method)
    title = 'Temporary imputation using interpolation [Iteration ' + str(numiter+2) + ']'
    X_label_A1[X_label_A1 == 2] = 1
    plot_dataset(X=X_A1, X_label=X_label_A1, Time=Time, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

    # Determination of number of PCs
    RMSE_CV, PRESS_CV = cross_validate_pca(X=X_A1, A=A_CV, VarMethod='venetian_blind', G_obs=7, Kind='ekf_fast')
    A = np.argmin(RMSE_CV) + 1
    print ('Iteration ' + str(numiter+1) + ' completed for outlier detection')

# Plot the results after Step A-2
title = 'Preprocessed dataset using interpolation'
plot_dataset(X=X_A0, X_label=X_label_A2, Time=Time, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

if if_showplot:
    plt.figure()
    cmap = mpl.colors.ListedColormap(["blue", "cyan", "red", "black", "green", "green"])
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 5), cmap.N)
    im = plt.imshow(X_label_A2, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
    values = np.unique(X_label_A2.ravel())
    legend_labels = ['Observed Values', 'Missing Values', 'Detected Outliers', 'Removed Rows']
    patches = [mpatches.Patch(color=cmap.colors[values[i]], label=legend_labels[values[i]]) for i in range(len(values))]
    plt.xlabel('Variables', fontsize=20)
    plt.ylabel('Observations', fontsize=20)
    plt.legend(handles=patches, borderaxespad=0., framealpha=0.5, fancybox=True)
    plt.tight_layout()
    if if_saveplot:
        plt.savefig(fname + 'preprocessed_data_indication.png')
    plt.show()


### Step A-3: Elimination of low-quality observations
X_B = copy.deepcopy(X_A2)
X_label_B = copy.deepcopy(X_label_A2)
Time_B = copy.deepcopy(Time)
X_B = X_B[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i,:] == 0]) >= numminelem)], :]
X_label_B = X_label_B[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i,:] == 0]) >= numminelem)], :]
Time_B = np.array([Time_B[i] for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i, :] == 0]) >= numminelem)])
X_label_A2[[i for i in range(X_A2.shape[0]) if (np.sum([X_label_A2[i,:] == 0]) < numminelem)], :] = 3

if if_showplot:
    plt.figure()
    im = plt.imshow(X_label_A2, cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
    values = np.unique(X_label_A2.ravel())
    patches = [mpatches.Patch(color=cmap.colors[values[i]], label=legend_labels[values[i]]) for i in range(len(values))]
    plt.xlabel('Variables', fontsize=20)
    plt.ylabel('Observations', fontsize=20)
    plt.legend(handles=patches, borderaxespad=0., framealpha=0.5, fancybox=True)
    plt.tight_layout()
    if if_saveplot:
        plt.savefig(fname + 'preprocessed_data_indication_after_removing.png')
    plt.show()

X_label_B[X_label_B==1] = 3
X_label_B[X_label_B==2] = 4


### Step B: Imputation of missing values using different imputation algorithms
X_MI, A_MI, time_MI = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='MI')
title = 'Filled in dataset with MI'
plot_dataset(X=X_MI, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_Alternating, A_Alternating, time_Alternating = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='Alternating')
title = 'Filled in dataset with Alternating'
plot_dataset(X=X_Alternating, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_SVDImpute, A_SVDImpute, time_SVDImpute = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='SVDImpute')
title = 'Filled in dataset with SVDImpute'
plot_dataset(X=X_SVDImpute, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_PCADA, A_PCADA, time_PCADA = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='PCADA')
title = 'Filled in dataset with PCADA'
plot_dataset(X=X_PCADA, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_PPCA, A_PPCA, time_PPCA = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='PPCA')
title = 'Filled in dataset with PPCA'
plot_dataset(X=X_PPCA, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_PPCAM, A_PPCAM, time_PPCAM = fill_missing(X=X_B, Time=Time_B, method=method, A=A, algorithm='PPCA-M')
title = 'Filled in dataset with PPCA-M'
plot_dataset(X=X_PPCAM, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_BPCA, A_BPCA, time_BPCA = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='BPCA')
title = 'Filled in dataset with BPCA'
plot_dataset(X=X_BPCA, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_SVT, A_SVT, time_SVT = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='SVT')
title = 'Filled in dataset with SVT'
plot_dataset(X=X_SVT, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

X_ALM, A_ALM, time_ALM = fill_missing(X=X_B, Time=Time_B, method='mean', A=A, algorithm='ALM')
title = 'Filled in dataset with ALM'
plot_dataset(X=X_ALM, X_label=X_label_B, Time=Time_B, variables=variables, units=units, title=title, xlabel=xlabel, ylim_lb=ylim_lb, ylim_ub=ylim_ub, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, if_nrmse=0, nrmse=[])

### Step C: Evaluation of each algorithm
feasibility = np.zeros(9)
plausibility = np.zeros(9)
X_final = np.dstack((X_MI, X_Alternating, X_SVDImpute, X_PCADA, X_PPCA, X_PPCAM, X_BPCA, X_SVT, X_ALM))
A_final = [A_MI, A_Alternating, A_SVDImpute, A_PCADA, A_PPCA, A_PPCAM, A_BPCA, A_SVT, A_ALM]
time_final = [time_MI, time_Alternating, time_SVDImpute, time_PCADA, time_PPCA, time_PPCAM, time_BPCA, time_SVT, time_ALM]

for i in range(9):
    model = build_pca(X=X_final[:, :, i], A=A_final[i], ErrBasedOn='scaled', ConLim=0.9999, Contrib=Contrib)
    T_sq_con = model['diagnostics']['T_sq_con']
    SRE_con = model['diagnostics']['SRE_con']
    lim_T_sq_con = model['estimates']['lim_T_sq_con']
    lim_SRE_con = model['estimates']['lim_SRE_con']

    # feasibility = # of imputed values outside the boundaries
    feasibility[i] = np.sum((np.multiply(X_final[:, :, i], np.isnan(X_B)) < filter_lb) | (np.multiply(X_final[:, :, i], np.isnan(X_B)) > filter_ub))
    # plausibility = # of imputed values considered outliers
    plausibility[i] = np.sum((abs(np.multiply(T_sq_con, np.isnan(X_B))) > lim_T_sq_con) | (abs(np.multiply(SRE_con, np.isnan(X_B))) > lim_SRE_con))

title = 'Algorithm evaluation results'
plot_evaluation(feasibility=feasibility, plausibility=plausibility, time_final=time_final, if_saveplot=if_saveplot, if_showplot=if_showplot, fname=fname, title=title)


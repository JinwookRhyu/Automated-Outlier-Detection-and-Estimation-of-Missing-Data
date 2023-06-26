close all;
clear; clc;
rng(0);

addpath('.\PCA_related')
%% Calling X from the data file
% The given data should be [Observation x Variable]
file_name = "Dataset/mAb_dataset_demonstration.xlsx"; % File name
[X, ~] = xlsread(file_name);
variables = {'Diameter', 'TotalDensity', 'ViableDensity', 'Ca', 'Gln', 'Glu', 'Gluc', 'K', 'Lac', 'Na', 'NH_4', 'Osmo', 'P_{CO2}', 'pH'};
Time = X(:, 1) / 24 / 60;   % Convert [min] to [day]
X = X(:, 2:end);            % Exclude time column from the dataset

%% Setting whether to normalize the dataset and save/show the plots
if_normalize = true;  % Do we normalize the given data before we put it into the code? (true or false)
if_saveplot = false;  % Do we save the plots?
fname = 'Plots/';     % File directory to save plots if if_saveplot == True
if_showplot = false;   % Do we show plots during iteration of steps A-1 and A-2?

%% Setting units, upper/lower bounds for plotting, and upper/lower bounds for feasibility metric per each variable.
units = cell({['\mu','m'], '10^5 cells/mL', '10^5 cells/mL', 'mmol/L', 'mmol/L', 'mmol/L', 'g/L', 'mmol/L', 'g/L', 'mmol/L', 'mmol/L', 'mOsm/kg', 'mmHg', '-'});
filter_lb = zeros(1, size(X,2));
filter_ub = Inf * ones(1, size(X,2));
ylim_lb = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 6];
ylim_ub = [26, 500, 500, 0.2, 6, 8, 6, 20, 5, 200, 5, 700, 300, 8.5];
variables_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; % 1 to include, 0 to exclude
observations_mask = Time < 30; % Only consider the datapoints collected during the first 30 days

%% Parameters related to outlier detection in Step A.
maxiter_outlier = 20;           % Number of maximum iterations for outlier detection
Conlim_preprocessing = 0.9999;  % Confidence limit used for outlier detection
Contrib = '2D';                 % Method used for calculating T^2 and Q contributions. ['2D'/'simple'/'absolute']
numminelem = 5;                 % Threshold for determining low-quality observations (i.e. remove observations with less than numminelem elements)

%% Step A-0: Preprocessing before Step A (Only use variables_mask and observations_mask)
variables_mask = find(variables_mask);
X_A0 = preprocessing_A0(X, variables_mask, observations_mask);
Time = Time(observations_mask);
variables = variables(variables_mask);
units = units(variables_mask);
filter_lb = filter_lb(variables_mask);
filter_ub = filter_ub(variables_mask);
ylim_lb = ylim_lb(variables_mask);
ylim_ub = ylim_ub(variables_mask);
title = 'Given dataset';
xlabelname = 'Time (day)';
X_label_A0 = double(isnan(X_A0));
if_nrmse = 0;
nrmse = [];
numrow = 3;
plot_dataset(X_A0, zeros(size(X_A0)), Time, variables, units, title, xlabelname, ylim_lb, ylim_ub, if_saveplot, if_showplot, fname, if_nrmse, nrmse, numrow);
N = size(X_A0, 1);  % Number of observations;
V = size(X_A0, 2);  % Number of variables;

%% Step A-1: Temporary imputation of missing values
method = 'interpolation';   % Method used for temporary imputation. ['mean', 'interpolation', 'last_observed']
X_label_A1 = X_label_A0;
X_A1 = X_A0;
X_A1 = preprocessing_A1(X_A1, Time, method);
title = 'Temporary imputation using interpolation [Iteration 1]';
plot_dataset(X_A1, X_label_A1, Time, variables, units, title, xlabelname, ylim_lb, ylim_ub, if_saveplot, if_showplot, fname, if_nrmse, nrmse, numrow);

%% Step A-2: Outlier detection based on T^2 and Q contributions
% Determine number of PCs (using cross-validation)
A_CV = round(0.8 * min([V, N]));
[RMSE_CV, PRESS_CV] = cross_validate_pca(X_A1, A_CV, 'G_obs', 7);
[~, A] = min(RMSE_CV);
num_outliers = 0;

% List of indices detected as outliers
ind_normal = cell(V,1);
ind_outlier = cell(V,1);
ind_outlier_old = cell(V,1);
ind_outlier_new = cell(V,1);

ind_time = find(Time > 10000 / 24 / 60);   % Indices that are not protected from considered outliers

% Iteration of [Outlier detection / Temporary imputation / Determination of number of PCs] for outlier detection
X_label_A2 = X_label_A0;
X_A2 = X_A0;
for numiter = 1:maxiter_outlier

    % Calculation of T^2 and Q contributions
    model = build_pca(X_A1, A, 'ConLim', Conlim_preprocessing, 'Contrib', Contrib);
    P = model.parameters.P;
    T = model.prediction.T;
    T_sq_con = model.diagnostics.T_sq_con;
    SRE_con = model.diagnostics.SRE_con;
    lim_T_sq_con = model.estimates.lim_T_sq_con;
    lim_SRE_con = model.estimates.lim_SRE_con;

    % Outlier detection based on the T^2 and Q contributions
    for i = 1:V
        ind_outlier_new{i} = setdiff(find((abs(T_sq_con(:,i)) > lim_T_sq_con(i)) + (abs(SRE_con(:,i)) > lim_SRE_con(i))), ind_outlier{i});
        ind_outlier_old{i} = ind_outlier{i};
        ind_outlier_new{i} = intersect(ind_outlier_new{i}, ind_time);
        ind_outlier_old{i} = intersect(ind_outlier_old{i}, ind_time);
        ind_outlier{i} = unique([ind_outlier_old{i}; ind_outlier_new{i}]);
        ind_normal{i} = setdiff(find(X_label_A0(:, i) == 0), ind_outlier{i});
        X_label_A1(intersect(ind_outlier_new{i}, find(X_label_A0(:, i) == 0)), i) = 2;
        X_label_A2(intersect(ind_outlier_new{i}, find(X_label_A0(:, i) == 0)), i) = 2;
    end
    title = ['Outlier detection [Iteration ', num2str(numiter), ']'];
    plot_dataset(X_A1, X_label_A1, Time, variables, units, title, xlabelname, ylim_lb, ylim_ub, if_saveplot, if_showplot, fname, if_nrmse, nrmse, numrow);
   
    if sum(sum(X_label_A2 == 2)) == num_outliers % If there is no newly detected outliers
        break
    end
    num_outliers = sum(sum(X_label_A2 == 2));

    % Convert detected outliers as NaN values
    X_A2(X_label_A2 == 2) = nan;

    % Temporary imputation
    X_A1 = X_A2;
    X_A1 = preprocessing_A1(X_A1, Time, method);
    title = ['Temporary imputation using interpolation [Iteration ', num2str(numiter+2), ']'];
    X_label_A1(X_label_A1 == 2) = 1;
    plot_dataset(X_A1, X_label_A1, Time, variables, units, title, xlabelname, ylim_lb, ylim_ub, if_saveplot, if_showplot, fname, if_nrmse, nrmse, numrow);

    % Determination of number of PCs
    [RMSE_CV, PRESS_CV] = cross_validate_pca(X_A1, A_CV, 'G_obs', 7);
    [~, A] = min(RMSE_CV);
    disp(['Iteration ', num2str(numiter), ' completed for outlier detection'])
end

% Plot the results after Step A-2
title = 'Preprocessed dataset using interpolation';
plot_dataset(X_A0, X_label_A2, Time, variables, units, title, xlabelname, ylim_lb, ylim_ub, if_saveplot, if_showplot, fname, if_nrmse, nrmse, numrow)

if if_showplot
    figure(); clf();
    numcol = ceil(size(X_A0,2) / numrow);
    t = tiledlayout(numrow, numcol);
    set(gcf, 'WindowState', 'maximized');
    imagesc(X_label_A2)
    cmap = [0 0 1; 0 1 1; 1 0 0];
    colormap(cmap)
    xlabel('Variables', 'FontSize', 20);
    ylabel('Observations', 'FontSize', 20);
    xticks(1:V);
    set(gca, 'XTickLabel', variables, 'FontSize', 20)
    if if_saveplot
        savefig([fname, 'preprocessed_data_indication.png'])
    end
end

%% Step A-3: Elimination of low-quality observations
X_B = X_A2;
X_label_B = X_label_A2;
Time_B = Time;
X_B = X_B((sum((X_label_A2 == 0), 2) >= numminelem), :);
X_label_B = X_label_B((sum((X_label_A2 == 0), 2) >= numminelem), :);
Time_B = Time_B(sum((X_label_A2 == 0), 2) >= numminelem);
X_label_A2((sum((X_label_A2 == 0), 2) < numminelem), :) = 3;

if if_showplot
    figure(); clf();
    t = tiledlayout(numrow, numcol);
    set(gcf, 'WindowState', 'maximized');
    imagesc(X_label_A2)
    cmap = [0 0 1; 0 1 1; 1 0 0; 0 0 0];
    colormap(cmap)
    xlabel('Variables', 'FontSize', 20);
    ylabel('Observations', 'FontSize', 20);
    xticks(1:V);
    set(gca, 'XTickLabel', variables, 'FontSize', 20)
    if if_saveplot
        savefig([fname, 'preprocessed_data_indication_after_removing.png'])
    end
end

X_label_B(X_label_B==1) = 3;
X_label_B(X_label_B==2) = 4;


%% Step B: Imputation of missing values using different imputation algorithms
numiterals = 5;
[X_final, if_error, A_final, time_final] = fill_missing(X_B, if_normalize, A);

Algorithm_list = {'MI', 'ALS', 'Alternating', 'SVDImpute', 'PCADA', 'PPCA', 'PPCA-M', 'BPCA', 'SVT', 'ALS'};
for i = 1:length(Algorithm_list)
    title = ['Filled in dataset with ', Algorithm_list{i}];
    plot_dataset(X_final(:,:,i), X_label_B, Time_B, variables, units, title, xlabelname, ylim_lb, ylim_ub, if_saveplot, true, fname, if_nrmse, nrmse, numrow);
end


%% Step C: Evaluation of each algorithm
feasibility = zeros(1, 10);
plausibility = zeros(1, 10);

for i = 1:10
    model = build_pca(X_final(:, :, i), A_final(i), 'ConLim', 0.9999, 'Contrib', Contrib);
    T_sq_con = model.diagnostics.T_sq_con;
    SRE_con = model.diagnostics.SRE_con;
    lim_T_sq_con = model.estimates.lim_T_sq_con;
    lim_SRE_con = model.estimates.lim_SRE_con;

    % feasibility = # of imputed values outside the boundaries
    feasibility(i) = sum(sum(((X_final(:, :, i) .* isnan(X_B)) < filter_lb) | ((X_final(:, :, i) .* isnan(X_B)) > filter_ub)));
    % plausibility = # of imputed values considered outliers
    plausibility(i) = sum(sum(((abs(T_sq_con .* isnan(X_B))) > lim_T_sq_con) | (abs(SRE_con .* isnan(X_B)) > lim_SRE_con)));
end

title = 'Algorithm evaluation results';
plot_evaluation(feasibility, plausibility, time_final, fname, title, if_saveplot, true)








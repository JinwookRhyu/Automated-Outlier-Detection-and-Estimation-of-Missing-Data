function [RMSECV, PRESS] = cross_validate_pca (X_in, A_in, varargin)


%% Input assignments

X = X_in;
A = A_in;

%% Initial checks

% Number of observations and number of variables
[N, V] = size(X);

% Check if the data array is unfolded
if size(X, 3) ~= 1
	error('The data array must be unfolded for PCA model cross-validation')
end
% Check if the requested number of PCs is feasible
if N >= V
	if A > V
		error('The number of PCs cannot exceed the number of varibles')
	end
else
	if A > N
		error('The number of PCs cannot exceed the number of observations')
	end
end
% Check if there are missing values
if sum(ismissing(X), 'all') ~= 0
	error('Missing values found: PCA cannot be cross-validated')
end

%% Optional arguments development

% Optionals initialisation
G_obs = [];
preprocess = 'standardize';

% Development cycle
if ~isempty(varargin)
	for i = 1:2:length(varargin)
		key = varargin{i};
		switch key
			case 'Preprocessing'
				preprocess = varargin{i + 1};
				if strcmp(preprocess, 'standardize') || strcmp(preprocess, 'mean_center')
				else
					error(['Supported preprocessing methods are standardize ...'
						'and mean_center'])
                end
			case 'G_obs'
				G_obs = varargin{i + 1};
		end
	end
end

%% Grouping of samples and variables

cv = cvpartition(N, 'Kfold', G_obs);
grouping_obs = zeros(N,1);
% Grouping of observations
for i = 1:7
    grouping_obs = grouping_obs + cv.test(i) * i;
end

%% Cross-validation

% Pre-allocation
PRESS = zeros(A, G_obs);

% Loop on groups of observations
for g = 1:G_obs
	% Get calibration and validation data
	X_cal = X(grouping_obs ~= g, :);
	X_val = X(grouping_obs == g, :);
	% Choiche of the preprocessing
	switch preprocess
        case 'standardize'
			[X_cal, mu, sigma] = autoscale(X_cal);
        case 'mean_center'
			[X_cal, mu, sigma] = autoscale(X_cal);
			X_cal = scaleby(X_cal, zeros(1, V), 1./sigma);
			sigma = ones(1, V);
	end
	% Scale the validation data
	X_val = scale_by(X_val, mu, sigma);
	% Propagate the number of PCs
	A_model = A;
	% Get data rank

	% Calibrate the PCA model
	
	P = pca_by_svd(X_cal, A_model);
	
	% Initialise the replacement matrix
	rep_mat = eye(V);
	% Loop on PCs
	for a = 1:A
		% Deflate the replacement matrix and assign it to a temporary
		% variable
		P_replaced = rep_mat - P(:, a)*P(:, a)';
		% Check if any term on the diagonal is gone to zero or
		% negative
		if ~any(diag(P_replaced) < eps*10)
			% if this not happened, keep the deflated matrix,
			% otherwise keep the old one
			rep_mat = P_replaced;
        end

		% Assign the replacement matrix to be worked out
		rep_a = rep_mat;
		% Deti diagonal of matrix
		d = diag(rep_a);
		% Excludes negative terms from the diagonal and replaces with eps
		% (effecgive zero)
		d = max(d, eps);
		% Scale each column of the replacement matrix on the diagonal
		% term (they use a loop as they say that it is faster than the
		% matrix method using diag(1./d)*rep)
		for v = 1:V
			rep_a(:, v) = (1/d(v))*rep_a(:, v);
		end
		% Compute the press: project the validation data using the
		% replacement matrix, square it and sum on columns (variables),
		% then get the mean of all variables to correct for the number of
		% variables
		PRESS(a, g) = mean(sum((X_val*rep_a).^2, 1), 2);
    end

RMSECV = sqrt(sum(PRESS, 2, 'omitnan')/N);
	
end
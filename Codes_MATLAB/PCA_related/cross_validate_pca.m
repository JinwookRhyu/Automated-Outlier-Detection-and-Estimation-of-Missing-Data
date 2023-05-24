function [RMSECV_out, PRESS_out, groups_out, Q_sq_out, alpha_CV_out, P_CV_out] = cross_validate_pca (X_in, A_in, method_in, varargin)


%% Input assignments

X = X_in;
A = A_in;
method = method_in;

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
band_thick_obs = [];
preprocess = 'standardize';
alg = 'svd';
tol = 1e-12;
max_iter = 100;
kind = 'ekf_fast';
method_var = '';
G_var = [];
band_thick_var = [];
% Development cycle
if ~isempty(varargin)
	if isnumeric(varargin{1})
		G_obs = varargin{1};
		if length(varargin) > 1 && isnumeric(varargin{2})
			band_thick_obs = varargin{2};
			varargin = varargin(3:end);
		else
			varargin = varargin(2:end);
		end
	end
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
			case 'Algorithm'
				alg = varargin{i + 1};
				if strcmp(alg, 'svd') || strcmp(alg, 'nipals') || strcmp(alg, 'override')
				else
					error('Supported algorithms for PCA are SVD and NIPaLS')
				end
			case 'Tol'
				tol = varargin{i + 1};
			case 'MaxIter'
				max_iter = varargin{i + 1};
			case 'Kind'
				kind = varargin{i + 1};
				if strcmp(kind, 'rkf') || strcmp(kind, 'ekf') || strcmp(kind, 'ekf_fast')
				else
					error(['Supported kinds of cross-validation are rkf, ekf '...
						'and ekf_fast'])
				end
			case 'VarMethod'
				method_var = varargin{i + 1};
				if strcmp(method_var, 'leave_one_out') || strcmp(method_var, 'continuous_blocks') || strcmp(method_var, 'venetian_blind')
				else
					error(['Supported methods for grouping of variables are '...
						'leave-one-out, continuous blocks and venetian blind'])
				end
			case 'VarGroups'
				G_var = varargin{i + 1};
			case 'VarThickness'
				band_thick_var = varargin{i + 1};
			otherwise
				error(['Key ' key ' undefined'])
		end
	end
end

%% More checks

% rkf cross-validation does not require variable grouping
if strcmp(kind, 'rkf') && ~isempty(method_var)
	warning(['Cross-validation of rkf kind does not require variable '... 
		'grouping, resetting'])
	method_var = '';
	G_var = [];
	band_thick_var = [];
end
% Default values for G_var and/or band_thick_var
if strcmp(kind, 'ekf')
	if (strcmp(method_var, 'continuous_blocks') || strcmp(method_var, 'venetian_blind')) && isempty(G_var)
		G_var = V;
	end
	if strcmp(method_var, 'venetian_blind') && isempty(band_thick_var)
		band_thick_var = 1;
	end
end
% ekf_fast cross-validation works only with leave_one_out variable grouping
if strcmp(kind, 'ekf_fast')
	if isempty(method_var)
		method_var = 'leave_one_out';
	elseif ~strcmp(method_var, 'leave_one_out')
		warning(['Cross-validation of ekf_fast kind can une only leave_one_out '...
			'variable grouping, resetting'])
		method_var = 'leave_one_out';
% 		G_var = [];
% 		band_thick_var = [];
	end
end

%% Grouping of samples and variables

% Grouping of observations
grouping_obs = cross_validation_grouping(method, N, G_obs, band_thick_obs);

% Grouping of variables
if ~strcmp(kind, 'rkf')
	grouping_var = cross_validation_grouping(method_var, V, G_var, band_thick_var);
end

%% Cross-validation

% Pre-allocation
PRESS = zeros(A, G_obs);
PRESS_V = zeros(V, A, G_obs);
alpha_CV = zeros(V, A, G_obs);
P_CV = zeros(V, A, G_obs);

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
	data_rank = get_data_rank(X_cal);
	% Reassign number of components for modelling
	if A_model > data_rank
% 		warning(['Requested numebr of PCs exceeds rank of X (%i), resetting A '...
% 			'to %i'], data_rank, data_rank)
		A_model = data_rank;
	end
	% Calibrate the PCA model
	switch alg
		case 'svd'
			P = pca_by_svd(X_cal, A_model);
		case 'nipals'
			P = pca_by_nipals(X_cal, A_model, 'Tol', tol, 'MaxIter', max_iter);
		case 'override'
			P = spca(X_cal, [], A_model, inf, 0.5, 3000, 1e-9, false);
	end
	
	% KEEP MODIFYING FROM HERE
	
	% Choice of the kind of cross-validation
	switch kind
		case 'rkf'
			T_val = X_val*P;
			for a = 1:A_model
				E_val_a = X_val - T_val(:,1:a)*(P(:,1:a))';
				PRESS(a, g) = sum(E_val_a.^2, 'all');
				PRESS_V(:, a, g) = sum(E_val_a.^2, 1);
				alpha_CV(:, a, g) = diag(P(:,1:a)*P(:,1:a)');
				P_CV(:, a, g) = sign(P(1, a))*P(:,a); % Force loading on first variable to be always positive by convention
			end
			for a = A_model + 1:A
				PRESS(a, g) = NaN;
				PRESS_V(:, a, g) = NaN(1, V);
				alpha_CV(:, a, g) = NaN(1, V);
				P_CV(:, a, g) = NaN(1, V);
			end
		case 'ekf'
			% Loop on PCs
			for a = 1:A
				% Initialise the SSE
				SSE = zeros(1, V);
				% Find the boundary of loadings
				A_max = min([a, A_model]);
				% Loop on groups of variables
				for gv = 1:G_var
					% Find variables to be replaced
					rep_var = (grouping_var == gv);
					% Selects the a-th PC
					P_replaced = P(:, 1:A_max);
					% Kill the loadings of the vairables to be replaced
					P_replaced(rep_var, :) = 0;
					% Use least-squares to get the projection of the validation
					% data on the variables that are not replaced; note that
					% using X_val*P_replaced*P is not correct in a variable
					% replecement scenario
					E = X_val(:, rep_var) - X_val/P_replaced'*P(rep_var, 1:A_max)';
					% compute press contributions for thehis replacement
					SSE(1, rep_var) = sum(E.^2);
				end
				% Average the SSE to level out differences in the number of
				% vairables per group and get the PRESS
				PRESS(a, g) = mean(SSE);
			end
		case 'ekf_fast'
			% Initialise the replacement matrix
			rep_mat = eye(V);
			% Loop on PCs
			for a = 1:A
				% Check if deflation is possible
				if a < A_model
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
	end
end

% Compute the RMSECV
% if strcmp(kind, 'rkf') && A_model < A
% 	RMSECV = zeros(A, 1);
% 	Q_sq = zeros(V, A);
% 	idx = ~ismissing(PRESS);
% 	idx_V = ~ismissing(PRESS_V);
% 	for a = 1:A
% % 		RMSECV(a) = sqrt(sum(PRESS(idx(a, :)))/sum(idx(a, :)));
% % 		Q_sq(:, a) = 1 - sum(PRESS_V(idx_V(:, a, :)), 3)'./sum(autoscale(X).^2, 1);
% 		RMSECV(a) = sqrt(sum(PRESS(idx(a, :)))/sum(idx(a, :)));
% 		Q_sq(:, a) = 1 - sum(PRESS_V(idx_V(:, a, :)), 3)'./sum(autoscale(X).^2, 1);
% 
% 	end
% 	Q_sq = [];
% else
	RMSECV = sqrt(sum(PRESS, 2, 'omitnan')/N);
	Q_sq = zeros(V, A);
	for a = 1:A
		Q_sq(:, a) = 1 - sum(PRESS_V(:, a, :), 3, 'omitnan')'./sum(autoscale(X).^2, 1);
	end
% end

%% Output assignments

RMSECV_out = RMSECV;
PRESS_out = PRESS;
groups_out = grouping_obs;

if strcmp(kind, 'rkf')
	Q_sq_out = Q_sq;
	alpha_CV_out = alpha_CV;
	P_CV_out = P_CV;
else
	Q_sq_out = [];
	alpha_CV_out = [];
	P_CV_out = [];
end

end
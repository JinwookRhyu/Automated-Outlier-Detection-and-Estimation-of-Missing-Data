function model_out = build_pca (X_in, A_in, varargin)
% BUILD_PCA PCA model calibration
%
% Syntax:
%	model = build_pca (X, A)
%	model = build_pca (X, A, 'Key', value)
%
% Inputs:
%	X:		Data array to be used to calibrate the model
%	A:		Number of principal components
%
% Outputs:
%	model:	PCA model structure
%
% Keys and Values:
%	'Preprocessing': ['none' | 'mean_centre' | 'autoscale']
%		Preprocessing method to be applied to the input data structure, can be:
%		no scaling at all ('none'); mean-centring only ('mean_centre');
%		mean-centring and scaling to unit variance ('autoscale', default); note
%		that the algorithm implemented by this function requires mean centred
%		data, therefore if the option 'none' is used the user is in charge of
%		providing mean centred data, otherwise an error is issued
%	'Algorithm': ['svd' | 'nipals']
%		Algorithm to be used for PCA calibration, can be either singular value
%		decomposition (SVD) or non-linear iteratrive partial least squared
%		(NIPaLS) (default = 'svd')
%	'Tol': [tol]
%		Tolerance for convergence as 2-norm on relative scores variation
%		(NIPaLS algorithm only, default = 1e-15)
%	'MaxIter': [max_iter]
%		Maximum number of iterations allowed (NIPaLS algorithm only,
%		default = 100)
%	'ErrBasedOn': ['unscaled', 'scaled']
%		Scaling of data reconstruction, whether is should be returned and
%		computed from scaled or unscaled entities (default = 'unscaled'); note
%		that SSE, RMSE, bias and SE will be reported as scaled even in the
%		unscaled errors are requested as they are meant to assess the performance
%		of the model
%	'Contrib': ['simple' | 'absolute']
%		Kind of contributions to diagnostics, which can be simple indictors
%		either positive or negative according to the direction of deviation and
%		with two-tail confidence limits (approach of Miller, 1998), or purely
%		positive contributions (like "squadred") that sum up the diagnostic
%		values and with one-tail confidence limits (approach of Westerhuis, 2000)
%		(default = 'simple')
%	'ConLim': [lim]
%		Confidence limit for diagnostics statistics (default = 0.95)
%	'DOFMethod:['naive']
%		Method for computing degrees of freedom of the model, can be either based
%		on the number of latent variables ('naive', dof = A + 1, default)
%		WILL BE EXTENDED IN THE FUTURE
%	'TsqLimMethod': ['chisq' | 'F']
%		Method for computing the confidence limits on T_sq and T_sq, can be
%		either the chi squared distribution method (default) or the F
%		distribution method
%	'SqErrLimMethod': ['chisq' | 'jack_mod']
%		Method for computing the confidence limit on SRE_X, can be either the
%		chi squared distribution method (default) or the Jackson-Mudholkar
%		equation
%	'ContribLimMethod': ['norm' | 't']
%		Method for computing the confidence limits of contributions to
%		diagnostics, can be based on a normal distribution or on a t distribution
%		(default = 'norm')
%	'EllipseForPlots: ['two' | 'full']
%		Method for computing the semiaxes of the confidence ellipse for score
%		plot, compute from a F distribution with considering only two principal
%		components (A_F = 2) or all the requested ones (A_F = A, default)
%	'ObsNames': [obs_names]
%		Names of the observations as chars in a cell array (default are
%		progressive numerical identifiers prefixed by the letter O)
%	'XVarNames': [X_var_names]
%		Names of the variables as chars in a cell array (default are progressive
%		numerical identifiers prefixed by the letter X)
%
%
% NOTE
%	A convention of the sings of loadings is imposed for reproducibility:
%	principal components are always directed towards the direction for which the
%	loading with the maximum absolute value ha spositive sign.

%% Input assignments

X_unscaled = X_in;
A = A_in;

%% Initial checks

% Check if the data array is unfolded
if size(X_unscaled, 3) ~= 1
	error('The data array must be unfolded for PCA model calibration')
end
% Check if the requested number of PCs is feasible
if A > min(size(X_unscaled))
	error(['The number of principal components cannot exceed min(size(X)) = '...
		num2str(min(size(X_unscaled)))])
end
% Check if there are missing values
if sum(ismissing(X_unscaled), 'all') ~= 0
	%error('Missing values found: PCA cannot be calibrated')
end

% Number of observations and number of variables
[N, V] = size(X_unscaled);

%% Optional arguments development

% Optionals initialisation
preprocess = 'autoscale';
alg = 'svd';
tol = 1e-15;
max_iter = 100;
err_on = 'unscaled';
contrib = 'simple';
lim = 0.95;
dof_method = 'naive';
Tsq_lim_method = 'chisq';
SqE_lim_method = 'chisq';
con_lim_method = 'norm';
l_kind = 'full';

obs_names = cellstr([repmat('O', N, 1) char(pad(replace(string(num2str((1:N)')),  ' ', ''), length(num2str(N)), 'left', '0'))]);
var_names = cellstr([repmat('X', V, 1) char(pad(replace(string(num2str((1:V)')),  ' ', ''), length(num2str(V)), 'left', '0'))]);

% Development cycle
if ~isempty(varargin)
	for i = 1:2:length(varargin)
		key = varargin{i};
		switch key
			case 'Preprocessing'
				preprocess = varargin{i + 1};
				if strcmp(preprocess, 'none') || strcmp(preprocess, 'mean_centre') || strcmp(preprocess, 'autoscale')
				else
					error(['Supported preprocessing methods are no scaling, '...
						'mean-centring and autoscaling'])
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
			case 'ErrBasedOn'
				err_on = varargin{i + 1};
				if strcmp(err_on, 'scaled') || strcmp(err_on, 'unscaled')
				else
					error('Undefined key for ErrBasedOn')
				end
			case 'Contrib'
				contrib = varargin{i + 1};
				if strcmp(contrib, 'simple') || strcmp(contrib, 'absolute') || strcmp(contrib, '2D')
				else
					error(['Supported kinds of contributions to diagnostics '...
						'are simple and absolute'])
				end
			case 'ConLim'
				lim = varargin{i + 1};
			case 'DOFMethod'
				dof_method = varargin{i + 1};
				if strcmp(dof_method, 'naive')
				else
					error(['Supported methods for degrees of freedom '...
						'estimation is naive'])
				end
			case 'TsqLimMethod'
				Tsq_lim_method = varargin{i + 1};
				if strcmp(Tsq_lim_method, 'chisq') || strcmp(Tsq_lim_method, 'F')
				else
					error(['Supported methods for limit on T_sq are '...
						'chi^2 method and F distribution method'])
				end
			case 'SqErrLimMethod'
				SqE_lim_method = varargin{i + 1};
				if strcmp(SqE_lim_method, 'chisq') || strcmp(SqE_lim_method, 'jack_mod')
				else
					error(['Supported methods for limit on SRE_X are '...
						'chi^2 method and Jackson-Mudholkar equation'])
				end
			case 'ContribLimMethod'
				con_lim_method = varargin{i + 1};
				if strcmp(con_lim_method, 'norm') || strcmp(con_lim_method, 't')
				else
					error(['Supported methods for contribution limits are '...
						'normal distribution method or t distribution method'])
				end
			case 'EllipseForPlots'
				l_kind = varargin{i + 1};
				if strcmp(l_kind, 'two') || strcmp(l_kind, 'full')
				else
					error('Unrecognised value of the EllipseForPlots key')
				end
			case 'ObsNames'
				obs_names = varargin{i + 1};
				if any(size(obs_names) == N)
				else
					error(['Number of observation labels does not match the '...
						'number of observations'])
				end
			case 'XVarNames'
				X_var_names = varargin{i + 1};
				if any(size(X_var_names) == V)
				else
					error(['Number of variable labels does not match the '...
						'number of variables'])
				end
			otherwise
				error(['Key ' key ' undefined'])
		end
	end
end

%% PCA model structure initialisation

% Model structure
model = initialise_pca;

% Dimesions and labels assignments
model.dimensions.N = N;
model.dimensions.V = V;
model.dimensions.A = A;
model.info.obs_names = obs_names;
model.info.var_names = var_names;
model.info.preprocessing = preprocess;
model.info.algorithm = alg;
model.info.error_based_on = err_on;
model.info.contribution_method = contrib;
model.info.dof_method = dof_method;
model.info.Tsq_lim_method = Tsq_lim_method;
model.info.SqE_lim_method = SqE_lim_method;
model.info.con_lim_method = con_lim_method;
model.info.l_kind = l_kind;

%% Preprocessing

% Choiche of the preprocessing method
switch preprocess
	case 'none'
		mu = zeros(1, V);
		sigma = ones(1, V);
		X = X_unscaled;
	case 'mean_centre'
		mu = mean(X_unscaled);
		X = X_unscaled - mu;
		sigma = ones(1, V);
	case 'autoscale'
		[X, mu, sigma] = autoscale(X_unscaled);
end

% Assigments to the model structure
model.data.X = X;
model.data.X_uns = X_unscaled;
model.scaling.mu = mu;
model.scaling.sigma = sigma;
		
%% PCA model calibration

% Choiche of the algorithm for model calibration
switch alg
	case 'svd'
		[P, T] = pca_by_svd(X, A);
	case 'nipals'
		[P, T] = pca_by_nipals(X, A, 'Tol', tol, 'MaxIter', max_iter);
    case 'als'
        [P, T] = pca(X, 'algorithm', 'als', 'NumComponents', A);
end

% Variance of scores for diagnostics
sigma_sq = var(T);

% Assigments to the model structure
model.parameters.P = P;
model.parameters.sigma_sq = sigma_sq;
model.prediction.T = T;

%% Model performance

% Pre-allocation
SSE = zeros(A, 1);
bias = zeros(A, 1);
SE = zeros(A, 1);

% Reconstructued X
X_rec = T*P';
% Reconstruction error
E = X - X_rec;

% Unscaled explained variance
ssqT = diag(T'*T);
% Explained variance
EV = ssqT/trace(X'*X);
% Cumulative explained variance
CEV = cumsum(EV);
% Loop over principal components
for a = 1:A
	% Reconstruction error with PC up to a
	E_a = X - T(:, 1:a)*P(:, 1:a)';
	% SSE in calibration
	SSE(a) = sum(E_a.^2, 'all');
	% Error bias
	bias(a) = mean(E_a, 'all');
	% Standard error
	SE(a) = sum((E_a - bias(a)).^2, 'all')/(N - 1);
end
% Root mean squared error
RMSE = sqrt(SSE/N);
% Extracted eigenvalues (non-normalised)
lambda = ssqT/(N - 1);

% Copy entites for diagnostics
E_fd = E;

% Rescale entities according to preferences
if strcmp(err_on, 'unscaled')
	X_rec = X_rec*diag(sigma) + mu;
	E = E*diag(sigma);
end

if strcmp(alg, 'override')
	EV = EV_SPCA;
	CEV = cumsum(EV);
end

% Assigments to the model structure
model.prediction.X_rec = X_rec;
model.prediction.E = E;
model.performance.EV = EV;
model.performance.CEV = CEV;
model.performance.SSE = SSE;
model.performance.RMSE = RMSE;
model.performance.bias = bias;
model.performance.SE = SE;
model.performance.lambda = lambda;

%% Model diagnostics

% Hotelling's T^2
T_sq = sum((T.^2)*diag(sigma_sq.^(- 1)), 2);
% Squared reconstruction errors (SRE)
SRE = sum(E_fd.^2, 2);

% Choice of the kind of contributions
switch contrib
	case 'simple'
		% Upper and lower confidence limits
		lim_con = lim + (1 - lim)/2;
		% Contributions to Hotelling's T^2
		T_sq_con = T*sqrt(diag(sigma_sq.^(- 1)))*P';
		% Contributions to SRE
		SRE_con = E_fd;
	case 'absolute'
		% Upper and lower confidence limits
		lim_con = lim;
		% Contributions to Hotelling's T^2
		T_sq_con = T*sqrt(diag(sigma_sq.^(- 1)))*P'.*X;
		% Contributions to SRE
		SRE_con = E_fd.^2;
    case '2D'
		% Upper and lower confidence limits
		lim_con = lim;
        T_sq_con = zeros(N,V);
		% Contributions to Hotelling's T^2
        for k = 1:N
            t = P' * X(k,:)';
            cont = t .* X(k,:) .* P' ./ sigma_sq';
            T_sq_con(k,:) = sum(cont, 1);
        end
		
		% Contributions to SRE
		SRE_con = E_fd.^2;
end

% Assigments to the model structure
model.diagnostics.T_sq = T_sq;
model.diagnostics.SRE = SRE;
model.diagnostics.T_sq_con = T_sq_con;
model.diagnostics.SRE_con = SRE_con;

%% Estimation of confidence limits

% Degrees of freedom of the model
dof = (1:A)' + 1;

% Choice of the method for T_sq confidence limits
switch Tsq_lim_method
	case 'chisq'
		% Confindence limits on Hotelling's T^2
		DOF = 2*mean(T_sq)^2/var(T_sq);
		scalef = mean(T_sq)/DOF;
		lim_T_sq = scalef*chi2inv(lim, DOF);
	case 'F'
		% Confindence limits on Hotelling's T^2
		lim_T_sq = (A*(N - 1)/(N - A))*finv(lim, A, N - A);
end

% Choice of the method for SRE confidence limits
switch SqE_lim_method
	case 'chisq'
		% Confidence limits for SRE
		DOF = 2*mean(SRE)^2/var(SRE);
		scalef = mean(SRE)/DOF;
		lim_SRE = scalef*chi2inv(lim, DOF);
	case 'jack_mod'
		z = norminv(lim);
		% Confidence limits for SRE
		theta = zeros(1, 3);
		for j = 1:3
			theta(j) = sum(var(E_fd).^j);
		end
		h0 = 1 - (2*theta(1)*theta(3))/(3*(theta(2)^2));
		lim_SRE = theta(1)*( 1 -...
			theta(2)*h0*(1 - h0)/(theta(1)^2) +...
			sqrt(z*2*theta(2)*(h0^2))/theta(1)...
		)^(1/h0);
end

% Choice of the method for confidence limits of contributions
switch con_lim_method
	case 'norm'
		% Confindence limits for contribution to Hotelling's T^2
		%lim_T_sq_con = norminv(lim_con, zeros(1, V), std(T_sq_con));
        lim_T_sq_con = norminv(lim_con, mean(T_sq_con), std(T_sq_con));
		% Confindence limits for contributions to SRE
		%lim_SRE_con = norminv(lim_con, zeros(1, V), std(SRE_con));
        lim_SRE_con = norminv(lim_con, mean(SRE_con), std(SRE_con));
	case 't'
		% Degrees of freedom of the distribution
		DOF = N - dof;
		% Critical t-value
		t_cl = tinv(lim_con, DOF);
		% Confindence limits for contribution to Hotelling's T^2
		lim_T_sq_con = sqrt(diag(T_sq_con'*T_sq_con)/DOF)'*t_cl;
		% Confindence limits for contributions to SRE
		lim_SRE_con = sqrt(diag(SRE_con'*SRE_con)/DOF)'*t_cl;
% 	case 'quantile'
% 		% Confindence limits for contribution to Hotelling's T^2
% 		lim_T_sq_con = quantile(T_sq_con, lim_con);
% 		% Confindence limits for contributions to SRE
% 		lim_SRE_con = quantile(SRE_con, lim_con);
end

% Choice of the semiaxes for confidence ellipse
switch l_kind
	case 'two'
		% Issue a warning is chi^2 method is used for confidence limits
		if ~strcmp(Tsq_lim_method, 'F')
			warning(['Confidence ellipse requested as based on two principal '...
				'components only, possible only using F distribution-based '...
				'confidence limit for T_sq: the limit will be inconsistent'])
		end
		lim_T_sq_for_plots = (2*(N - 1)/(N - 2))*finv(lim, 2, N - 2);
	case 'full'
		lim_T_sq_for_plots = lim_T_sq;
end
% Confidence ellipsoid semiaxes
l = sqrt(sigma_sq*lim_T_sq_for_plots);

% Assigments to the model structure
model.estimates.lim = lim;
model.estimates.dof = dof;
model.estimates.lim_T_sq = lim_T_sq;
model.estimates.lim_SRE = lim_SRE;
model.estimates.lim_T_sq_con = lim_T_sq_con;
model.estimates.lim_SRE_con = lim_SRE_con;
model.estimates.l = l;

%% Output assignments

model_out = model;

end
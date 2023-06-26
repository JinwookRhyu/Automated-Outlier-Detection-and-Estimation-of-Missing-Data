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
%	'Contrib': ['simple' | 'absolute']
%		Kind of contributions to diagnostics, which can be simple indictors
%		either positive or negative according to the direction of deviation and
%		with two-tail confidence limits (approach of Miller, 1998), or purely
%		positive contributions (like "squadred") that sum up the diagnostic
%		values and with one-tail confidence limits (approach of Westerhuis, 2000)
%		(default = 'simple')
%	'ConLim': [lim]
%		Confidence limit for diagnostics statistics (default = 0.95)
%	'ContribLimMethod': ['norm' | 't']
%		Method for computing the confidence limits of contributions to
%		diagnostics, can be based on a normal distribution or on a t distribution
%		(default = 'norm')
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
contrib = 'simple';
lim = 0.95;
con_lim_method = 'norm';

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
			case 'Contrib'
				contrib = varargin{i + 1};
				if strcmp(contrib, 'simple') || strcmp(contrib, 'absolute') || strcmp(contrib, '2D')
				else
					error(['Supported kinds of contributions to diagnostics '...
						'are simple and absolute'])
				end
			case 'ConLim'
				lim = varargin{i + 1};
			case 'ContribLimMethod'
				con_lim_method = varargin{i + 1};
				if strcmp(con_lim_method, 'norm') || strcmp(con_lim_method, 't')
				else
					error(['Supported methods for contribution limits are '...
						'normal distribution method or t distribution method'])
                end
		end
	end
end

%% PCA model structure initialisation

% Model structure
model = [];

% Dimesions and labels assignments
model.dimensions.N = N;
model.dimensions.V = V;
model.dimensions.A = A;

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
[P, T] = pca_by_svd(X, A);
	

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

% Copy entites for diagnostics
E_fd = E;

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

% Confindence limits on Hotelling's T^2
DOF = 2*mean(T_sq)^2/var(T_sq);
scalef = mean(T_sq)/DOF;
lim_T_sq = scalef*chi2inv(lim, DOF);
	
% Confidence limits for SRE
DOF = 2*mean(SRE)^2/var(SRE);
scalef = mean(SRE)/DOF;
lim_SRE = scalef*chi2inv(lim, DOF);
	

% Choice of the method for confidence limits of contributions
switch con_lim_method
	case 'norm'
		% Confindence limits for contribution to Hotelling's T^2
        lim_T_sq_con = norminv(lim_con, mean(T_sq_con), std(T_sq_con));
		% Confindence limits for contributions to SRE
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
end

% Assigments to the model structure
model.estimates.lim = lim;
model.estimates.dof = dof;
model.estimates.lim_T_sq = lim_T_sq;
model.estimates.lim_SRE = lim_SRE;
model.estimates.lim_T_sq_con = lim_T_sq_con;
model.estimates.lim_SRE_con = lim_SRE_con;

%% Output assignments

model_out = model;

end
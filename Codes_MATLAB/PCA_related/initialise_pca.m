function model_out = initialise_pca ()
% INITIALISE_PCA Intilisation of the PCA model structure
%
% Syntax:
%	model = initilise_pca ()
%
% Outputs:
%	model:	Structured array of the initialised PCA model
%
%
% FIELDS OF THE STRUCTURE
%
% 	model:	main structure of the model
% 		parameters:	parameters of the PCA model
% 			P:			loadings
%			sigma_sq:	variance of scores for diagnostics
% 		prediction:	results of the application of the model
% 			T:		scores
% 			X_rec:	reconstructed X
% 			E:		residuals for reconstruction of X
% 		data:	data used for model calibration
% 			X:		array of scaled data
% 			X_uns:	array of unscaled data
% 		dimensions:	dimensions of the arrays
% 			N:		number of observations
% 			V:		number of variables
% 			A:		number of latent variables
% 		scaling:	scaling parameters
% 			mu:			means of variables
% 			sigma:		standard deviations of variables
% 		performance:	performance of the model in calibration
% 			EV:				explained variance 
% 			CEV:			cumulative explained variance
% 			SSE:			sum of squared error
% 			RMSE:			root mean squared error
% 			bias:			residual biase
% 			SE:				standard error
% 			lambda:			eigenvalues extracted by scores
% 		diagnostics:	diagnostics on the model in calibration
% 			T_sq:				Hotelling's T^2
% 			SRE:				squared reconstruction error (Q)
% 			T_sq_con:			contribution to Hotelling's T^2
% 			SRE_con:			contributions to SRE
%		estimates:	estimated confidence limits on diagnostics
% 			lim:			singificance level for confidence limits
%			dof:			degrees of freedom of the model
% 			lim_T_sq:		confidence limit of Hotelling's T^2
% 			lim_SRE:		confidence limit of squared reconstruction error
% 			lim_T_sq_con:	confidence limit of contribution to Hotelling's T^2
% 			lim_SRE_con:	confidence limit of contribution to SRE
% 			l:				confidence ellipsoid axes
% 		info:	information about the model
% 			preprocessing:			kind of preprocessing used
%			algorithm:				algorithm used to calibrate the model
% 			error_based_on:			scaled on unsceld predictions and errors
%			contribution_method:	method used for computing contributions to diagnostics
%			dof_method:				method used for computing degrees of freedom
%			Tsq_lim_method:			method used for computing confidence limits on T_sq
%			SqE_lim_method:			method used for computing confidence limits on SRE
%			con_lim_method:			method used for computing confidence limits on contributions to diagnostics
%			l_kind:					number of PCs considered for the confidence ellipse
% 			obs_names:				observation names
% 			var_names:				variable names

%% Structure initialisatin

model = struct;
	% Model paramters
	model.parameters = struct;
		model.parameters.P = [];
		model.parameters.sigma_sq = [];
	% Scores, predictions and errors
	model.prediction = struct;
		model.prediction.T = [];
		model.prediction.X_rec = [];
		model.prediction.E = [];
	% Data used for calibration
	model.data = struct;
		model.data.X = [];
		model.data.X_uns = [];
	% Dimensions of the model entities
	model.dimensions = struct;
		model.dimensions.N = [];
		model.dimensions.V = [];
		model.dimensions.A = [];
	% Sclaing applied to data
	model.scaling = struct;
		model.scaling.mu = [];
		model.scaling.sigma = [];
	% Perfomance of the model
	model.performance = struct;
		model.performance.EV = [];
		model.performance.CEV = [];
		model.performance.SSE = [];
		model.performance.RMSE = [];
		model.performance.bias = [];
		model.performance.SE = [];
		model.performance.lambda = [];
	% Diagnostics on the model
	model.diagnostics = struct;
		model.diagnostics.T_sq = [];
		model.diagnostics.SRE = [];
		model.diagnostics.T_sq_con = [];
		model.diagnostics.SRE_con = [];
	% Confidence limits
	model.estimates = struct;
		model.estimates.lim = [];
		model.estimates.dof = [];
		model.estimates.lim_T_sq = [];
		model.estimates.lim_SRE = [];
		model.estimates.lim_T_sq_con = [];
		model.estimates.lim_SRE_con = [];
		model.estimates.l = [];
	% Infos on the model
	model.info = struct;
		model.info.preprocessing = '';
		model.info.algorithm = '';
		model.info.error_based_on = '';
		model.info.diagnostics_based_on = '';
		model.info.contribution_method = '';
		model.info.dof_method = '';
		model.info.Tsq_lim_method = '';
		model.info.SqE_lim_method = '';
		model.info.con_lim_method = '';
		model.info.l_kind = '';
		model.info.obs_names = {};
		model.info.var_names = {};

%% Output assignments

model_out = model;

end
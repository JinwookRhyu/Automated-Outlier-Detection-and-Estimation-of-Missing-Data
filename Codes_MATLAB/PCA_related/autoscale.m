function [X_s, mu, sigma] = autoscale (X)
% AUTOSCALE Data arrays autoscaling
%	Mean centering and strandrd deviation scaling
%
% Syntax:
%	[X_s, mu, sigma] = autoscale (X)
%
% Inputs:
%	X:		Data array to be autoscaled
%
% Outputs:
%	X_s:	Autoscaled data array
%	mu:		Means of the variables in X (row vector)
%	sigma:	Standard deviations of the variables in X (row vector)

% Means
mu = mean(X);
% Standard deviations for output
sigma = std(X);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Find logical columns
% logical_cols = all(X == 0 | X == 1, 1);
% % Scaling factor of logical columns are sqrt of probability of 1s in columns
% sigma_lcols = sqrt(sum(X, 1)/size(X, 1));
% % Join standard deviations of numerical and logical variables
% sigma = sigma.*~logical_cols + sigma_lcols.*logical_cols;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Standard deviations for autoscaling
sigma_s = sigma;

% Find very small standard deviations
nrm = sqrt(trace(X'*X));
nrm(nrm == 0) = 1/eps;
% Indexes for replacement
repl_std = (sigma./nrm) < eps*3;

% Replace useless standard deviations
sigma(repl_std) = 0;
sigma_s(repl_std) = inf;

% Mean centring and standard deviation scaling
X_s = (X - mu)./repmat(sigma_s, size(X, 1), 1);
end
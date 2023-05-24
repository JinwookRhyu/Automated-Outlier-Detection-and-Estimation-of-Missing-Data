function X_s = scale_by (X, mu, sigma)
% SCALE_BY Data arrays scaling with specified means and standard deviations
%
% Syntax:
%	X_s = scale_by (X, mu, sigma)
%
% Inputs:
%	X:		Data array to be scaled
%	mu:		Means to be used for scaling (row vector)
%	sigma:	Standard deviations to be used for scaling (row vector)
%
% Outputs:
%	X_s:	Scaled data array

% Replace null standard deviations
sigma(sigma == 0) = inf;

% Mean centring and standard deviation scaling
X_s = (X - mu)./repmat(sigma, size(X, 1), 1);
end
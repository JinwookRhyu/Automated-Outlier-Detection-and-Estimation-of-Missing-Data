function X = rescale_by (X_s, mu, sigma)
% RESCALE_BY Data arrays rescaling with specified means and standard deviations
%
% Syntax:
%	X = rescale_by (X_s, mu, sigma)
%
% Inputs:
%	X_s:	Scaled data array to be rescaled
%	mu:		Means to be used for rescaling (row vector)
%	sigma:	Standard deviations to be used for rescaling (row vector)
%
% Outputs:
%	X:		Rescaled data array

% Mean recentring and standard deviation rescaling
X = X_s.*repmat(sigma, size(X_s, 1), 1) + mu;
end
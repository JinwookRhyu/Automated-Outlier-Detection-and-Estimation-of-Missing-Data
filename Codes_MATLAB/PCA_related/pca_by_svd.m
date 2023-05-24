function [P_out, T_out] = pca_by_svd (X_in, A_in)
% PCA_BY_SVD PCA model calibration using the singular value decomposition (SVD)
%	algorithm
%
% Syntax:
%	P = pca_by_svd (X, A)
%	[P, T] = pca_by_svd (X, A)
%
% Inputs:
%	X:		Data array to be used to calibrate the model
%	A:		Number of principal components
%
% Outputs:
%	P:		Loadings matrix
%	T:		Scores matrix
%
%
% NOTE
%	The data array to be used to calibrate the model must be mean-centred (each
%	varible must have zero mean) and possibly scaled to unit variance (each
%	variable must have unit variance).

%% Input assignments

X = X_in;
A = A_in;

%% Initial checks

% Check if the data array is unfolded
if size(X, 3) ~= 1
	error('The data array must be unfolded for PCA model calibration')
end
% Check if the requested number of PCs is feasible
if A > min(size(X))
	error(['The number of principal components cannot exceed min(size(X)) = '...
		num2str(min(size(X)))])
end
% Check if the data-array is mean-centered
if sum(mean(X) > 1e-9) ~= 0
	error('The data array must be mean-centred for PCA model calibration')
end
% Check if there are missing values
if sum(ismissing(X), 'all') ~= 0
	error('Missing values found: PCA cannot be calibrated')
end

% Number of observations and number of variables
[N, V] = size(X);

%% PCA model building

% Less observations than variables
if N < V
	% SVD of kernel matrix
	[~, ~, v] = svds((X*X')/(V - 1), A);
	% Loadings from kernel matrix
	P = X'*v;
	% Normalise loadings
	P = P/diag(sqrt(diag(P'*P)));
% Less variables than observations
else
	% SVD of covariance matrix
	[~, ~, P] = svds((X'*X)/(N - 1), A);
end

% Loadings directed as the largest component
%[~, index] = max(abs(P), [], 1);
%colsign = sign(P(index + (0:V:(A - 1)*V)));
%P = P.*colsign;

% Scores calculation
T = X*P;

%% Output assignments

P_out = P;
T_out = T;

end
function [R_out, lambda_out, EV_out, CEV_out] = get_data_rank (X_in)
% GET_DATA_RANK Computes rank of a data array to be decomposed
%
% Syntax:
%	R = get_data_rank (X)
%	[R, lambda, EV, CEV] = get_data_rank (X)
%
% Inputs:
%	X:		Data array to get the rank of
%
% Outputs:
%	R:		Rank of the data array
%	lambda:	Eigenvalues of the covariance of X
%	EV:		Variance explained by each princiapl component of X
%	CEV:	Cumulative variance explained by each princiapl component of X
%
%
% NOTE
%	The data array to get the rank of must be mean-centred (each varible must
%	have zero mean) and possibly scaled to unit variance (each variable must have
%	unit variance).

%% Input assignments

X = X_in;

%% Initial checks

% Check if the data array is unfolded
if size(X, 3) ~= 1
	error('The data array must be unfolded')
end
% Check if the data-array is mean-centered
if sum(mean(X) > 1e-12) ~= 0
	error('The data array must be mean-centred')
end
% Check if there are missing values
if sum(ismissing(X), 'all') ~= 0
	error('Missing values found')
end

%% Get rank of data

% Number of observations and number of variables
[N, V] = size(X);

% Singular values of X
if N >= V	% tall array: limited by variables
	s = svd(X);
else		% large array: limited by observations
	s = svd(X');
end

% Scaled eigenvalues
if N > 1
	lambda = s.^2/(N - 1);
else
	lambda = s.^2;
end

% Data rank as the number of eigenvaules greater than the largest one multiplied
% by the non-limiting dimension of X
R = sum(lambda > (lambda(1)*max([N, V])*eps));

% Reduce eigenvalues on the data rank
lambda = lambda(1:R);

% De-sclae eigenvalues
if N > 1
	lambda = lambda*(N - 1);
end

% Explained variance and cumulative explained variance
EV = lambda/sum(diag(X'*X));
CEV = cumsum(EV);

%% Output assignments

R_out = R;
lambda_out = lambda;
EV_out = EV;
CEV_out = CEV;
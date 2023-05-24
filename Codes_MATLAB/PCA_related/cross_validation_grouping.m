function grouping_out = cross_validation_grouping (method_in, N_elems_in, varargin)
% CROSS_VALIDATION_GROUPING Generate gruoping assignments for cross validation
%
% Syntax:
%	grouping = cross_validation_grouping ('leave_one_out', N_elems)
%	grouping = cross_validation_grouping ('continuous_blocks', N_elems, G)
%	grouping = cross_validation_grouping ('venetian_blind', N_elems, G)
%	grouping = cross_validation_grouping ('venetian_blind', N_elems, G, band_thick)
%	grouping = cross_validation_grouping ('random_subsets', N_elems, G)
%
% Inputs:
%	method:		Method to be used for grouping
%	N_elems:	Number of elements to be grouped
%	G:			Number of groups to be generated (only for continuous blocks,
%				venetian blind and random subsets methods)
%	band_thick:	Thickenss of a single band of the blind (only for venetian blind
%				method, optional, default = 1)
%
% Outputs:
%	grouping:	Vector of the assignements of objects to groups
%
%
% NOTE
%	The input argument method can only assume three values:
%		'leave_one_out':		leave-one-out cross-validation
%		'continuous_blocks':	continuous blocks k-fold cross-validation
%		'venetian_blind':		venetian blind k-fold cross-validation
%		'random_subsets':		random assignment k-fold cross-validation
%	Note that leave-k-out cross validation is not directly supported, but can be
%	obtained by computing the number of groups from a previously declared number
%	of samples to leave out at every iterations.

%% Input assignments

method = method_in;
N_elems = N_elems_in;

%% Optional arguments development

% Optionals initialisation
G = [];
band_thick = [];

% Development cycle
if length(varargin) == 1
	G = varargin{1};
elseif length(varargin) == 2
	G = varargin{1};
	band_thick = varargin{2};
end

%% Initial checks

% Check consistency of inputs
switch method
	case 'leave_one_out'
		if isempty(G) && isempty(band_thick)
		else
			warning(['The leave-one-out method does not require to specify '...
				'any other argument besides N_elems'])
		end
	case 'continuous_blocks'
		if isempty(G)
			error(['The continuous blocks method requires to specify the '...
				'number of blocks'])
		elseif ~isempty(band_thick)
			warning(['The continuous blocks method does not require to '...
				'specify the thickness of splits'])
		end
	case 'venetian_blind'
		if isempty(G)
			error(['The venetian blind method requires to specify at least '...
				'the number of splits'])
		elseif isempty(band_thick)
			band_thick = 1;
		end
	case 'random_subsets'
		if isempty(G)
			error(['The random subsets method requires to specify the '...
				'number of blocks'])
		elseif ~isempty(band_thick)
			warning(['The continuous blocks method does not require to '...
				'specify the thickness of splits'])
		end
	otherwise
		error(['Supported methods for cross-validation grouping are '...
			'leave-one-out, continuous blocks, venetian blind and ranomdn '...
			'subsets'])
end

%% Groups generation

% Grouping of elements in different methods
switch method
	case 'leave_one_out'
		grouping = (1:N_elems)';
	case 'continuous_blocks'
		grouping = floor(((1:N_elems)' - 1)/(N_elems/G)) + 1;
	case 'venetian_blind'
		grouping = mod(floor(((1:N_elems)' - 1)/band_thick), G) + 1;
	case 'random_subsets'
		grouping = floor(((randperm(N_elems))' - 1)/(N_elems/G)) + 1;
end

%% Output assignments

grouping_out = grouping;

end
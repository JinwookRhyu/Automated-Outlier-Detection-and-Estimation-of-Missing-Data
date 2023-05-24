function Xout = preprocessing_A0(Xin, variables_mask, observations_mask)
    assert(size(Xin,2) == numel(variables_mask), ['Length of variable_mask: ', num2str(numel(variables_mask)),  ' does not match with the number of variables: ', num2str(size(Xin,2))]);
    assert(size(Xin,1) == numel(observations_mask), ['Length of observations_mask: ', num2str(numel(observations_mask)),  ' does not match with the number of observations: ', num2str(size(Xin,1))]);
    Xout = Xin(observations_mask, variables_mask);
end
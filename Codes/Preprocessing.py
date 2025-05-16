import numpy as np
import pandas as pd
from skimage import measure

# Generate a subset using only the ones in variables_mask and observations_mask
def preprocessing_A0(X, variables_mask, observations_mask):
    assert X.shape[1] == len(variables_mask), f"Length of variable_mask: {len(variables_mask)} does not match with the number of variables: {X.shape[1]}"
    assert X.shape[0] == len(observations_mask), f"Length of observations_mask: {len(observations_mask)} does not match with the number of observations: {X.shape[0]}"
    X = X[:, variables_mask]
    X = X[observations_mask, :]
    return X

# Temporary imputation
def preprocessing_A1(X, Time, method):
    method_list = ['mean', 'interpolation', 'last_observed']
    assert any(method == x for x in method_list), f"Selected method: {method} is not included in {method_list}"
    if method == 'mean':
        for j in range(X.shape[1]):
            indnan = np.argwhere(pd.isnull(X[:, j]))
            colmean = np.nanmean(X[:, j])
            X[indnan, j] = colmean
    elif method == 'interpolation':
        for j in range(X.shape[1]):
            all_labels = measure.label(pd.isnull(X[:, j]))
            numgroups = np.max(all_labels)
            for k in range(1, numgroups + 1):
                indnan = np.argwhere(all_labels == k)
                if indnan[-1] != X.shape[0] - 1:
                    if indnan[0] != 0:
                        Time_init = Time[np.where(all_labels == k)[0][0] - 1]
                        Time_end = Time[np.where(all_labels == k)[0][-1] + 1]
                        value_init = X[np.where(all_labels == k)[0][0] - 1, j]
                        value_end = X[np.where(all_labels == k)[0][-1] + 1, j]
                        for l in range(len(indnan)):
                            X[indnan[l], j] = value_init + (value_end - value_init) / (Time_end - Time_init) * (Time[indnan[l]] - Time_init)
                    else:
                        value_init = X[len(np.where(all_labels == k)[0]), j]
                        X[indnan, j] = value_init
                else:
                    value_init = X[np.where(all_labels == k)[0][0] - 1, j]
                    X[indnan, j] = value_init
    elif method == 'last_observed':
        for j in range(X.shape[1]):
            all_labels = measure.label(pd.isnull(X[:, j]))
            numgroups = np.max(all_labels)
            for k in range(1, numgroups + 1):
                indnan = np.argwhere(all_labels == k)
                if indnan[0] != 0:
                    value_init = X[np.where(all_labels == k)[0][0] - 1, j]
                    X[indnan, j] = value_init
                else:
                    value_init = X[len(np.where(all_labels == k)[0]), j]
                    X[indnan, j] = value_init
    return X
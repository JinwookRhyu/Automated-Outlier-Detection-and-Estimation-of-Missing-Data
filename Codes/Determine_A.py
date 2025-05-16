import numpy as np
import pandas as pd
import copy
import math
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from scipy.stats.distributions import chi2
from numpy.linalg import svd
from sklearn.model_selection import KFold, GroupKFold
from collections import Counter


def build_pca(X, A, **kwargs):
    '''
     BUILD_PCA PCA model calibration

    Syntax:
	model = build_pca (X, A)
	model = build_pca (X, A, 'Key', value)

    Inputs:
	X:		Data array to be used to calibrate the model
	A:		Number of principal components

    Outputs:
	model:	PCA model structure

    Keys and Values:
	'Preprocessing': ['none' | 'mean_centre' | 'standardize']
		Preprocessing method to be applied to the input data structure, can be:
		no scaling at all ('none'); mean-centring only ('mean_centre');
		mean-centring and scaling to unit variance ('autoscale', default); note
		that the algorithm implemented by this function requires mean centred
		data, therefore if the option 'none' is used the user is in charge of
		providing mean centred data, otherwise an error is issued
	'Contrib': ['simple' | 'absolute' | '2D']
		Kind of contributions to diagnostics, which can be simple indictors
		either positive or negative according to the direction of deviation and
		with two-tail confidence limits (approach of Miller, 1998), or purely
		positive contributions (like "squadred") that sum up the diagnostic
		values and with one-tail confidence limits (approach of Westerhuis, 2000)
		(default = 'simple')
	'ConLim': [lim]
		Confidence limit for diagnostics statistics (default = 0.95)
	'ContribLimMethod': ['norm' | 't']
		Method for computing the confidence limits of contributions to
		diagnostics, can be based on a normal distribution or on a t distribution
		(default = 'norm')

    NOTE
	A convention of the sings of loadings is imposed for reproducibility:
	principal components are always directed towards the direction for which the
	loading with the maximum absolute value ha spositive sign.
    '''
    # Input assignments
    X_unscaled = X
    varargin = kwargs

    # Initial checks
    # Check if the data array is unfolded
    assert len(X.shape) < 3, f"The data array must be unfolded for PCA model calibration"
    # Check if the requested number of PCs is feasible
    assert A <= np.min(X.shape), f"The number of principal components cannot exceed min(size(X)) = {np.min(X.shape)}"
    # Check if there are missing values
    assert np.sum(pd.isnull(X)) == 0, f"Missing values found: PCA cannot be calibrated"

    # Number of observations and number of variables
    N = X.shape[0]
    V = X.shape[1]

    ## Optional arguments development
    # Optionals initialisation
    preprocess = 'standardize'
    contrib = 'simple'
    lim = 0.95
    con_lim_method = 'norm'

    # Development cycle
    key_list = ['Preprocessing', 'Contrib', 'ConLim', 'ContribLimMethod']
    if len(varargin.keys()) > 0:
        for k, v in varargin.items():
            assert any(k == x for x in key_list), f"Selected method: {k} is not included in {key_list}"
            if k == 'Preprocessing':
                preprocess = v
                method_list = ['none', 'mean_center', 'standardize']
                assert any(preprocess == x for x in
                           method_list), f"Selected Preprocessing method: {preprocess} is not included in {method_list}"
            elif k == 'Contrib':
                contrib = v
                method_list = ['simple', 'absolute', '2D-Chiang', '2D-Alcala']
                assert any(contrib == x for x in
                           method_list), f"Selected Contrib method: {contrib} is not included in {method_list}"
            elif k == 'ConLim':
                lim = v
            elif k == 'ContribLimMethod':
                con_lim_method = v
                method_list = ['norm', 't', '2D-Alcala']
                assert any(con_lim_method == x for x in
                           method_list), f"Selected ContribLim method: {con_lim_method} is not included in {method_list}"

    # PCA model initialization
    model = {'dimensions': {}, 'data': {}, 'scaling': {}, 'parameters': {}, 'prediction': {},
             'diagnostics': {}, 'estimates': {}}
    model['dimensions'] = {'N': N, 'V': V, 'A': A}
    model['data'] = {'X': [], 'X_uns': []}
    model['scaling'] = {'mu': [], 'sigma': []}
    model['parameters'] = {'P': [], 'sigma_sq': []}
    model['prediction'] = {'T': []}
    model['diagnostics'] = {'T_sq': [], 'SRE': [], 'T_sq_con': [], 'SRE_con': []}
    model['estimates'] = {'lim': [], 'dof': [], 'lim_T_sq': [], 'lim_SRE': [], 'lim_T_sq_con': [], 'lim_SRE_con': []}

    # Preprocessing
    if preprocess == 'none':
        mu = np.zeros((1, V))
        sigma = np.ones((1, V))
        X = X_unscaled
    elif preprocess == 'mean_center':
        mu = X_unscaled.mean(axis=0)
        X = X_unscaled - mu
        sigma = np.ones((1, V))
    elif preprocess == 'standardize':
        [X, mu, sigma] = autoscale(X_unscaled)

    model['data']['X'] = X
    model['data']['X_uns'] = X_unscaled
    model['scaling']['mu'] = mu
    model['scaling']['sigma'] = sigma

    # PCA model calibration
    P, T = pca_by_svd(X=X, A=A)

    sigma_sq = np.var(T, axis=0)
    model['parameters']['P'] = P
    model['parameters']['sigma_sq'] = sigma_sq
    model['prediction']['T'] = T

    # Model performance
    SSE = np.zeros((A, 1))
    bias = np.zeros((A, 1))
    SE = np.zeros((A, 1))

    X_rec = np.matmul(T, P.T)
    E = X - X_rec

    for a in range(A):
        E_a = X - np.matmul(T[:, :a + 1], P[:, :a + 1].T)
        SSE[a] = np.sum(np.square(E_a))
        bias[a] = np.mean(E_a)
        SE[a] = np.sqrt(np.sum(np.square(E_a - bias[a])) / (N - 1))

    E_fd = E

    # Model Diagnostics
    T_sq = np.sum(np.matmul(np.square(T), np.diag(np.power(sigma_sq, -1))), axis=1)
    SRE = np.sum(np.square(E_fd), axis=1)

    if contrib == 'simple':
        lim_con = lim + (1 - lim) / 2
        T_sq_con = np.matmul(np.matmul(T, np.sqrt(np.diag(np.power(sigma_sq, -1)))), P)
        SRE_con = E_fd
    elif contrib == 'absolute':
        lim_con = lim
        T_sq_con = np.multiply(np.matmul(np.matmul(T, np.sqrt(np.diag(np.power(sigma_sq, -1)))), P), X)
        SRE_con = np.square(E_fd)
    elif contrib == '2D-Chiang':
        # Reference: Chiang, Leo H., Evan L. Russell, and Richard D. Braatz. Fault detection and diagnosis in industrial systems. Springer Science & Business Media, 2000.
        lim_con = lim
        T_sq_con = np.zeros((N, V))
        for i in range(N):
            t = np.matmul(P.T, X[i, :].T).reshape((A, 1))
            cont = np.divide(np.multiply(np.multiply(t, X[i, :]), P.T), sigma_sq.reshape((A, 1)))
            T_sq_con[i, :] = np.sum(cont, axis=0)
        SRE_con = np.square(E_fd)
    elif contrib == '2D-Alcala':
        # Reference: Alcala, Carlos F., and S. Joe Qin. "Reconstruction-based contribution for process monitoring." Automatica 45.7 (2009): 1593-1600.
        lim_con = lim
        C_tilde = np.eye(V) - P @ P.T
        D = P @ np.diag(np.power(sigma_sq, -1)) @ P.T
        T_sq_con = np.zeros((N, V))
        SRE_con = np.zeros((N, V))
        for i in range(V):
            xi = np.zeros(V)
            xi[i] = 1

            numerator_T_sq = (X @ D @ xi) ** 2
            denominator_T_sq = xi @ D @ xi
            if denominator_T_sq > 1e-10:
                T_sq_con[:,i] = numerator_T_sq / denominator_T_sq

            numerator_SRE = (X @ C_tilde @ xi) ** 2
            denominator_SRE = xi @ C_tilde @ xi
            if denominator_SRE > 1e-10:
                SRE_con[:,i] = numerator_SRE / denominator_SRE

    model['diagnostics']['T_sq'] = T_sq
    model['diagnostics']['SRE'] = SRE
    model['diagnostics']['T_sq_con'] = T_sq_con
    model['diagnostics']['SRE_con'] = SRE_con

    dof = np.array([x + 2 for x in range(A)])

    DOF = 2 * (T_sq.mean()) ** 2 / T_sq.var()
    scalef = T_sq.mean() / DOF
    lim_T_sq = scalef * chi2.ppf(lim, DOF)

    DOF = 2 * (SRE.mean()) ** 2 / SRE.var()
    scalef = SRE.mean() / DOF
    lim_SRE = scalef * chi2.ppf(lim, DOF)


    if con_lim_method == 'norm':
        lim_T_sq_con = st.norm.ppf(lim_con, np.mean(T_sq_con, axis=0), np.std(T_sq_con, axis=0))
        lim_SRE_con = st.norm.ppf(lim_con, np.mean(SRE_con, axis=0), np.std(SRE_con, axis=0))
    elif con_lim_method == 't':
        DOF = N - dof
        t_cl = st.t.ppf(lim_con, DOF)
        lim_T_sq_con = np.sqrt(np.diag(np.matmul(T_sq_con.T, T_sq_con)) / DOF) * t_cl
        lim_SRE_con = np.sqrt(np.diag(np.matmul(SRE_con.T, SRE_con)) / DOF) * t_cl
    elif con_lim_method == '2D-Alcala':
        # Reference: Alcala, Carlos F., and S. Joe Qin. "Reconstruction-based contribution for process monitoring." Automatica 45.7 (2009): 1593-1600.
        C_tilde = np.eye(V) - P @ P.T
        D = P @ np.diag(np.power(sigma_sq, -1)) @ P.T
        lim_T_sq_con = np.zeros(V)
        lim_SRE_con = np.zeros(V)

        for i in range(V):
            xi = np.zeros(V)
            xi[i] = 1

            numerator_T_sq = xi @ D @ np.cov(X.T) @ D @ xi
            denominator_T_sq = xi @ D @ xi
            g_T_sq = numerator_T_sq / denominator_T_sq
            lim_T_sq_con[i] = g_T_sq * chi2.ppf(lim, df=1)

            numerator_SRE = xi @ C_tilde @ np.cov(X.T) @ C_tilde @ xi
            denominator_SRE = xi @ C_tilde @ xi
            g_SRE = numerator_SRE / denominator_SRE
            lim_SRE_con[i] = g_SRE * chi2.ppf(lim, df=1)

    model['estimates']['lim'] = lim
    model['estimates']['dof'] = dof
    model['estimates']['lim_T_sq'] = lim_T_sq
    model['estimates']['lim_T_SRE'] = lim_SRE
    model['estimates']['lim_T_sq_con'] = lim_T_sq_con.flatten()
    model['estimates']['lim_SRE_con'] = lim_SRE_con.flatten()

    return model


def cross_validate_pca(X, A, **kwargs):
    '''
    CROSS_VALIDATE_PCA Cross validation of PCA model in calibration

    Inputs:
	X:			Data array to be used to cross-validate the PCA model
	A:			Number of PCs to be assessed
	G:			Number of groups to be generated (only for continuous blocks and
				venetian blind methods)
	band_thick:	Thickenss of a single band of the blind (only for venetian blind
				method, optional, default = 1)

    Outputs:
	RMSECV:		Root mean squared error in cross validation
	PRESS:		Prediction error sum of squared

    Keys and Values:
	'Preprocessing': ['autoscaling' | 'meam_centring']
		Preprocessing method to be applied to the input data structure, can be
		either mean-centring and scaling to unit variance (autoscaling) or
		mean-centring only (defualt = 'autoscale')
	'G_obs':    Number of groups used for cross-validation

    NOTE
	The rationale of  cross validation is to split the dataset into G groups of
	observations and iterate over groups. At each iteration, a model is built
	with all groups but one and the excluded one is used as a validation data
	structure; errors are computed for the validation data and their sum of
	squares is the prediction erorr sum of squares (PRESS) of the group g. As all
	the observations will be used as validation samples once and only once, G
	residuls will be availbe at the end of the loop, hence the sum of all PRESS_g
	will yield the overall PRESS of the cross-validated data structure. Dividing
	the PRESS by N (number of observations) and takinf the square-root, the
	root mean square error in cross validation (RMSECV) is obtained.
    '''

    # Initial checks
    # Check if the data array is unfolded
    assert len(X.shape) < 3, f"The data array must be unfolded for PCA model calibration"
    # Check if the requested number of PCs is feasible
    assert A <= np.min(X.shape), f"The number of principal components cannot exceed min(size(X)) = {np.min(X.shape)}"
    # Check if there are missing values
    assert np.sum(pd.isnull(X)) == 0, f"Missing values found: PCA cannot be calibrated"

    # Optionals initialization
    G_obs = []
    preprocess = 'standardize'

    N = X.shape[0]
    V = X.shape[1]
    varargin = kwargs

    # Development cycle
    key_list = ['Preprocessing', 'G_obs']
    if len(varargin.keys()) > 0:
        for k, v in varargin.items():
            assert any(k == x for x in key_list), f"Selected method: {k} is not included in {key_list}"
            if k == 'Preprocessing':
                preprocess = v
                method_list = ['mean_center', 'standardize']
                assert any(preprocess == x for x in
                           method_list), f"Selected Preprocessing method: {preprocess} is not included in {method_list}"
            elif k == 'G_obs':
                G_obs = v

    # Cross-validation
    PRESS = np.zeros((A, G_obs))

    kf_cv = KFold(n_splits=G_obs, shuffle=True, random_state=11)
    for split, (ix_train, ix_test) in enumerate(kf_cv.split(X)):
        X_train = X[ix_train, :]
        X_test = X[ix_test, :]
        if preprocess == 'mean_center':
            mu = np.mean(X, axis=0)
            sigma = np.std(X, axis=0)
            X = X - mu
        elif preprocess == 'standardize':
            mu = np.mean(X_train, axis=0)
            sigma = np.std(X_train, axis=0)
            X_train = scale_by(X=X_train, mu=mu, sigma=sigma)
        X_test = scale_by(X=X_test, mu=mu, sigma=sigma)

        P, _ = pca_by_svd(X=X_train, A=A)

        rep_mat = np.identity(V)
        for a in range(A):
            # Check if deflation is possible
            # Deflate the replacement matrix and assign it to a temporary variable
            P_replaced = rep_mat - np.outer(P[:, a], P[:, a].T)
            # Check if any term on the diagonal is gone to zero or negative
            if not any(np.diag(P_replaced) < np.finfo(float).eps * 10):
                # if this not happened, keep the deflated matrix otherwise keep the old one
                rep_mat = P_replaced

            rep_a = copy.deepcopy(rep_mat)
            d = np.diag(rep_a)
            d = [np.finfo(float).eps if x < np.finfo(float).eps else x for x in d]

            for v in range(V):
                rep_a[:, v] = (1 / d[v]) * rep_a[:, v]
            PRESS[a, split] = np.mean(np.sum(np.matmul(X_test, rep_a) ** 2, axis=0))

    RMSECV = np.sqrt(np.nansum(PRESS, axis=1) / X_test.shape[0])

    return RMSECV, PRESS

def pca_by_svd(X, A):
    # Initial checks
    # Check if the data array is unfolded
    assert len(X.shape) < 3, f"The data array must be unfolded for PCA model calibration"
    # Check if the requested number of PCs is feasible
    assert A <= np.min(X.shape), f"The number of principal components cannot exceed min(size(X)) = {np.min(X.shape)}"
    # Check if there are missing values
    assert np.sum(np.isnan(X)) == 0, f"Missing values found: PCA cannot be calibrated"
    # Check if the data array is mean-centered
    assert np.max(X.mean(axis=0)) < 10**(-9), f"The data array must be mean-centered for PCA model calibration"

    [N, V] = X.shape
    if N < V:
        _, _, vh = svd(np.matmul(X, X.T) / (V - 1))
        vh = vh[:A, :]
        P = np.matmul(X.T, vh.T)
        P = np.linalg.lstsq(np.diag(np.sqrt(np.diag(P.T @ P))), P.T, rcond=None)[0].T
    else:
        _, _, P = svd(np.matmul(X.T, X) / (N - 1))
        P = P.T

    colsign = np.sign(P[np.abs(P).argmax(axis=0), [x for x in range(0, P.shape[1])]])
    P = np.multiply(P, colsign)
    P = P[:, :A]
    T = np.matmul(X, P)

    return P, T

def rescale_by(X, mu, sigma):
    X = np.multiply(X, np.tile(sigma, (X.shape[0], 1))) + mu
    return X

def scale_by(X, mu, sigma):
    sigma = [math.inf if i == 0 else i for i in sigma]
    X = np.divide((X - mu), np.tile(sigma, (X.shape[0], 1)))
    return X

def autoscale(X):
    mu_X = np.mean(X, axis=0)
    sigma_X = np.std(X, axis=0, ddof=1)
    X_scaled = (X - mu_X) / sigma_X
    return X_scaled, mu_X, sigma_X


def pca_cross_validation(X, n_splits=5, num_repeat=5, num_components_ub=10):
    """
    Perform K-Fold Cross Validation to determine the optimal number of components in PCA.

    Parameters:
    X (array): Input data matrix of shape (n_samples, n_features).
    max_components (int): Maximum number of principal components to test.
    n_splits (int): Number of folds in K-Fold cross-validation.

    Returns:
    best_n_components (int): Optimal number of principal components.
    errors (array): Array of average reconstruction errors for each component count.
    """

    max_components = min(int(np.floor(X.shape[0] * (1 - 1 / n_splits))), num_components_ub)
    PRESS = np.empty((max_components, n_splits * num_repeat))
    g = 0

    for kk in range(num_repeat):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=kk)

        # K-Fold cross-validation
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index, :], X[test_index, :]

            # Preprocessing
            [X_train, mu, sigma] = autoscale(X_train)
            X_test = scale_by(X_test, mu, sigma)

            # Fit PCA on the training data
            [P, T] = pca_by_svd(X_train, max_components)

            # Initialize the replacement matrix
            rep_mat = np.identity(X.shape[1])
            for a in range(max_components):
                # Check if deflation is possible
                # Deflate the replacement matrix and assign it to a temporary variable
                P_replaced = rep_mat - np.outer(P[:, a], P[:, a].T)
                # Check if any term on the diagonal is gone to zero or negative
                if not any(np.diag(P_replaced) < np.finfo(float).eps * 10):
                    # if this not happened, keep the deflated matrix otherwise keep the old one
                    rep_mat = P_replaced

                rep_a = copy.deepcopy(rep_mat)
                d = np.diag(rep_a)
                d = [np.finfo(float).eps if x < np.finfo(float).eps else x for x in d]

                for v in range(X.shape[1]):
                    rep_a[:, v] = (1 / d[v]) * rep_a[:, v]
                PRESS[a, g] = np.mean(np.sum(np.matmul(X_test, rep_a) ** 2, axis=0))
            g = g + 1

    # Remove splits where such case is outlier for representing PRESS
    PRESS_outlier_ind = np.zeros(PRESS.shape)
    for a in range(0, max_components):
        lim_PRESS_con = st.norm.ppf(0.999, np.mean(PRESS[a,:]), np.std(PRESS[a,:]))
        PRESS_outlier_ind[a,:] = PRESS[a,:] > lim_PRESS_con

    PRESS_outlier_colind = np.sum(PRESS_outlier_ind, axis=0) == max_components
    PRESS = PRESS[:, ~PRESS_outlier_colind]

    return PRESS

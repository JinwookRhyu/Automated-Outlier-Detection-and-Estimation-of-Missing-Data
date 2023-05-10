import numpy as np
import copy
import math
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from scipy.stats.distributions import chi2
from numpy.linalg import svd
from sklearn.model_selection import KFold


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
	'Preprocessing': ['none' | 'mean_centre' | 'autoscale']
		Preprocessing method to be applied to the input data structure, can be:
		no scaling at all ('none'); mean-centring only ('mean_centre');
		mean-centring and scaling to unit variance ('autoscale', default); note
		that the algorithm implemented by this function requires mean centred
		data, therefore if the option 'none' is used the user is in charge of
		providing mean centred data, otherwise an error is issued
	'Algorithm': ['svd' | 'nipals']
		Algorithm to be used for PCA calibration, can be either singular value
		decomposition (SVD) or non-linear iteratrive partial least squared
		(NIPaLS) (default = 'svd')
	'Tol': [tol]
		Tolerance for convergence as 2-norm on relative scores variation
		(NIPaLS algorithm only, default = 1e-15)
	'MaxIter': [max_iter]
		Maximum number of iterations allowed (NIPaLS algorithm only,
		default = 100)
	'ErrBasedOn': ['unscaled', 'scaled']
		Scaling of data reconstruction, whether is should be returned and
		computed from scaled or unscaled entities (default = 'unscaled'); note
		that SSE, RMSE, bias and SE will be reported as scaled even in the
		unscaled errors are requested as they are meant to assess the performance
		of the model
	'Contrib': ['simple' | 'absolute']
		Kind of contributions to diagnostics, which can be simple indictors
		either positive or negative according to the direction of deviation and
		with two-tail confidence limits (approach of Miller, 1998), or purely
		positive contributions (like "squadred") that sum up the diagnostic
		values and with one-tail confidence limits (approach of Westerhuis, 2000)
		(default = 'simple')
	'ConLim': [lim]
		Confidence limit for diagnostics statistics (default = 0.95)
	'DOFMethod:['naive']
		Method for computing degrees of freedom of the model, can be either based
		on the number of latent variables ('naive', dof = A + 1, default)
		WILL BE EXTENDED IN THE FUTURE
	'TsqLimMethod': ['chisq' | 'F']
		Method for computing the confidence limits on T_sq and T_sq, can be
		either the chi squared distribution method (default) or the F
		distribution method
	'SqErrLimMethod': ['chisq' | 'jack_mod']
		Method for computing the confidence limit on SRE_X, can be either the
		chi squared distribution method (default) or the Jackson-Mudholkar
    	equation
	'ContribLimMethod': ['norm' | 't']
		Method for computing the confidence limits of contributions to
		diagnostics, can be based on a normal distribution or on a t distribution
		(default = 'norm')
	'EllipseForPlots: ['two' | 'full']
		Method for computing the semiaxes of the confidence ellipse for score
		plot, compute from a F distribution with considering only two principal
		components (A_F = 2) or all the requested ones (A_F = A, default)
	'ObsNames': [obs_names]
		Names of the observations as chars in a cell array (default are
		progressive numerical identifiers prefixed by the letter O)
	'XVarNames': [X_var_names]
		Names of the variables as chars in a cell array (default are progressive
		numerical identifiers prefixed by the letter X)

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
    assert np.sum(np.isnan(X)) == 0, f"Missing values found: PCA cannot be calibrated"

    # Number of observations and number of variables
    N = X.shape[0]
    V = X.shape[1]

    ## Optional arguments development
    # Optionals initialisation
    preprocess = 'standardize'
    alg = 'svd'
    tol = 1e-15
    max_iter = 100
    err_on = 'unscaled'
    contrib = 'simple'
    lim = 0.95
    dof_method = 'naive'
    Tsq_lim_method = 'chisq'
    SqE_lim_method = 'chisq'
    con_lim_method = 'norm'
    l_kind = 'full'

    # Development cycle
    key_list = ['Preprocessing', 'Algorithm', 'Tol', 'MaxIter', 'ErrBasedOn', 'Contrib', 'ConLim', 'DOFMethod',
                'TsqLimMethod', 'SqErrLimMethod', 'ContribLimMethod', 'EllipseForPlots']
    if len(varargin.keys()) > 0:
        for k, v in varargin.items():
            assert any(k == x for x in key_list), f"Selected method: {k} is not included in {key_list}"
            if k == 'Preprocessing':
                preprocess = v
                method_list = ['none', 'mean_center', 'standardize']
                assert any(preprocess == x for x in
                           method_list), f"Selected Preprocessing method: {preprocess} is not included in {method_list}"
            elif k == 'Algorithm':
                alg = v
                method_list = ['svd', 'nipals']
                assert any(
                    alg == x for x in method_list), f"Selected Algorithm method: {alg} is not included in {method_list}"
            elif k == 'Tol':
                tol = v
            elif k == 'MaxIter':
                max_iter = v
            elif k == 'ErrBasedOn':
                err_on = v
                method_list = ['scaled', 'unscaled']
                assert any(err_on == x for x in
                           method_list), f"Selected ErrBasedOn method: {err_on} is not included in {method_list}"
            elif k == 'Contrib':
                contrib = v
                method_list = ['simple', 'absolute', '2D']
                assert any(contrib == x for x in
                           method_list), f"Selected Contrib method: {contrib} is not included in {method_list}"
            elif k == 'ConLim':
                lim = v
            elif k == 'DOFMethod':
                dof_method = v
                method_list = ['naive']
                assert any(dof_method == x for x in
                           method_list), f"Selected DOF method: {dof_method} is not included in {method_list}"
            elif k == 'TsqLimMethod':
                Tsq_lim_method = v
                method_list = ['chisq', 'F']
                assert any(Tsq_lim_method == x for x in
                           method_list), f"Selected TsqLim method: {Tsq_lim_method} is not included in {method_list}"
            elif k == 'SqErrLimMethod':
                SqE_lim_method = v
                method_list = ['chisq', 'jack_mod']
                assert any(SqE_lim_method == x for x in
                           method_list), f"Selected SqELim method: {SqE_lim_method} is not included in {method_list}"
            elif k == 'ContribLimMethod':
                con_lim_method = v
                method_list = ['norm', 't']
                assert any(con_lim_method == x for x in
                           method_list), f"Selected ContribLim method: {con_lim_method} is not included in {method_list}"
            elif k == 'EllipseForPlots':
                l_kind = v
                method_list = ['two', 'full']
                assert any(l_kind == x for x in
                           method_list), f"Selected EEllipseForPlots method: {l_kind} is not included in {method_list}"

    # PCA model initialization
    model = {'dimensions': {}, 'info': {}, 'data': {}, 'scaling': {}, 'parameters': {}, 'prediction': {},
             'performance': {}, 'diagnostics': {}, 'estimates': {}}
    model['dimensions'] = {'N': N, 'V': V, 'A': A}
    model['info'] = {'preprocessing': preprocess, 'algorithm': alg, 'error_based_on': err_on,
                     'contribution_method': contrib,
                     'dof_method': dof_method, 'Tsq_lim_method': Tsq_lim_method, 'SqE_lim_method': SqE_lim_method,
                     'con_lim_method': con_lim_method, 'l_kind': l_kind}
    model['data'] = {'X': [], 'X_uns': []}
    model['scaling'] = {'mu': [], 'sigma': []}
    model['parameters'] = {'P': [], 'sigma_sq': []}
    model['prediction'] = {'T': [], 'X_rec': [], 'E': []}
    model['performance'] = {'EV': [], 'CEV': [], 'SSE': [], 'RMSE': [], 'bias': [], 'SE': [], 'lambda': []}
    model['diagnostics'] = {'T_sq': [], 'SRE': [], 'T_sq_con': [], 'SRE_con': []}
    model['estimates'] = {'lim': [], 'dof': [], 'lim_T_sq': [], 'lim_SRE': [], 'lim_T_sq_con': [], 'lim_SRE_con': [],
                          'l': []}

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
        scaler = StandardScaler()
        X = scaler.fit_transform(X_unscaled)
        mu = scaler.mean_
        sigma = scaler.var_

    model['data']['X'] = X
    model['data']['X_uns'] = X_unscaled
    model['scaling']['mu'] = mu
    model['scaling']['sigma'] = sigma

    # PCA model calibration
    if alg == 'svd':
        P, T = pca_by_svd(X=X, A=A)
    elif alg == 'nipals':
        P, T = pca_by_nipals(X=X, A=A, Tol=tol, MaxIter=max_iter)

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
    ssqT = np.diag(np.matmul(T.T, T))
    EV = ssqT / np.trace(np.matmul(X.T, X))
    CEV = np.cumsum(EV)

    for a in range(A):
        E_a = X - np.matmul(T[:, :a + 1], P[:, :a + 1].T)
        SSE[a] = np.sum(np.square(E_a))
        bias[a] = np.mean(E_a)
        SE[a] = np.sqrt(np.sum(np.square(E_a - bias[a])) / (N - 1))

    RMSE = np.sqrt(SSE / N)
    lamb = ssqT / (N - 1)
    E_fd = E

    if err_on == 'unscaled':
        X_rec = np.matmul(X_rec, np.diag(sigma)) + mu
        E = np.matmul(E, np.diag(sigma))

    model['prediction']['X_rec'] = X_rec
    model['prediction']['E'] = E
    model['performance']['EV'] = EV
    model['performance']['CEV'] = CEV
    model['performance']['SSE'] = SSE
    model['performance']['RMSE'] = RMSE
    model['performance']['bias'] = bias
    model['performance']['SE'] = SE
    model['performance']['lambda'] = lamb

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
    elif contrib == '2D':
        lim_con = lim
        T_sq_con = np.zeros((N, V))
        for i in range(N):
            t = np.matmul(P.T, X[i,:].T).reshape((A,1))
            cont = np.divide(np.multiply(np.multiply(t, X[i,:]), P.T), sigma_sq.reshape((A,1)))
            T_sq_con[i,:] = np.sum(cont, axis=0)
        SRE_con = np.square(E_fd)

    model['diagnostics']['T_sq'] = T_sq
    model['diagnostics']['SRE'] = SRE
    model['diagnostics']['T_sq_con'] = T_sq_con
    model['diagnostics']['SRE_con'] = SRE_con

    dof = [x + 2 for x in range(A)]
    if Tsq_lim_method == 'chisq':
        DOF = 2 * (T_sq.mean()) ** 2 / T_sq.var()
        scalef = T_sq.mean() / DOF
        lim_T_sq = scalef * chi2.ppf(lim, DOF)
    elif Tsq_lim_method == 'F':
        lim_T_sq = (A * (N - 1) / (N - A)) * st.f.isf(lim, A, N - A)

    if SqE_lim_method == 'chisq':
        DOF = 2 * (SRE.mean()) ** 2 / SRE.var()
        scalef = SRE.mean() / DOF
        lim_SRE = scalef * chi2.ppf(lim, DOF)
    elif SqE_lim_method == 'jack_mod':
        z = st.norm.ppf(lim)
        theta = np.zeros((1, 3))
        for j in range(3):
            theta[j] = np.sum(np.power(np.var(E_fd, axis=0), j + 1))
        h0 = 1 - (2 * theta[0] * theta[2]) / (3 * (theta[1]) ** 2)
        lim_SRE = theta[0] * (
                    1 - theta[1] * h0 * (1 - h0) / (theta[0]) ** 2 + np.sqrt(z * 2 * theta[1] * (h0) ** 2) / theta[
                0]) ** (1 / h0)

    if con_lim_method == 'norm':
        #lim_T_sq_con = st.norm.ppf(lim_con, np.zeros((1, V)), np.std(T_sq_con, axis=0))
        #lim_SRE_con = st.norm.ppf(lim_con, np.zeros((1, V)), np.std(SRE_con, axis=0))
        lim_T_sq_con = st.norm.ppf(lim_con, np.mean(T_sq_con, axis=0), np.std(T_sq_con, axis=0))
        lim_SRE_con = st.norm.ppf(lim_con, np.mean(SRE_con, axis=0), np.std(SRE_con, axis=0))
    elif con_lim_method == 't':
        DOF = N - dof
        t_cl = st.t.ppf(lim_con, DOF)
        lim_T_sq_con = np.sqrt(np.diag(np.matmul(T_sq_con.T, T_sq_con)) / DOF) * t_cl
        lim_SRE_con = np.sqrt(np.diag(np.matmul(SRE_con.T, SRE_con)) / DOF) * t_cl

    if l_kind == 'two':
        assert Tsq_lim_method == 'F', f"Confidence ellipse requested as based on two principal components only, possible only using F distribution-based confidence limit for T_sq: the limit will be inconsistent"
        lim_T_sq_for_plots = 2 * (N - 1) / (N - 2) * st.f.ppf(lim, 2, N - 2)
    elif l_kind == 'full':
        lim_T_sq_for_plots = lim_T_sq

    l = np.sqrt(sigma_sq * lim_T_sq_for_plots)

    model['estimates']['lim'] = lim
    model['estimates']['dof'] = dof
    model['estimates']['lim_T_sq'] = lim_T_sq
    model['estimates']['lim_T_SRE'] = lim_SRE
    model['estimates']['lim_T_sq_con'] = lim_T_sq_con.flatten()
    model['estimates']['lim_SRE_con'] = lim_SRE_con.flatten()
    model['estimates']['l'] = l

    return model


def cross_validate_pca(X, A, **kwargs):
    '''
    CROSS_VALIDATE_PCA Cross validation of PCA model in calibration

    Inputs:
	X:			Data array to be used to cross-validate the PCA model
	A:			Number of PCs to be assessed
	method:		Method to be used for cross-validation (grouping of observations)
	G:			Number of groups to be generated (only for continuous blocks and
				venetian blind methods)
	band_thick:	Thickenss of a single band of the blind (only for venetian blind
				method, optional, default = 1)

    Outputs:
	RMSECV:		Root mean squared error in cross validation
	PRESS:		Prediction error sum of squared
	groups:		Vector of the assignements of observations to groups

    Keys and Values:
	'Preprocessing': ['autoscaling' | 'meam_centring']
		Preprocessing method to be applied to the input data structure, can be
		either mean-centring and scaling to unit variance (autoscaling) or
		mean-centring only (defualt = 'autoscale')
	'Algorithm': ['svd' | 'nipals']
		Algorithm to be used for PCA calibration, can be either singular value
		decomposition (SVD) or non-linear iteratrive partial least squared
		(NIPaLS) (default = 'svd')
	'Tol': [tol]
		Tolerance for convergence as 2-norm on scores (NIPaLS algorithm only,
		default = 1e-12)
	'MaxIter': [max_iter]
		Maximum number of iterations allowed (NIPaLS algorithm only,
		default = 100)
	'Kind': ['rkf' | 'ekf' | 'ekf_fast']
		Kind of cross-validation procedure to follow, can be row-wise k-fold
		(rkf), element-wise k-fold (ekf) or element-wise k-fold based on fast PCA
		(ekf_fast) (default = 'ekf_fast')
	'VarMethod': ['leave_one_out' | 'continuous_blocks' | 'venetian_blind']
		Method for grouping variables during replacement in ekf cross-validation
		(ekf cross-validation only, default = 'leave_one_out'); this key has no
		effect if Kind is set to 'rkf' or to 'ekf_fast'
	'VarGroups': [G_var]
		Number of groups to be generated for grouping of variables (defualt =
		number_of_variables'); this key has no effect if only VarMethod is set to
		'leave_one_out'
	'VarThickness: [band_thick_var]
		Thickenss of a single band of the blind for grouping of variables
		(defualt = 1); this key has no effect if only VarMethod is set to
		'leave_one_out' or to 'continuous_blocks'


    NOTE
	The input argument method can only assume three values:
		'leave_one_out':		leave-one-out cross-validation
		'continuous_blocks':	continuous blocks k-fold cross-validation
		'venetian_blind':		venetian blind k-fold cross-validation
	Note that leave-k-out cross validation is not directly supported, but can be
	obtained by computing the number of groups from a previously declared number
	of samples to leave out at every iterations.

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
    assert np.sum(np.isnan(X)) == 0, f"Missing values found: PCA cannot be calibrated"

    # Optionals initialization
    G_obs = []
    preprocess = 'standardize'
    alg = 'svd'
    tol = 1e-12
    kind = 'ekf_fast'
    method_var = ''
    G_var = []
    band_thick_var = []

    N = X.shape[0]
    V = X.shape[1]
    varargin = kwargs

    # Development cycle
    key_list = ['Preprocessing', 'Algorithm', 'G_obs', 'Tol', 'MaxIter', 'Kind', 'VarMethod', 'VarGroups',
                'VarThickness']
    if len(varargin.keys()) > 0:
        for k, v in varargin.items():
            assert any(k == x for x in key_list), f"Selected method: {k} is not included in {key_list}"
            if k == 'Preprocessing':
                preprocess = v
                method_list = ['mean_center', 'standardize']
                assert any(preprocess == x for x in
                           method_list), f"Selected Preprocessing method: {preprocess} is not included in {method_list}"
            elif k == 'Algorithm':
                alg = v
                method_list = ['svd', 'nipals']
                assert any(
                    alg == x for x in method_list), f"Selected Algorithm method: {alg} is not included in {method_list}"
            elif k == 'G_obs':
                G_obs = v
            elif k == 'Tol':
                tol = v
            elif k == 'MaxIter':
                max_iter = v
            elif k == 'Kind':
                kind = v
                method_list = ['rkf', 'ekf', 'ekf_fast']
                assert any(kind == x for x in
                           method_list), f"Selected ErrBasedOn method: {err_on} is not included in {method_list}"
            elif k == 'VarMethod':
                method_var = v
                method_list = ['leave_one_out', 'continuous_blocks', 'venetian_blind']
                assert any(method_var == x for x in
                           method_list), f"Selected Contrib method: {contrib} is not included in {method_list}"
            elif k == 'VarGroups':
                G_var = v
            elif k == 'VarThickness':
                band_thick_var = v

    # rkf cross - validation does not require variable grouping
    if (kind == 'rkf') & (method_var != ''):
        print(['Cross-validation of rkf kind does not require variable grouping, resetting'])
        method_var = ''
        G_var = []
        band_thick_var = []
    # Default values for G_var and / or band_thick_var
    if kind == 'ekf':
        if ((method_var == 'continuous_blocks') | (method_var == 'venetian_blind')) & G_var == []:
            G_var = V
        if (method_var == 'venetian_blind') & (band_thick_var == []):
            band_thick_var = 1
    # ekf_fast cross-validation works only with leave_one_out variable grouping
    if kind == 'ekf_fast':
        if method_var == '':
            method_var = 'leave_one_out'
        elif method_var != 'leave_one_out':
            #print(['Cross-validation of ekf_fast kind can une only leave_one_out variable grouping, resetting'])
            method_var = 'leave_one_out'

    # Cross-validation
    PRESS = np.zeros((A, G_obs))
    PRESS_V = np.zeros((V, A, G_obs))
    alpha_CV = np.zeros((V, A, G_obs))
    P_CV = np.zeros((V, A, G_obs))

    #grouping = [x % 7 + 1 for x in range(X.shape[0])]
    #for split in range(7):
    kf_cv = KFold(n_splits=G_obs, shuffle=True, random_state=11)
    for split, (ix_train, ix_test) in enumerate(kf_cv.split(X)):
        #ix_train = [x != split + 1 for x in grouping]
        #ix_test = [x == split + 1 for x in grouping]
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
        if alg == 'svd':
            P, _ = pca_by_svd(X=X_train, A=A)
        elif alg == 'nipals':
            P, _ = pca_by_nipals(X=X_train, A=A)

        if kind == 'rkf':
            T_test = np.matmul(X_test, P)
            for a in range(A):
                E_val_a = X_test - np.matmul(T_test[:, :a + 1], P[:, :a + 1].T)
                PRESS[a, split] = np.sum(E_val_a ** 2)
                PRESS_V[:, a, split] = np.sum(E_val_a ** 2, axis=0)
                alpha_CV[:, a, split] = np.diag(np.matmul(P[:, :a + 1], P[:, :a + 1].T))
                P_CV[:, a, split] = np.sign(P[0, a]) * P[:, a]
        elif kind == 'ekf_fast':
            # Initialise the replacement matrix rep_mat = eye(V)
            # Loop on PCs
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

    N = X.shape[0]
    V = X.shape[1]

    if N < V:
        _, _, vh = svd(np.matmul(X, X.T) / (V - 1))
        vh = vh.T
        P = np.matmul(X.T, vh)
    else:
        _, _, P = svd(np.matmul(X.T, X) / (V - 1))
        P = P.T

    colsign = np.sign(P[np.abs(P).argmax(axis=0), [x for x in range(0, V)]])
    P = np.multiply(P, colsign)
    P = P[:, :A]
    T = np.matmul(X, P)
    return P, T

def pca_by_nipals(X, A, **kwargs):
    # Initial checks
    # Check if the data array is unfolded
    assert len(X.shape) < 3, f"The data array must be unfolded for PCA model calibration"
    # Check if the requested number of PCs is feasible
    assert A <= np.min(X.shape), f"The number of principal components cannot exceed min(size(X)) = {np.min(X.shape)}"
    # Check if there are missing values
    assert np.sum(np.isnan(X)) == 0, f"Missing values found: PCA cannot be calibrated"
    # Check if the data array is mean-centered
    assert np.max(X.mean(axis=0)) < 10 ** (-9), f"The data array must be mean-centered for PCA model calibration"

    tol = 10**(-15)
    max_iter = 100
    varargin = kwargs
    key_list = ['Tol', 'MaxIter']
    if len(varargin.keys()) > 0:
        for k, v in varargin.items():
            assert any(k == x for x in key_list), f"Selected key: {k} is not included in {key_list}"
            if k == 'Tol':
                tol = v
            elif k == 'MaxIter':
                max_iter = v

    N = X.shape[0]
    V = X.shape[1]
    T = np.zeros((N, A))
    P = np.zeros((V, A))
    E = X

    for a in range(A):
        t = np.ones((N, 1))
        iter = 0
        err = tol + 1
        while err >= tol & iter < max_iter:
            iter = iter + 1
            p = np.matmul(E.T, t) / (np.matmul(t.T, t))
            p = p / np.sqrt(np.matmul(p.T, p))
            t_new = E * p
            err = np.linalg(t_new - t) / np.linalg(t_new)
            t = t_new
        if iter == max_iter:
            print(f"Maximum number of iterations reached calculating PC: {max_iter}")
        p = np.matmul(E.T, t) / np.matmul(t.T, t)
        p = p / np.sqrt(np.matmul(p.T, p))
        index = np.argmax(np.abs(p))
        colsign = np.sign(p[index])
        p = p * colsign
        t = np.matmul(E, p)
        T[:, a] = t
        P[:, a] = p
        E = E - np.matmul(t, p.T)

    return P, T

def rescale_by(X, mu, sigma):
    X = np.multiply(X, np.tile(sigma, (X.shape[0], 1))) + mu
    return X

def scale_by(X, mu, sigma):
    sigma = [math.inf if i == 0 else i for i in sigma]
    X = np.divide((X - mu), np.tile(sigma, (X.shape[0], 1)))
    return X

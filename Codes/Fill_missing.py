import numpy as np
import copy
import time
from Determine_A import pca_cross_validation, rescale_by, scale_by
from Preprocessing import preprocessing_A1
from Algorithms import mifit, alternatingfit, svdimputefit, pcadafit, ppcafit, ppcamfit, bpcafit, svtfit, ialmfit

# Filling in missing values using various imputation algorithms
def fill_missing(X, Time, method, A, algorithm, n_splits, num_repeat, num_components_ub, SE_rule_PCA, verbose=False, verbose_alg=True):
    M = np.isnan(X)
    X_rec = copy.deepcopy(X)
    X_rec = preprocessing_A1(X=X_rec, Time=Time, method=method)
    mu = np.mean(X_rec, axis=0)
    sigma = np.std(X_rec, axis=0)
    X_scaled = scale_by(X=X_rec, mu=mu, sigma=sigma)
    A_list = np.zeros(10)

    for aa in range(10): # Repeat until the number of principal components converge
        if algorithm == 'MI':
            t0 = time.time()
            fitmodel = mifit()
        elif algorithm == 'Alternating':
            t0 = time.time()
            fitmodel = alternatingfit(A=A, n_iters=1000, verbose=verbose)
        elif algorithm == 'SVDImpute':
            t0 = time.time()
            fitmodel = svdimputefit(A=A, verbose=verbose)
        elif algorithm == 'PCADA':
            t0 = time.time()
            fitmodel = pcadafit(A=A, n_iters=1000, verbose=verbose, K=50)
        elif algorithm == 'PPCA':
            t0 = time.time()
            fitmodel = ppcafit(A=A, n_iters=1000, verbose=verbose)
        elif algorithm == 'PPCA-M':
            t0 = time.time()
            fitmodel = ppcamfit(A=A, n_iters=1000, verbose=verbose)
        elif algorithm == 'BPCA':
            t0 = time.time()
            fitmodel = bpcafit(A=A, n_iters=1000, verbose=verbose)
        elif algorithm == 'SVT':
            t0 = time.time()
            fitmodel = svtfit(A=A, n_iters=1000, verbose=verbose, tau=5*X.shape[0])
        elif algorithm == 'ALM':
            t0 = time.time()
            fitmodel = ialmfit(A=A, n_iters=1000, verbose=verbose)


        fitmodel.fit(X=X_scaled, M=M)
        X_scaled_rec = fitmodel.recover()
        X_scaled = np.multiply(X_scaled, 1 - M) + np.multiply(X_scaled_rec, M)
        comp_time = time.time() - t0

        if algorithm == 'MI':
            if verbose_alg:
                print(algorithm + ' completed.')
            break
        else:
            PRESS_PCA = pca_cross_validation(X_scaled, n_splits=n_splits, num_repeat=num_repeat,
                                                          num_components_ub=num_components_ub)
            indmin_PCA = np.argmin(np.mean(PRESS_PCA, axis=1))
            A_new = np.where(np.mean(PRESS_PCA, axis=1) < np.mean(PRESS_PCA[indmin_PCA, :]) + SE_rule_PCA * np.std(
                PRESS_PCA[indmin_PCA, :], ddof=1) / np.sqrt(PRESS_PCA.shape[1]))[0][0] + 1
            A_list[aa] = A_new
            if A in A_list:
                if verbose_alg:
                    print(algorithm + ' completed. #PC converged to ' + str(A) + ' for ' + algorithm)
                break
            if verbose_alg:
                print(algorithm + ' trial ' + str(aa+1) + ' completed. #PC updated from ' + str(A) + ' to ' + str(A_new))
            A = A_new

    X_rec = rescale_by(X=X_scaled, mu=mu, sigma=sigma)
    return X_rec, A, comp_time
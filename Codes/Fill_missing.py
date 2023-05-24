import numpy as np
import copy
import time
from Determine_A import cross_validate_pca, rescale_by, scale_by
from Preprocessing import preprocessing_A1
from Algorithms import mifit, alternatingfit, svdimputefit, pcadafit, ppcafit, ppcamfit, bpcafit, svtfit, ialmfit

# Filling in missing values using various imputation algorithms
def fill_missing(X, Time, method, A, algorithm, verbose=False):
    M = np.isnan(X)
    X_rec = copy.deepcopy(X)
    X_rec = preprocessing_A1(X=X_rec, Time=Time, method=method)
    mu = np.mean(X_rec, axis=0)
    sigma = np.std(X_rec, axis=0)
    X_scaled = scale_by(X=X_rec, mu=mu, sigma=sigma)

    for aa in range(10): # Repeat until the number of principal components converge
        if algorithm == 'MI':
            t0 = time.time()
            fitmodel = mifit()
            print('MI completed')
        elif algorithm == 'Alternating':
            t0 = time.time()
            fitmodel = alternatingfit(A=A, n_iters=1000, verbose=verbose)
            print('Alternating completed')
        elif algorithm == 'SVDImpute':
            t0 = time.time()
            fitmodel = svdimputefit(A=A, verbose=verbose)
            print('SVDImpute completed')
        elif algorithm == 'PCADA':
            t0 = time.time()
            fitmodel = pcadafit(A=A, n_iters=1000, verbose=verbose, K=50)
            print('PCADA completed')
        elif algorithm == 'PPCA':
            t0 = time.time()
            fitmodel = ppcafit(A=A, n_iters=1000, verbose=verbose)
            print('PPCA completed')
        elif algorithm == 'PPCA-M':
            t0 = time.time()
            fitmodel = ppcamfit(A=A, n_iters=1000, verbose=verbose)
            print('PPCA-M completed')
        elif algorithm == 'BPCA':
            t0 = time.time()
            fitmodel = bpcafit(A=A, n_iters=1000, verbose=verbose)
            print('BPCA completed')
        elif algorithm == 'SVT':
            t0 = time.time()
            fitmodel = svtfit(A=A, n_iters=1000, verbose=verbose, tau=5*X.shape[0])
            print('SVT completed')
        elif algorithm == 'ALM':
            t0 = time.time()
            fitmodel = ialmfit(A=A, n_iters=1000, verbose=verbose)
            print('ALM completed')


        fitmodel.fit(X=X_scaled, M=M)
        X_scaled_rec = fitmodel.recover()
        X_scaled = np.multiply(X_scaled, 1 - M) + np.multiply(X_scaled_rec, M)
        comp_time = time.time() - t0

        if algorithm == 'MI':
            break
        else:
            RMSECV, _ = cross_validate_pca(X=X_scaled, A=int(np.round(0.8*np.min(X.shape))), VarMethod='venetian_blind', G_obs=7, Kind='ekf_fast')
            if A == np.argmin(RMSECV) + 1:
                break
            A = np.argmin(RMSECV) + 1

    X_rec = rescale_by(X=X_scaled, mu=mu, sigma=sigma)
    return X_rec, A, comp_time
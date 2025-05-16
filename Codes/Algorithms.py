import numpy as np
import copy
import random
import math
from numpy.linalg import inv, svd, norm
from Determine_A import scale_by, pca_by_svd



class mifit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        self.X_rec = X

        for i in range(self.V):
            self.X_rec[M[:, i], i] = np.mean(X[M[:, i] == False, i])

    def recover(self):
        return self.X_rec


## PCA-based algorithms
class alternatingfit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose

    def _init_paras(self, X, N, V, A):
        self.e_X = X
        X0 = copy.deepcopy(X)
        X0[self.M] = 0
        self.X0 = X0
        P, T = pca_by_svd(X0, A)
        self.e_T = T
        self.e_P = P
        self.e_mu = np.zeros((V, 1))
        self.e_Tdev = np.zeros((N, A, V))
        self.e_Pdev = np.zeros((V, A, N))

        self.O_list_row = []
        self.O_list_col = []
        for j in range(self.V):
            self.O_list_col.append([i for i in range(self.N) if self.M[i, j] == False])
            self.e_Tdev[self.O_list_col[j], :, j] = self.e_T[self.O_list_col[j], :]
        for i in range(self.N):
            self.O_list_row.append([j for j in range(self.V) if self.M[i, j] == False])
            self.e_Pdev[self.O_list_row[i], :, i] = self.e_P[self.O_list_row[i], :]

        self.e_mudev = np.zeros((V, N))

        self.tol = 1e-6
        self.sse = []

    def _update_T(self, N, V, A):
        for i in range(N):
            AA = self.e_Pdev[:, :, i]
            b = self.X0[i, :].T - self.e_mudev[:, i]
            self.e_T[i, :] = np.linalg.lstsq(AA, b, rcond=None)[0].T
        self.e_Tdev = np.zeros((N, A, V))
        for j in range(V):
            self.e_Tdev[self.O_list_col[j], :, j] = self.e_T[self.O_list_col[j], :]

    def _update_mu(self, N, V, A):
        self.e_mu = np.zeros((V, 1))
        self.e_mudev = np.zeros((V, N))
        for j in range(V):
            self.e_mu[j] = 1 / len(self.O_list_col[j]) * np.sum(
                np.multiply(self.e_X[:, j] - np.matmul(self.e_T, self.e_P[j, :].T), 1 - self.M[:, j]))
            self.e_mudev[j, self.O_list_col[j]] = self.e_mu[j]

    def _update_P(self, N, V, A):
        for j in range(V):
            AA = self.e_Tdev[:, :, j]
            b = self.X0[:, j] - self.e_mu[j]
            self.e_P[j, :] = np.linalg.lstsq(AA, b, rcond=None)[0].T
        self.e_Pdev = np.zeros((V, A, N))
        for i in range(N):
            self.e_Pdev[self.O_list_row[i], :, i] = self.e_P[self.O_list_row[i], :]

    def sse_obs(self, N):
        self.e_X_rec = np.matmul(self.e_T, self.e_P.T)
        SSE_obs = np.sum(np.multiply(self.e_X - self.e_X_rec, 1 - self.M) ** 2)

        return SSE_obs

    def _update(self, N, V, A):
        self._update_T(N, V, A)
        self._update_mu(N, V, A)
        self._update_P(N, V, A)
        SSE_obs = self.sse_obs(N)
        self.sse.append(SSE_obs)
        if self.verbose:
            print("SSE_obs: {}".format(SSE_obs))

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        A = self.A
        N = self.N
        V = self.V
        self._init_paras(X, N, V, A)

        for it in range(self.n_iters):
            self._update(N, V, A)
            if it >= 1:
                if np.abs((self.sse[it] - self.sse[it - 1]) / self.sse[it - 1]) < self.tol:
                    break

    def recover(self):
        return self.e_T.dot(self.e_P.T)


class svdimputefit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        self.X_rec = X

        SSE_obs_list = np.zeros((self.n_iters, 1))
        SSE_obs_old = 0
        for ii in range(self.n_iters):
            S = 1 / (self.X_rec.shape[0] - 1) * np.matmul(self.X_rec.T, self.X_rec)
            V, _, _ = svd(S)
            P = V[:, :self.A]
            T = np.matmul(self.X_rec, P)
            SSE_obs_list[ii] = np.sum(np.multiply((self.X_rec - np.matmul(T, P.T)) ** 2, (1 - M))) / np.sum((1 - M))
            SSE_obs_new = SSE_obs_list[ii]
            self.X_rec = np.multiply(self.X_rec, (1 - M)) + np.multiply(np.matmul(T, P.T), M)
            self.SSE_obs_list = SSE_obs_list

            if np.abs((SSE_obs_new - SSE_obs_old) / SSE_obs_new) < 10 ** (-6):
                break
            SSE_obs_old = SSE_obs_new

    def recover(self):
        return self.X_rec


class pcadafit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False, K=100):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose
        self.K = K

    def _init_paras(self, X, N, V, A, K):
        self.e_X = X
        X0 = copy.deepcopy(X)
        self.X0 = X0
        self.X0_scaled = scale_by(X=X0, mu=np.mean(X0, axis=0), sigma=np.std(X0, axis=0))
        self.e_P, _ = pca_by_svd(X=self.X0_scaled, A=self.A)
        self.e_X_k = np.repeat(self.e_X[:, :, np.newaxis], K, axis=2)
        self.e_P_k = np.repeat(self.e_P[:, :, np.newaxis], K, axis=2)

        self.M_list = []
        self.O_list = []
        for j in range(self.V):
            self.M_list.append([i for i in range(self.N) if self.M[i, j] == True])
            self.O_list.append([i for i in range(self.N) if self.M[i, j] == False])

        self.tol = 1e-6
        self.sse = np.zeros((self.n_iters))

    def _update_X(self):
        self.e_X_nf = np.matmul(np.matmul(self.e_X, self.e_P), self.e_P.T)
        self.e_R = self.X0 - self.e_X_nf

        for k in range(self.K):
            random.seed(k)
            self.e_X_k[:, :, k] = np.multiply(self.e_X_k[:, :, k], 1 - self.M) + np.multiply(self.e_X_nf, self.M)
            for j in range(self.V):
                self.e_X_k[self.M_list[j], j, k] = self.e_X_k[self.M_list[j], j, k] + self.e_R[
                    random.choices(self.O_list[j], k=len(self.M_list[j])), j]
        self.e_X = np.mean(self.e_X_k, axis=2)

    def _update_P(self):
        for k in range(self.K):
            _, _, self.e_P = svd(self.e_X_k[:, :, k])
            self.e_P = self.e_P.T
            self.e_P_k[:, :, k] = self.e_P[:, :self.A]

        self.e_P = np.mean(self.e_P_k, axis=2)

    def sse_obs(self):
        SSE_obs = np.sum(np.multiply(self.X0 - self.e_X_nf, 1 - self.M) ** 2) / np.sum((1 - self.M))

        return SSE_obs

    def _update(self, it):
        self._update_X()
        self._update_P()
        self.sse[it] = self.sse_obs()
        if self.verbose:
            print("SSE_obs: {}".format(self.sse[it]))

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        A = self.A
        N = self.N
        V = self.V
        K = self.K
        self._init_paras(X, N, V, A, K)

        for it in range(self.n_iters):
            self._update(it)
            if it >= 1:
                if np.abs((self.sse[it] - self.sse[it - 1]) / self.sse[it - 1]) < self.tol:
                    break

    def recover(self):
        return self.e_X


## PPCA-based algorithms
class ppcafit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose

    def _init_paras(self, X, N, V, A):
        self.e_X = X
        self.e_T = np.zeros((N, A))
        X0 = copy.deepcopy(X)
        self.X0 = X0
        self.X0_scaled = scale_by(X=X0, mu=np.mean(X0, axis=0), sigma=np.std(X0, axis=0))
        self.e_P, _ = pca_by_svd(X=self.X0_scaled, A=self.A)
        self.e_sigma_sq_old = np.var(X)
        self.e_mu = np.zeros((V, 1))
        self.e_TtT = np.zeros((A, A, N))
        self.e_W = np.zeros((A, A, N))

        self.O_list_row = []
        self.O_list_col = []
        for j in range(self.V):
            self.O_list_col.append([i for i in range(self.N) if self.M[i, j] == False])

        for i in range(self.N):
            self.O_list_row.append([j for j in range(self.V) if self.M[i, j] == False])
            self.e_W[:, :, i] = self.e_W[:, :, i] + np.matmul(self.e_P[self.O_list_row[i], :].T,
                                                              self.e_P[self.O_list_row[i],
                                                              :]) + self.e_sigma_sq_old * np.eye(A)

        self.tol = 1e-6
        self.lbs = []

    def _update_T(self, N, A):
        for i in range(N):
            self.Px = np.matmul(self.e_P[self.O_list_row[i], :].T,
                                self.e_X[i, self.O_list_row[i]].T.reshape(len(self.O_list_row[i]), 1) - self.e_mu[
                                    self.O_list_row[i]])
            self.e_T[i, :] = np.matmul(inv(self.e_W[:, :, i]), self.Px).flatten()
            self.e_TtT[:, :, i] = self.e_sigma_sq_old * inv(self.e_W[:, :, i]) + np.outer(self.e_T[i, :].T,
                                                                                          self.e_T[i, :])

    def _update_params(self, N, V, A):
        self.e_mu = np.zeros((V, 1))
        for j in range(V):
            self.e_mu[j] = 1 / len(self.O_list_col[j]) * np.sum(
                self.e_X[self.O_list_col[j], j] - np.matmul(self.e_T[self.O_list_col[j], :], self.e_P[j, :]))

        for j in range(V):
            self.e_P1 = np.matmul(self.e_T[self.O_list_col[j], :].T,
                                  (self.e_X[self.O_list_col[j], j] - self.e_mu[j])).reshape((A, 1))
            self.e_P2 = np.sum(self.e_TtT[:, :, self.O_list_col[j]], axis=2)
            self.e_P[j, :] = np.matmul(inv(self.e_P2), self.e_P1).reshape((1, A))

        self.e_sigma_sq = 0
        for i in range(N):
            for j in self.O_list_row[i]:
                self.e_sigma_sq = self.e_sigma_sq + 1 / (N * V) * (
                            (self.e_X[i, j] - np.inner(self.e_P[j, :], self.e_T[i, :]) - self.e_mu[j]) ** 2
                            + self.e_sigma_sq_old * np.matmul(self.e_P[j, :],
                                                              np.matmul(inv(self.e_W[:, :, i]), self.e_P[j, :].T)))

        self.e_W = np.zeros((A, A, N))
        for i in range(N):
            self.e_W[:, :, i] = np.matmul(self.e_P[self.O_list_row[i], :].T,
                                          self.e_P[self.O_list_row[i], :]) + self.e_sigma_sq_old * np.eye(A)

        self.e_sigma_sq_old = self.e_sigma_sq

    def lower_bound(self, N, V):
        LB = 0
        for i in range(N):
            LB = LB - (V / 2 * np.log(self.e_sigma_sq) + 1 / 2 * np.trace(
                self.e_TtT[:, :, i]) + 1 / 2 / self.e_sigma_sq * np.trace(
                np.matmul(np.matmul(self.e_P.T, self.e_P), self.e_TtT[:, :, i])))
            LB = LB - (1 / 2 / self.e_sigma_sq * np.inner(
                self.e_X[i, self.O_list_row[i]].T - self.e_mu[self.O_list_row[i]].flatten(),
                self.e_X[i, self.O_list_row[i]].T - self.e_mu[
                    self.O_list_row[i]].flatten()) - 1 / self.e_sigma_sq * np.matmul(
                np.matmul(self.e_T[i, :], self.e_P[self.O_list_row[i], :].T),
                self.e_X[i, self.O_list_row[i]].T - self.e_mu[self.O_list_row[i]].flatten()))

        return LB

    def _update(self, N, V, A):
        self._update_T(N, A)
        self._update_params(N, V, A)
        LB = self.lower_bound(N, V)
        self.lbs.append(LB)
        if self.verbose:
            print("Lower bound: {}".format(LB))

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        A = self.A
        N = self.N
        V = self.V
        self._init_paras(X, N, V, A)

        for it in range(self.n_iters):
            self._update(N, V, A)
            if it >= 1:
                if np.abs((self.lbs[it] - self.lbs[it - 1]) / self.lbs[it - 1]) < self.tol:
                    break

    def recover(self):
        return self.e_T.dot(self.e_P.T)


class ppcamfit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose

    def _init_paras(self, X, N, V, A):
        self.e_X = X
        self.e_T = np.zeros((N, A))
        X0 = copy.deepcopy(X)
        self.X0 = X0
        self.X0_scaled = scale_by(X=X0, mu=np.mean(X0, axis=0), sigma=np.std(X0, axis=0))
        self.e_P, _ = pca_by_svd(X=self.X0_scaled, A=self.A)
        self.e_sigma_sq = np.var(X)
        self.e_mu = np.zeros((V, 1))
        self.e_TtT = np.zeros((A, A, N))
        self.e_XtX = np.zeros((V, V, N))
        self.e_XtT = np.zeros((V, A, N))
        self.e_W = np.zeros((A, A, N))

        self.M_list = []
        self.O_list = []
        for i in range(self.N):
            self.M_list.append([j for j in range(self.V) if self.M[i, j] == True])
            self.O_list.append([j for j in range(self.V) if self.M[i, j] == False])

        for i in range(N):
            self.e_W[:, :, i] = self.e_W[:, :, i] + np.matmul(self.e_P[self.O_list[i], :].T,
                                                              self.e_P[self.O_list[i], :]) + self.e_sigma_sq * np.eye(A)

        self.tol = 1e-6
        self.lbs = []

    def _update_T(self, N, A):
        for i in range(N):
            self.Px = np.sum(np.multiply(self.e_P[self.O_list[i], :].T, (
                        self.e_X[i, self.O_list[i]].reshape(1, len(self.O_list[i])) - self.e_mu[self.O_list[i]].T)),
                             axis=1).reshape((self.A, 1))
            self.e_T[i, :] = np.matmul(inv(self.e_W[:, :, i]), self.Px).flatten()
            self.e_TtT[:, :, i] = self.e_sigma_sq * inv(self.e_W[:, :, i]) + np.outer(self.e_T[i, :].T, self.e_T[i, :])

    def _update_X(self, N, V):
        for i in range(N):
            self.e_X[i, self.M_list[i]] = np.matmul(self.e_T[i, :], self.e_P[self.M_list[i], :].T) + self.e_mu[
                self.M_list[i]].T

        for i in range(N):
            self.e_XtX[:, :, i] = np.multiply(np.outer(self.e_X[i, :], self.e_X[i, :]),
                                              1 - np.outer(self.M[i, :], self.M[i, :])) + \
                                  np.multiply(self.e_sigma_sq * (
                                          np.identity(V) + np.matmul(np.matmul(self.e_P, inv(self.e_W[:, :, i])),
                                                                     self.e_P.T)) + \
                                              np.outer(self.e_X[i, :], self.e_X[i, :]),
                                              np.outer(self.M[i, :], self.M[i, :]))

        for i in range(N):
            self.e_XtT[:, :, i] = np.multiply(self.M[i, :].T.reshape(V, 1),
                                              self.e_sigma_sq * np.matmul(self.e_P, inv(self.e_W[:, :, i]))) + \
                                  np.outer(self.e_X[i, :].T, self.e_T[i, :])

    def _update_params(self, N, V, A):
        self.e_mu = np.zeros((V, 1))
        for i in range(N):
            self.e_mu = self.e_mu + 1 / N * (self.e_X[i, :].T - np.matmul(self.e_P, self.e_T[i, :].T)).reshape((V, 1))

        self.e_P1 = np.zeros((V, A))
        self.e_P2 = np.zeros((A, A))
        for i in range(N):
            self.e_P1 = self.e_P1 + self.e_XtT[:, :, i] - np.outer(self.e_mu, self.e_T[i, :])
            self.e_P2 = self.e_P2 + self.e_TtT[:, :, i]
        self.e_P = np.matmul(self.e_P1, inv(self.e_P2))

        self.e_sigma_sq = 0
        for i in range(N):
            self.e_sigma_sq = self.e_sigma_sq + 1 / (N * V) * (
                np.trace(self.e_XtX[:, :, i] - 2 * np.matmul(self.e_XtT[:, :, i], self.e_P.T)
                         - 2 * np.outer(self.e_mu, self.e_X[i, :]) + 2 * np.outer(self.e_mu,
                                                                                  np.matmul(self.e_T[i, :], self.e_P.T))
                         + np.matmul(self.e_P, np.matmul(self.e_TtT[:, :, i], self.e_P.T)) + np.outer(self.e_mu,
                                                                                                      self.e_mu)))

        self.e_W = np.zeros((A, A, N))
        for i in range(N):
            self.e_W[:, :, i] = np.matmul(self.e_P[self.O_list[i], :].T,
                                          self.e_P[self.O_list[i], :]) + self.e_sigma_sq * np.eye(A)

    def lower_bound(self, N, V):
        LB = 0
        for i in range(N):
            LB = LB - (V / 2 * np.log(self.e_sigma_sq) + 1 / 2 * np.trace(
                self.e_TtT[:, :, i]) + 1 / 2 / self.e_sigma_sq * np.trace(
                np.matmul(np.matmul(self.e_P.T, self.e_P), self.e_TtT[:, :, i])))
            LB = LB - (1 / 2 / self.e_sigma_sq * np.inner(self.e_X[i, self.O_list[i]], self.e_X[
                i, self.O_list[i]]) - 1 / self.e_sigma_sq * np.matmul(
                np.matmul(self.e_T[i, :], self.e_P[self.O_list[i], :].T), self.e_X[i, self.O_list[i]].T))
            LB = LB - (-1 / self.e_sigma_sq * np.trace(
                np.matmul(self.e_P[self.M_list[i], :].T, self.e_XtT[self.M_list[i], :, i])))
            if len(self.M_list[i]) > 1:
                submatrix = self.e_XtX[:, :, i]
                LB = LB - 1 / 2 / self.e_sigma_sq * np.trace(submatrix[np.ix_(self.M_list[i], self.M_list[i])])
            elif len(self.M_list[i]) == 1:
                LB = LB - 1 / 2 / self.e_sigma_sq * self.e_XtX[self.M_list[i], self.M_list[i], i]
        return LB

    def _update(self, N, V, A):
        self._update_T(N, A)
        self._update_X(N, V)
        self._update_params(N, V, A)
        LB = self.lower_bound(N, V)
        self.lbs.append(LB)
        if self.verbose:
            print("Lower bound: {}".format(LB))

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        A = self.A
        N = self.N
        V = self.V
        self._init_paras(X, N, V, A)

        for it in range(self.n_iters):
            self._update(N, V, A)
            if it >= 1:
                if np.abs((self.lbs[it] - self.lbs[it - 1]) / self.lbs[it - 1]) < self.tol:
                    break

    def recover(self):
        return self.e_T.dot(self.e_P.T)


class bpcafit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose

    def _init_paras(self, X, N, V, A):
        self.e_X = X
        self.e_T = np.zeros((N, A))
        X0 = copy.deepcopy(X)
        self.X0 = X0
        self.X0_scaled = scale_by(X=X0, mu=np.mean(X0, axis=0), sigma=np.std(X0, axis=0))
        self.e_P, _ = pca_by_svd(X=self.X0_scaled, A=self.A)
        self.obsnomiss = [i for i in range(N) if sum(self.M[i, :]) == 0]
        self.obsmiss = [i for i in range(N) if sum(self.M[i, :]) > 0]
        self.Xnomiss = self.e_X[self.obsnomiss, :]
        self.Xcov = 1 / (N - 1) * np.matmul(self.e_X.T, self.e_X)
        U, S, _ = svd(self.Xcov)
        self.e_W = np.matmul(U[:, :A], np.sqrt(np.diag(S[:A])))
        self.e_mu = np.zeros(V)
        self.tau = 1 / (np.sum(np.diag(self.Xcov)) - np.sum(S[:A]))
        self.taumax = 1e10
        self.taumin = 1e-10
        self.tau = max(min(self.tau, self.taumax), self.taumin)

        self.galpha0 = 1e-10
        self.balpha0 = 1
        self.alpha = np.divide((2 * self.galpha0 + V),
                               (self.tau * np.diag(np.matmul(self.e_W.T, self.e_W)) + 2 * self.galpha0 / self.balpha0))

        self.gmu0 = 0.001

        self.btau0 = 1
        self.gtau0 = 1e-10
        self.sigW = np.eye(A)

    def _dostep(self, X):
        A = self.A
        N = self.N
        V = self.V

        Rx = np.eye(A) + self.tau * np.matmul(self.e_W.T, self.e_W) + self.sigW
        Rxinv = inv(Rx)
        idx = self.obsnomiss
        n = len(idx)
        dX = X[idx, :] - np.tile(self.e_mu, (n, 1))
        x = self.tau * np.matmul(Rxinv, np.matmul(self.e_W.T, dX.T))
        T = np.matmul(dX.T, x.T)
        trS = np.sum(dX ** 2)

        for n in range(len(self.obsmiss)):
            i = self.obsmiss[n]
            M_list = [j for j in range(self.V) if self.M[i, j] == True]
            O_list = [j for j in range(self.V) if self.M[i, j] == False]
            dXo = self.e_X[i, O_list] - self.e_mu[O_list]
            Wm = self.e_W[M_list, :]
            Wo = self.e_W[O_list, :]
            Rxinv = inv(Rx - self.tau * np.matmul(Wm.T, Wm))
            ex = self.tau * np.matmul(Wo.T, dXo.T)
            x = np.matmul(Rxinv, ex)
            dXm = np.matmul(Wm, x)
            dX = X[i, :]
            dX[O_list] = dXo.T
            dX[M_list] = dXm.T
            self.e_X[i, :] = dX + self.e_mu
            T = T + np.outer(dX, x)
            T[M_list, :] = T[M_list, :] + np.matmul(Wm, Rxinv)
            trS = trS + np.inner(dX, dX) + len(M_list) / self.tau + np.trace(np.matmul(Wm, np.matmul(Rxinv, Wm.T)))

        T = T / N
        trS = trS / N

        Rxinv = inv(Rx)
        Dw = Rxinv + self.tau * np.matmul(T.T, np.matmul(self.e_W, Rxinv)) + np.diag(self.alpha) / N
        Dwinv = inv(Dw)
        self.e_W = np.matmul(T, Dwinv)
        self.tau = (self.V + 2 * self.gtau0 / self.N) / (trS - np.trace(np.matmul(T.T, self.e_W)) + (
                    self.gmu0 * np.inner(self.e_mu, self.e_mu) + 2 * self.gtau0 / self.btau0) / self.N)
        self.SigW = Dwinv * (self.V / self.N)

        self.alpha = np.divide((2 * self.galpha0 + self.V), (
                    self.tau * np.diag(np.matmul(self.e_W.T, self.e_W)) + np.diag(
                self.SigW) + 2 * self.galpha0 / self.balpha0))

    def fit(self, X, M):
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        A = self.A
        N = self.N
        V = self.V
        #         temporarily comment these two lines out
        #         if not D:
        #             D = N
        self._init_paras(X, N, V, A)
        tauold = 1000
        for it in range(self.n_iters):
            self._dostep(X)
            tau = self.tau
            dtau = abs(np.log10(tau) - np.log10(tauold))
            if self.verbose:
                print("dtau: {}".format(dtau))
            if it >= 1:
                if dtau < 1e-6:
                    break
            tauold = tau

    def recover(self):
        return self.e_X


## Matrix completion algorithms
class svtfit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False, tau=0):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose
        self.tau = tau

    def _init_paras(self, tau):
        self.tau = tau
        self.tol = 1e-4
        self.delta = 1.2 * self.N * self.V / (self.N * self.V - np.sum(self.M))

        self.M_list = []
        self.O_list = []
        for i in range(self.N):
            self.M_list.append([j for j in range(self.V) if self.M[i, j] == True])
            self.O_list.append([j for j in range(self.V) if self.M[i, j] == False])

        self.k0 = math.ceil(tau / self.delta / norm(self._operator_PO(self.X)))
        self.Z = self.k0 * self.delta * self._operator_PO(self.X)

    def _operator_Dtau(self, X, A, tau):
        U, S, Vh = svd(X)
        U = U[:, :A]
        Vh = Vh[:A, :]
        S = S[:A]
        for i in range(len(S)):
            if S[i] < tau:
                S[i] = 0
        S = np.diag(S)
        return np.matmul(np.matmul(U, S), Vh)

    def _operator_PO(self, X):
        X_PO = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            X_PO[i, self.O_list[i]] = X[i, self.O_list[i]]
        return X_PO

    def fit(self, X, M):
        self.X = X
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M

        tau = self.tau
        self._init_paras(tau)
        self.r = 0
        for k in range(self.n_iters):
            U, S, Vh = svd(self.Z)
            self.r = np.sum(S > tau)
            U = U[:, :self.r]
            Vh = Vh[:self.r, :]
            S = S[:self.r]
            self.Amat = np.matmul(np.matmul(U, np.diag(S-tau)), Vh)
            dnorm = norm(self._operator_PO(self.Amat - self.X)) / norm(self._operator_PO(self.X))
            if dnorm <= self.tol:
                break
            self.Z = self.Z + self.delta * np.multiply(self.X - self.Amat, 1-self.M)

    def recover(self):
        return self.Amat


class ialmfit:
    ## X: input continuous data with shape (N, V)
    ## A: number of ppca components
    def __init__(self, A=2, n_iters=1000, verbose=False):
        self.A = A
        self.n_iters = n_iters
        self.verbose = verbose

    def _init_paras(self):
        self.D = self.X
        self.D[self.M] = 0
        self.Y_old = np.zeros((self.X.shape[0], self.X.shape[1]))
        self.E_old = np.zeros((self.X.shape[0], self.X.shape[1]))

        self.mu_old = 1 / norm(self.D)
        self.rho = 1.2172 + 1.8588 * (1 - np.sum(self.M) / self.X.shape[0] / self.X.shape[1])

        self.e1 = 1e-7
        self.e2 = 1e-6

    def _operator_S(self, A, epsilon):
        S = np.multiply(A - epsilon, A > epsilon) + np.multiply(A + epsilon, A < -epsilon)
        return S

    def _update_Amat(self):
        U, S, Vh = svd(self.D - self.E_old + 1 / self.mu_old * self.Y_old)
        U = U[:, :len(S)]
        self.Amat = np.matmul(np.matmul(U, self._operator_S(np.diag(S), 1 / self.mu_old)), Vh)

    def _update_EY(self):
        self.E_new = np.multiply(self.M, (self.D - self.Amat + 1 / self.mu_old * self.Y_old))
        self.Y_new = self.Y_old + self.mu_old * (self.D - self.Amat - self.E_new)

    def _update_mu(self):
        if np.minimum(self.mu_old, np.sqrt(self.mu_old)) * norm(self.E_new - self.E_old) / norm(self.D) < self.e2:
            self.mu_new = self.mu_old * self.rho
        else:
            self.mu_new = self.mu_old

    def fit(self, X, M):
        self.X = X
        self.N = X.shape[0]
        self.V = X.shape[1]
        self.M = M
        self._init_paras()

        for it in range(self.n_iters):
            self._update_Amat()
            self._update_EY()
            self._update_mu()
            ratio1 = norm(self.D - self.Amat - self.E_new) / norm(self.D)
            ratio2 = np.minimum(self.mu_new, np.sqrt(self.mu_new)) * norm(self.E_new - self.E_old) / norm(self.D)
            if self.verbose:
                print("ratio1: {}, ratio2: {}".format(ratio1, ratio2))
            if it >= 1:
                if (ratio1 < self.e1) & (ratio2 < self.e2):
                    break
            self.mu_old = self.mu_new
            self.E_old = self.E_new
            self.Y_old = self.Y_new

    def recover(self):
        return self.Amat
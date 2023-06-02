import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DDSimca:

    def __init__(self, ncomps=2, alpha=0.01, gamma=0.01):
        self.n_comps = ncomps
        self.od = None
        self.sd = None
        self.extreme_objs = None
        self.outlier_objs = None
        self.scores = None
        self.loadings = None
        self.dof_od = None
        self.dof_sd = None
        self.alpha = alpha
        self.gamma = gamma
        self.training_set = None
        self.training_set_mean = None
        self.training_set_std = None

    def fit(self, X):
        self.training_set = X
        D, P = self.decomp()

        sd_vector = self.calculate_sd(P[:, 0:self.n_comps], D)
        od_vector = self.calculate_od(P[:, 0:self.n_comps])

        dof_sd, av_sd = self.calculate_dof(sd_vector)
        dof_od, av_od = self.calculate_dof(od_vector)


    def decomp(self):
        X = self.training_set
        _, D, P = np.linalg.svd(X)
        D = np.diag(D)
        D = D[0:self.n_comps, 0:self.n_comps]
        P = np.transpose(P)
        return D, P

    def calculate_sd(self, P, D):
        X = self.training_set
        n = len(X.index)
        T = X @ P
        v_lambda = D.diagonal()
        v_work = [[0 for _ in range(self.n_comps)] for _ in range(n)]
        for i in range(n):
            for j in range(self.n_comps):
                v_work[i][j] = T.iloc[i, j] / v_lambda[j]
        v_sd = [0 for _ in range(n)]
        for i in range(n):
            for j in range(self.n_comps):
                v_sd[i] += v_work[i][j] ** 2
        return v_sd

    def calculate_od(self, P):
        X = self.training_set
        n, p = X.shape
        E = X @ (np.eye(p, p) - (P @ np.transpose(P)))
        v_od = [0 for _ in range(n)]
        for i in range(n):
            for j in range(p):
                v_od[i] += E.iloc[i, j] ** 2
            v_od[i] = v_od[i] / p
        return v_od

    def calculate_dof(self, v):
        av = np.mean(v)
        dof = round(2 * (av/np.std(v)) ** 2)
        return dof, av


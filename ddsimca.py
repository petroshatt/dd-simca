import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats.distributions import chi2


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

        norm_sd_vector = np.divide(sd_vector, av_sd)
        norm_od_vector = np.divide(od_vector, av_od)

        sd_crit, od_crit = self.calulate_border(dof_sd, dof_od)
        extr_vector = self.find_extremes(norm_sd_vector, norm_od_vector, sd_crit, od_crit)


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

    def calulate_border(self, dof_sd, dof_od):
        d_crit = chi2.ppf(1 - self.alpha, dof_sd + dof_od)
        sd_crit = d_crit / dof_sd
        od_crit = d_crit / dof_od
        return sd_crit, od_crit

    def find_extremes(self, norm_sd_vector, norm_od_vector, sd_crit, od_crit):
        n = len(norm_od_vector)
        od_cur = [0 for _ in range(n)]
        extr_vector = [False for _ in range(n)]
        for i in range(n):
            od_cur[i] = od_crit * (1 - norm_sd_vector[i] / sd_crit)
            extr_vector[i] = (norm_sd_vector[i] > sd_crit) | (norm_od_vector[i] > od_cur[i])
        return extr_vector


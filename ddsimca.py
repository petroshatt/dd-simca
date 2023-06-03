import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

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
        self.eigenmatrix = None
        self.od_mean = None
        self.sd_mean = None
        self.dof_od = None
        self.dof_sd = None
        self.alpha = alpha
        self.gamma = gamma
        self.od_crit = None
        self.sd_crit = None
        self.od_out = None
        self.sd_out = None
        self.training_set = None
        self.training_set_mean = None
        self.training_set_std = None

    def fit(self, X):
        self.training_set = X
        n, _ = X.shape

        D, P = self.decomp()

        sd_vector = self.calculate_sd(P[:, 0:self.n_comps], D)
        od_vector = self.calculate_od(P[:, 0:self.n_comps])

        dof_sd, av_sd = self.calculate_dof(sd_vector)
        dof_od, av_od = self.calculate_dof(od_vector)

        norm_sd_vector = np.divide(sd_vector, av_sd)
        norm_od_vector = np.divide(od_vector, av_od)

        sd_crit, od_crit = self.calulate_border(dof_sd, dof_od, self.alpha)
        extr_vector = self.find_extremes(norm_sd_vector, norm_od_vector, sd_crit, od_crit)

        alpha_out = 1 - ((1 - self.gamma) ** (1 / n))
        sd_out, od_out = self.calulate_border(dof_sd, dof_od, alpha_out)
        out_vector = self.find_extremes(norm_sd_vector, norm_od_vector, sd_out, od_out)

        self.loadings = P[:, 0:self.n_comps]
        self.scores = X @ self.loadings
        self.eigenmatrix = D
        self.od_mean = av_od
        self.sd_mean = av_sd
        self.dof_od = dof_od
        self.dof_sd = dof_sd
        self.od_crit = od_crit
        self.sd_crit = sd_crit
        self.od_out = od_out
        self.sd_out = sd_out
        self.od = od_vector
        self.sd = sd_vector
        self.extreme_objs = extr_vector
        self.outlier_objs = out_vector

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
        dof = round(2 * (av / np.std(v)) ** 2)
        return dof, av

    def calulate_border(self, dof_sd, dof_od, error):
        d_crit = chi2.ppf(1 - error, dof_sd + dof_od)
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

    def acceptance_plot(self):
        plt.title("Acceptance Plot")
        plt.xlabel("log(1 + h/h_0)")
        plt.ylabel("log(1 + v/v_0)")

        x, y = self.border_plot(self.sd_crit, self.od_crit)
        plt.plot(x, y, 'g')
        x, y = self.border_plot(self.sd_out, self.od_out)
        plt.plot(x, y, 'r')

        oD = [0 for _ in range(len(self.od))]
        sD = [0 for _ in range(len(self.sd))]

        for i in range(len(self.od)):
            oD[i] = self.transform_(self.od[i] / self.od_mean)
        for i in range(len(self.sd)):
            sD[i] = self.transform_(self.sd[i] / self.sd_mean)

        for i in range(len(self.extreme_objs)):
            if (not self.extreme_objs[i]) and (not self.outlier_objs[i]):
                plt.plot(sD[i], oD[i], "o", color='lime', label='Regular')
            elif not self.outlier_objs[i]:
                plt.plot(sD[i], oD[i], 'ro', label='Extreme')
            elif not self.extreme_objs[i]:
                plt.plot(sD[i], oD[i], 'rs', label='Outlier')
            plt.annotate(str(i+1), (sD[i], oD[i]))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.show()

    def border_plot(self, sd_crit, od_crit):
        x = np.linspace(0, self.transform_(sd_crit), num=100)
        n = len(x)
        y = [0 for _ in range(n)]

        for k in range(n):
            if x[k] > self.transform_(sd_crit) or x[k] < 0:
                y[k] = 0
            else:
                y[k] = od_crit / sd_crit * (sd_crit - self.transform_reverse(x[k]))
                if y[k] < 0:
                    y[k] = 0

        for j in range(len(y)):
            y[j] = self.transform_(y[j])

        return x, y

    def transform_(self, input):
        return math.log(1 + input)

    def transform_reverse(self, input):
        return math.exp(input) - 1

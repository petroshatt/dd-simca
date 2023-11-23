import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure, show
import numpy as np
import math
import functools

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from scipy.stats.distributions import chi2

plots = []


def conjunction(conditions):
    return functools.reduce(np.logical_and, conditions)


def disjunction(conditions):
    return functools.reduce(np.logical_or, conditions)


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
        self.target_class = None
        self.training_set_mean = None
        self.training_set_std = None
        self.test_set = None
        self.test_set_labels = None
        self.test_set_mean = None
        self.test_set_std = None
        self.od_test = None
        self.sd_test = None
        self.external_objs_test = None
        self.conf_matrix = None
        self.metrics_list = None

    def preprocessing(self, X, centering=True, scaling=True):
        self.training_set = X
        if centering:
            X = X.apply(lambda x: x - x.mean())
        if scaling:
            temp = np.std(X)
            X = np.subtract(X, np.mean(X))
            X = np.divide(X, temp)
        return X

    def train_test_split(self, df, filters):
        X_target_cl = df[conjunction(filters)]
        X_target_cl = X_target_cl.iloc[:, 2:]
        X_target_cl.insert(loc=0, column='Class', value=1)
        y_target_cl = X_target_cl['Class']

        X_other_cl = df[~(conjunction(filters))]
        X_other_cl = X_other_cl.iloc[:, 2:]
        X_other_cl.insert(loc=0, column='Class', value=0)
        y_other_cl = X_other_cl['Class']

        X_train, X_test_target_cl, y_train, y_test_target_cl = train_test_split(X_target_cl, y_target_cl, test_size=0.2)
        X_test = pd.concat([X_test_target_cl, X_other_cl])
        y_test = pd.concat([y_test_target_cl, y_other_cl])

        print(X_train)
        print(X_test)

        return X_train, X_test, y_train, y_test

    def kfold_train_test_split(self, df, filters):
        X_target_cl = df[conjunction(filters)]
        X_target_cl = X_target_cl.iloc[:, 2:]
        X_target_cl.insert(loc=0, column='Class', value=1)
        y_target_cl = X_target_cl['Class']

        X_other_cl = df[~(conjunction(filters))]
        X_other_cl = X_other_cl.iloc[:, 2:]
        X_other_cl.insert(loc=0, column='Class', value=0)
        y_other_cl = X_other_cl['Class']

        dfs = []

        kf = KFold(n_splits=5, shuffle=True)
        for X_train_indices, X_test_target_cl_indices in kf.split(X_target_cl):
            X_train = X_target_cl.iloc[X_train_indices, :]
            y_train = X_train['Class']
            X_test_target_cl = X_target_cl.iloc[X_test_target_cl_indices, :]
            y_test_target_cl = X_test_target_cl['Class']

            X_test = pd.concat([X_test_target_cl, X_other_cl])
            y_test = pd.concat([y_test_target_cl, y_other_cl])

            dfs.append([X_train, y_train, X_test, y_test])
        return dfs


    def export_csvs(self, X_train, X_test, y_train, y_test):
        X_train.to_csv('X_train.csv')
        X_test.to_csv('X_test.csv')
        y_train.to_csv('y_train.csv')
        y_test.to_csv('y_test.csv')

        X_train.reset_index(drop=False, inplace=True)
        X_train_names = X_train['Sample']
        X_test.reset_index(drop=False, inplace=True)
        X_test_names = X_test['Sample']
        X_train_names.to_csv('X_train_names.csv')
        X_test_names.to_csv('X_test_names.csv')

    def fit(self, X):
        self.training_set = X.iloc[:, 1:]
        self.target_class = X.iloc[0, 0]
        n, _ = self.training_set.shape

        D, P = self.decomp()

        sd_vector = self.calculate_sd(self.training_set, P[:, 0:self.n_comps], D)
        od_vector = self.calculate_od(self.training_set, P[:, 0:self.n_comps])

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
        self.scores = self.training_set @ self.loadings
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

    def calculate_sd(self, x, P, D):
        X = x
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

    def calculate_od(self, x, P):
        X = x
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
        oD = [0 for _ in range(len(self.od))]
        sD = [0 for _ in range(len(self.sd))]

        for i in range(len(self.od)):
            oD[i] = self.transform_(self.od[i] / self.od_mean)
        for i in range(len(self.sd)):
            sD[i] = self.transform_(self.sd[i] / self.sd_mean)

        self.training_set.reset_index(drop=False, inplace=True)
        training_set_names = list(self.training_set['Sample'])
        self.training_set.set_index('Sample', inplace=True)

        point_type_list = [None for _ in range(len(self.extreme_objs))]
        color_list = [None for _ in range(len(self.extreme_objs))]
        for i in range(len(self.extreme_objs)):
            if (not self.extreme_objs[i]) and (not self.outlier_objs[i]):
                point_type_list[i] = 'Regular'
                color_list[i] = 'lime'
            elif not self.outlier_objs[i]:
                point_type_list[i] = 'Extreme'
                color_list[i] = 'red'
            elif not self.extreme_objs[i]:
                point_type_list[i] = 'Outlier'
                color_list[i] = 'red'

        plot_df = pd.DataFrame({'Sample': training_set_names, 'sD': sD, 'oD': oD,
                                'Type': point_type_list, 'Color': color_list})
        source = ColumnDataSource(plot_df)
        hover = HoverTool(
            tooltips=[
                ('Sample: ', '@Sample')
            ]
        )

        p = figure(title="Acceptance Plot - Training Set", width=600, height=600)
        p.add_tools(hover)
        p.xaxis.axis_label = "log(1 + h/h_0)"
        p.yaxis.axis_label = "log(1 + v/v_0)"

        x, y = self.border_plot(self.sd_crit, self.od_crit)
        p.line(x, y, line_width=2, color='green')
        x, y = self.border_plot(self.sd_out, self.od_out)
        p.line(x, y, line_width=2, color='red')

        p.scatter('sD', 'oD', size=8, source=source, color='Color', alpha=0.5)
        return p

    def extreme_plot(self):
        oD = [0 for _ in range(len(self.od))]
        sD = [0 for _ in range(len(self.sd))]
        for i in range(len(self.od)):
            oD[i] = self.od[i] / self.od_mean
        for i in range(len(self.sd)):
            sD[i] = self.sd[i] / self.sd_mean

        Nh = self.dof_sd
        Nv = self.dof_od

        c = list(np.array([item * Nh for item in sD]) + np.array([item * Nv for item in oD]))
        c.sort()
        Nc = Nh + Nv
        I = len(c)

        n = list(range(1, I + 1))
        alpha = np.divide(n, I)

        '''
        TO-DO
        '''

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

    def predict(self, Xtest):
        self.test_set = Xtest.iloc[:, 1:]
        self.test_set_labels = Xtest.iloc[:, 0]
        sd_vector_pred = self.calculate_sd(self.test_set, self.loadings, self.eigenmatrix)
        od_vector_pred = self.calculate_od(self.test_set, self.loadings)

        norm_sd_vector = np.divide(sd_vector_pred, self.sd_mean)
        norm_od_vector = np.divide(od_vector_pred, self.od_mean)

        extr_vector_pred = self.find_extremes(norm_sd_vector, norm_od_vector, self.sd_crit, self.od_crit)

        self.sd_test = sd_vector_pred
        self.od_test = od_vector_pred
        self.external_objs_test = extr_vector_pred

    def pred_acceptance_plot(self):
        oD = [0 for _ in range(len(self.od_test))]
        sD = [0 for _ in range(len(self.sd_test))]

        for i in range(len(self.od_test)):
            oD[i] = self.transform_(self.od_test[i] / self.od_mean)
        for i in range(len(self.sd_test)):
            sD[i] = self.transform_(self.sd_test[i] / self.sd_mean)

        self.test_set.reset_index(drop=False, inplace=True)
        test_set_names = list(self.test_set['Sample'])
        self.test_set.set_index('Sample', inplace=True)

        point_type_list = [None for _ in range(len(self.external_objs_test))]
        color_list = [None for _ in range(len(self.external_objs_test))]
        for i in range(len(self.external_objs_test)):
            if not self.external_objs_test[i]:
                point_type_list[i] = 'Regular'
                color_list[i] = 'lime'
            else:
                point_type_list[i] = 'Extreme'
                color_list[i] = 'red'

        plot_df = pd.DataFrame({'Sample': test_set_names, 'sD': sD, 'oD': oD,
                                'Type': point_type_list, 'Color': color_list})
        source = ColumnDataSource(plot_df)
        hover = HoverTool(
            tooltips=[
                ('Sample: ', '@Sample')
            ]
        )

        p = figure(title="Acceptance Plot - Test Set", width=600, height=600)
        p.add_tools(hover)
        p.xaxis.axis_label = "log(1 + h/h_0)"
        p.yaxis.axis_label = "log(1 + v/v_0)"

        x, y = self.border_plot(self.sd_crit, self.od_crit)
        p.line(x, y, line_width=2, color='green')

        p.scatter('sD', 'oD', size=8, source=source, color='Color', alpha=0.5)
        return p

    def confusion_matrix(self, plot_cm='off', print_metrics='on'):
        cm_pred = list(~np.array(self.external_objs_test))
        cm_actual = []
        for i in range(self.test_set_labels.shape[0]):
            cm_actual.append(self.target_class == self.test_set_labels.iloc[i])

        self.conf_matrix = metrics.confusion_matrix(cm_actual, cm_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=self.conf_matrix, display_labels=[False, True])

        if plot_cm == 'on:':
            cm_display.plot()
            plt.show()

        self.metrics_list = []
        self.metrics_list.append(metrics.accuracy_score(cm_actual, cm_pred))
        self.metrics_list.append(metrics.precision_score(cm_actual, cm_pred))
        self.metrics_list.append(metrics.recall_score(cm_actual, cm_pred))
        self.metrics_list.append(metrics.recall_score(cm_actual, cm_pred, pos_label=0))
        self.metrics_list.append(metrics.f1_score(cm_actual, cm_pred))

        if print_metrics == 'on':
            print("Accuracy:", self.metrics_list[0], "\nPrecision:", self.metrics_list[1], "\nSensitivity Recall:",
                  self.metrics_list[2],
                  "\nSpecificity:", self.metrics_list[3], "\nF1_score:", self.metrics_list[4])
        return self.metrics_list

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from ddsimca import DDSimca

if __name__ == '__main__':

    X = pd.read_csv("../matlab_dataset.csv")

    ddsimca = DDSimca()
    X = ddsimca.preprocessing(X, centering=True, scaling=False)
    ddsimca.fit(X)
    ddsimca.acceptance_plot()


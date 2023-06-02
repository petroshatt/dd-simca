from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from ddsimca import DDSimca

if __name__ == '__main__':
    # X = [[2, 0, 2],
    #      [0, 1, 0],
    #      [0, 0, 0]]

    X = pd.read_csv("../matlab_dataset.csv")

    ddsimca = DDSimca()
    ddsimca.fit(X)



import pandas as pd


from ddsimca import DDSimca


if __name__ == '__main__':

    df = pd.read_csv("data/honey.csv")

    df = df.drop(columns=['Geographical'])
    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    ddsimca = DDSimca(ncomps=5, alpha=0.07, gamma=0.07)
    X_train, X_test, y_train, y_test = ddsimca.train_test_split(df, class_name='Botanical', target_class=2)
    # X = ddsimca.preprocessing(X, centering=False, scaling=False)
    ddsimca.fit(X_train)
    ddsimca.acceptance_plot()
    ddsimca.predict(X_test)
    ddsimca.pred_acceptance_plot()
    ddsimca.confusion_matrix()

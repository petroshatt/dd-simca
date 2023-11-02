import pandas as pd


from ddsimca import DDSimca


if __name__ == '__main__':

    df = pd.read_csv("data/honey.csv")

    df = df.drop(columns=['Geographical'])
    df.set_index('Sample', inplace=True)

    ddsimca = DDSimca(ncomps=5, alpha=0.1, gamma=0.1)
    X_train, X_test = ddsimca.train_test_split(df, class_name='Botanical', target_class=1)
    # X = ddsimca.preprocessing(X, centering=False, scaling=False)
    ddsimca.fit(X_train)
    ddsimca.acceptance_plot()
    ddsimca.predict(X_test)
    ddsimca.pred_acceptance_plot()
    ddsimca.confusion_matrix()

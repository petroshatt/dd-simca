import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import show


from ddsimca import DDSimca


if __name__ == '__main__':

    plots = []
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    df = pd.read_csv("data/honey.csv")

    df = df.drop(columns=['Geographical'])
    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    ddsimca = DDSimca(ncomps=3, alpha=0.15, gamma=0.15)
    for i in range(10):
        print(f"ITERATION {i+1}")
        X_train, X_test, y_train, y_test = ddsimca.train_test_split(df, class_name='Botanical', target_class=1)
        # X_train = ddsimca.preprocessing(X_train, centering=True, scaling=False)
        ddsimca.fit(X_train)
        ddsimca.predict(X_test)
        metrics.loc[len(metrics)] = ddsimca.confusion_matrix(print_metrics='off')

        # plots.append(ddsimca.acceptance_plot())
        # plots.append(ddsimca.pred_acceptance_plot())
        # show(row(plots))
    print(metrics)
    print(metrics.mean())


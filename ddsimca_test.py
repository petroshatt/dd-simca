import pandas as pd
import numpy as np
from bokeh.layouts import row
from bokeh.plotting import show


from ddsimca import DDSimca


if __name__ == '__main__':

    plots = []
    metrics = pd.DataFrame(columns=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_score'])

    df = pd.read_csv("data/honey.csv")

    df = df.drop(df.loc[:, '499.96':'748.25'].columns, axis=1)
    df = df.drop(df.loc[:, '1802.15':'4000.12'].columns, axis=1)
    df.set_index('Sample', inplace=True)

    ddsimca = DDSimca(ncomps=5, alpha=0.05, gamma=0.05)
    filters = [(df['Botanical'] == 1), (df['Geographical'] == 2)]
    for i in range(1):
        print(f"ITERATION {i+1}")
        X_train, X_test, y_train, y_test = ddsimca.train_test_split(df, filters)
        ddsimca.fit(X_train)
        ddsimca.predict(X_test)
        metrics.loc[len(metrics)] = ddsimca.confusion_matrix(print_metrics='off')

        plots.append(ddsimca.acceptance_plot())
        plots.append(ddsimca.pred_acceptance_plot())
        show(row(plots))
    print(metrics)
    print(metrics.mean())


from parsers import filepath_parser, filter_nan
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


def scatter_plot(args):
    df = pd.read_csv(args.filepath)
    numeric_columns = df.select_dtypes(include='number').columns
    df = df[numeric_columns]
    df = df.drop(['Index'], axis=1)
    columns = df.columns
    results = []
    for i in range(0, len(columns)):
        for j in range(i+1, len(columns)):
            vals = list(zip(df[columns[i]].to_numpy(),
                        df[columns[j]].to_numpy()))
            vals = list(filter(filter_nan, vals))
            val1, val2 = list(zip(*vals))
            val1, val2 = np.array(val1), np.array(val2)
            koeff_pirs = (np.mean(val1 * val2) - np.mean(val1) *
                          np.mean(val2)) / (np.std(val1) * np.std(val2))
            if abs(koeff_pirs) > 0.9:
                results.append((str(columns[i]), str(columns[j])))

    df = pd.read_csv(args.filepath)
    for col1, col2 in results:
        sns.scatterplot(data=df, x=col1, y=col2, hue='Hogwarts House')
        plt.savefig('scatter_plot_' + str(col1)
                    [0:5] + '_' + str(col2)[0:5] + '.png')
        plt.show()


if __name__ == '__main__':
    scatter_plot(filepath_parser().parse_args())

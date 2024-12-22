from parsers import filepath_parser
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


def pair_plot(args):
    df = pd.read_csv(args.filepath)
    df = df.drop(['Index'], axis=1)
    numeric_columns = df.select_dtypes(include='number').columns
    sns.pairplot(data=df, vars=numeric_columns,
                 corner=True, hue='Hogwarts House')
    plt.savefig('pair_plot.png')
    plt.show()


if __name__ == '__main__':
    pair_plot(filepath_parser().parse_args())

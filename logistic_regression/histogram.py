from parsers import filepath_parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Qt5Agg')
# курс это дисциплина


def histogram(args):
    df = pd.read_csv(args.filepath)
    df = df.drop(['Index', 'First Name', 'Last Name',
                 'Best Hand', 'Birthday'], axis=1)

    disciplines = df.columns.to_numpy()
    disciplines = disciplines[disciplines != 'Hogwarts House']

    plt.figure(figsize=[13, 7])
    plt.suptitle(
        'Distribution of scores between faculties in different courses')
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    ax = plt.gca()

    for i, discipline in enumerate(disciplines):
        plt.subplot(3, int(len(disciplines)/3)+1, i+1)
        sns.histplot(data=df, x=str(discipline), hue="Hogwarts House")
        plt.title(str(discipline))
        ax = plt.gca()
        ax.get_legend().remove()
        ax.ticklabel_format(style='scientific', axis='y', scilimits=[-5, 5])
        plt.xlabel('')
        plt.ylabel('')
    plt.savefig('histogram.png')
    plt.show()


if __name__ == '__main__':
    histogram(filepath_parser().parse_args())

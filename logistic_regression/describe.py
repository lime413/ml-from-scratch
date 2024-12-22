import pandas as pd
import numpy as np
import math
from parsers import describe_parser


def describe(args):
    df = pd.read_csv(args.filepath)
    numeric_columns = df.select_dtypes(include='number').columns
    df = df[numeric_columns]
    print('File: ', end='')
    print(args.filepath)

    percentiles = np.array(args.percentiles)
    incorrect_indexes = np.where((percentiles > 100) | (percentiles < 0))
    if len(incorrect_indexes[0]) != 0:
        print('Incorrect percentile(s) was deleted')
        percentiles = np.delete(percentiles, incorrect_indexes)
    print('Percentiles: ', end='')
    print(percentiles)

    description = np.zeros((len(numeric_columns), 5 + len(percentiles)))
    val_length = 0

    for k, column in enumerate(df.columns):
        f_arr = np.sort(df[column].to_numpy())
        f_arr = np.delete(f_arr, np.where(np.isnan(f_arr)))
        arr = np.zeros(5 + len(percentiles))
        for f_val in f_arr:
            arr[0] += 1             # count
            arr[1] += f_val         # summ
            if f_val != 0:
                if math.ceil(math.log10(abs(f_val))) > val_length:
                    val_length = math.ceil(math.log10(abs(f_val)))
        arr[1] /= arr[0]
        arr[2] = math.sqrt(np.sum(np.power(f_arr - arr[1], 2)) / (arr[0]-1))
        arr[3] = f_arr[0]
        arr[-1] = f_arr[-1]
        for i, per in enumerate(args.percentiles):
            arr[4+i] = f_arr[math.ceil(per/100*arr[0])-1]
        description[k] = arr

    titles = ['count', 'mean', 'std', 'min']
    for per in percentiles:
        titles.append(str(per) + '%')
    titles.append('max')

    if args.output_format == 'truncated_titles':
        val_length += args.precision + 1
        # print(val_length)
        for i in range(0, val_length):
            print(' ', end='')
        for k, column in enumerate(df.columns):
            print(str(column)[:5].ljust(val_length+1), end='')
        print('')
        description = description.T
        for i, row in enumerate(description):
            print(titles[i].ljust(val_length), end='')
            for val in description[i]:
                print(format(val, '.'+str(args.precision) +
                      'f').ljust(val_length), end=' ')
            print('')
        for k, column in enumerate(df.columns):
            print(str(column)[:5] + ' - ' + str(column))
    elif args.output_format == 'dataframe':
        description = description.T
        df = pd.DataFrame(description, titles, columns=numeric_columns)
        print(df)
    else:
        print('The entered output format is not supported')


if __name__ == "__main__":
    describe(describe_parser().parse_args())

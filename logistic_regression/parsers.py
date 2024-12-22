import argparse
import numpy as np


def describe_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', type=str, default='data/logistic_regression_train.csv',
                        help='filepath for description')
    parser.add_argument('-percentiles', type=int, default=[
                        25, 50, 75], nargs='+', help='percentiles to include in the output')
    parser.add_argument('-precision', type=int,
                        default=2, help='for the output')
    parser.add_argument('-output_format', type=str, default='dataframe',
                        help='possible options: dataframe, truncated_titles')
    return parser


def filepath_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', type=str, default='data/logistic_regression_train.csv',
                        help='filepath for description')
    return parser


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', type=str, default='data/logistic_regression_train.csv',
                        help='path to csv-file with data for train')
    parser.add_argument('-learning_rate', type=str, default=0.0002,
                        help='learning rate for gradient descent')
    parser.add_argument('-precision', type=str, default=0.01,
                        help='required precision for gradient descent')
    return parser


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', type=str, default='data/logistic_regression_test.csv',
                        help='path to csv-file with data for test')
    parser.add_argument('-weights_path', type=str, default='data/weights.pickle',
                        help='path to pickle-file with weights for test')
    return parser


def filter_nan(arr):
    if np.isnan(arr).any():
        return False
    return True

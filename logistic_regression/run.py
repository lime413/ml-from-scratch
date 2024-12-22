from parsers import describe_parser, train_parser, filepath_parser, test_parser

from describe import describe
from histogram import histogram
from scatter_plot import scatter_plot
from pair_plot import pair_plot
from lg_train import train_net
from lg_predict import test_on_train_data, generate_test_csv

args = describe_parser().parse_args()
describe(args)

args = filepath_parser().parse_args()
histogram(args)
scatter_plot(args)
pair_plot(args)

args = train_parser().parse_args()
train_net(args)

args = test_parser().parse_args()
test_on_train_data(args)
generate_test_csv(args)

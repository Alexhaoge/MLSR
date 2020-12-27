from MLSR.data import DataSet
from MLSR.primary import *
import argparse as arg


def get_arguments():
    parser = arg.ArgumentParser()
    # parser.add_argument('-i', '--iteration', type=int, default=10)
    # parser.add_argument('-c', '--cuda', default=0)
    # parser.add_argument('--no-seed', action='store_true', dest='no_seed')
    parser.add_argument('--dt', action='store_true', help='Train decision tree')
    parser.add_argument('--rf', action='store_true', help='Train random forest')
    parser.add_argument('--nb', action='store_true', help='Train naive bayes')
    parser.add_argument('--svm', action='store_true', help='Train svm')
    parser.add_argument('--lr', action='store_true', help='Train logistic regression')
    parser.add_argument('--xgb', action='store_true', help='Train xgboost')
    return parser.parse_args()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = get_arguments()
    x = DataSet('data/rand_select_400_avg.csv')
    x.generate_feature()
    y = DataSet('data/not_selected_avg.csv')
    y.generate_feature()
    fake = DataSet.data_augment()
    z = DataSet.static_merge(x, y)
    zz = DataSet.static_merge(z, fake)
    if args.dt:
        do_decision_tree(zz, 'log/dTree')
    if args.rf:
        do_random_forest(zz, 'log/rf')
    if args.nb:
        do_naive_bayes(zz, 'log/nb')
    if args.svm:
        do_svm(zz, 'log/svm')
    if args.lr:
        do_logistic(zz, 'log/lr')
    if args.xgb:
        do_xgb(zz, 'log/xgb')
    hard, soso = zz.split_by_weak_label()
    hard.strong_label = hard.strong_label.map({0: 0, 1: 1})
    soso.strong_label = soso.strong_label.map({2: 0, 3: 1})
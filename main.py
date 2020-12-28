from MLSR.data import DataSet
from MLSR.primary import *
from MLSR.ssl import *
import logging
import os
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
    parser.add_argument('--tsvm', action='store_true', help='Train tsvm')
    return parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    import warnings
    warnings.filterwarnings("ignore")
    args = get_arguments()
    # load data
    x = DataSet('data/rand_select_400_avg.csv')
    x.generate_feature()
    y = DataSet('data/not_selected_avg.csv')
    y.generate_feature()
    fake = DataSet.data_augment()
    z = DataSet.static_merge(x, y)
    zz = DataSet.static_merge(z, fake)
    if args.dt:
        if not os.path.exists('log/dTree'):
            os.makedirs('log/dTree')
        do_decision_tree(zz, 'log/dTree')
        logger.log('Decision Tree training complete, output in log/dTree')
    if args.rf:
        if not os.path.exists('log/rf'):
            os.makedirs('log/rf')
        do_random_forest(zz, 'log/rf')
        logger.log('Random Forest complete, output in log/rf')
    if args.nb:
        if not os.path.exists('log/nb'):
            os.makedirs('log/nb')
        do_naive_bayes(zz, 'log/nb')
        logger.log('Naive Bayes training complete, output in log/nb')
    if args.svm:
        if not os.path.exists('log/svm'):
            os.makedirs('log/svm')
        do_svm(zz, 'log/svm')
        logger.log('SVM training complete, output in log/svm')
    if args.lr:
        if not os.path.exists('log/lr'):
            os.makedirs('log/lr')
        do_logistic(zz, 'log/lr')
        logger.log('Logistic Regression training complete, output in log/lr')
    if args.xgb:
        if not os.path.exists('log/xgb'):
            os.makedirs('log/xgb')
        do_xgb(zz, 'log/xgb')
        logger.log('XGBoost training complete, output in log/xgb')
    # create ssl dataset
    hard, soso = z.split_by_weak_label()
    hard.strong_label = hard.strong_label.map({-1: -1, 0: 0, 1: 1})
    soso.strong_label = soso.strong_label.map({-1: -1, 2: 0, 3: 1})
    if args.tsvm:
        if not os.path.exists('log/tsvm'):
            os.makedirs('log/tsvm')
        if not os.path.exists('log/tsvm/hard'):
            os.makedirs('log/tsvm/hard')
        if not os.path.exists('log/tsvm/soso'):
            os.makedirs('log/tsvm/soso')
        do_tsvm(hard, 'log/tsvm/hard')
        do_tsvm(soso, 'log/tsvm/soso')
        logger.log('Transductive SVM training complete, output in log/tvsm')
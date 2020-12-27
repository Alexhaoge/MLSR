from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import numpy as np
import pandas as pd
from time import strftime, localtime
from .data import DataSet
from .plot import plot_confusion_matrix


def grid_search_and_result_ssl(
        Xtrain: pd.DataFrame,
        ytrain: pd.Series,
        Xtest: pd.DataFrame,
        ytest: pd.Series,
        pipe: Pipeline,
        grid: dict,
        log_dir: str,
        score=None,
        verbose: int = 2,
        k: int = 5):
    """
    交叉验证网格搜索，测试集和训练集得分，混淆矩阵和ROC曲线绘制
    Args:
        Xtrain: 训练集特征
        ytrain: 训练集标签
        Xtest: 测试集特征
        ytest: 测试集标签
        pipe: 模型管道
        grid: 超参数搜索空间
        log_dir: 训练结果输出目录，注意一定要先创建该目录
        score: 评分指标，默认使用f1和acc，最后用f1 refit
        verbose: 日志级别，0为静默
        k: 交叉验证折数

    Returns: 训练好的GridSearchCV模型

    """
    file_prefix = log_dir + '/' + strftime("%Y_%m_%d_%H_%M_%S", localtime())
    file = open(file_prefix + '.log.txt', 'x')
    scoring = score
    if scoring is None:
        scoring = {
            'f1': 'f1_macro',
            'accuracy': 'accuracy'
        }
    gsCV = GridSearchCV(
        estimator=pipe,
        cv=k, n_jobs=-1, param_grid=grid,
        scoring=scoring,
        refit='f1',
        verbose=verbose
    )
    gsCV.fit(Xtrain, ytrain)
    dump(gsCV, log_dir + '/gsCV')
    dump(gsCV.best_estimator_, log_dir + '/best_model')
    if verbose > 2:
        file.write(gsCV.cv_results_.__str__())
    if verbose:
        file.write(gsCV.get_params().__str__())
        file.write('\nBest score on training set by grid search cross validation: {}\n'
                   .format(gsCV.score(Xtrain, ytrain)))
    best_model = load(log_dir + '/best_model')
    test_prediction = best_model.predict(Xtest)
    file.write('Accuracy on test set: {}\n'.format(accuracy_score(ytest, test_prediction)))
    file.write('F1-score on test set: {}\n'.format(f1_score(ytest, test_prediction, average='macro')))
    if verbose:
        cm = confusion_matrix(ytrain, best_model.predict(Xtrain))
        plot_confusion_matrix(cm, ['1级', '2级困难', '3级', '4级'], file_prefix + '_train_cm.png')
        file.write('\ntrain_cm:\n')
        file.write(cm.__str__())
        cm = confusion_matrix(ytest, test_prediction)
        plot_confusion_matrix(cm, ['1级', '2级困难', '3级', '4级'], file_prefix + '_test_cm.png')
        file.write('\ntest_cm:\n')
        file.write(cm.__str__())
    #         plot_roc(best_model, Xtest, ytest, file_prefix + '_roc.png')
    file.close()
    return gsCV


def do_naive_bayes(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    TSVM
    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from .tsvm import TSVM
    if grid is None:
        grid = {
            'tsvm__'
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('tsvm', TSVM())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result_ssl(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)
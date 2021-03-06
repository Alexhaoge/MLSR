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


@DeprecationWarning
def lower_bound(cv_results: dict):
    """
    Calculate the lower bound within 1 standard deviation
    of the best `mean_test_scores`.
    Author: Wenhao Zhang <wenhaoz@ucla.edu>

    Args:
        cv_results: dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`

    Returns: float
        Lower bound within 1 standard deviation of the
        best `mean_test_score`.

    """

    best_score_idx = np.argmax(cv_results['mean_test_score'])

    return (cv_results['mean_test_score'][best_score_idx]
            - cv_results['std_test_score'][best_score_idx])


@DeprecationWarning
def best_low_complexity(cv_results: dict):
    """
    Balance model complexity with cross-validated score.
    Author: Wenhao Zhang <wenhaoz@ucla.edu>

    Args:
        cv_results: dict of numpy(masked) ndarrays
        See attribute cv_results_ of `GridSearchCV`.


    Returns:int
        Index of a model that has the fewest PCA components
        while has its test score within 1 standard deviation of the best
        `mean_test_score`.

    """
    threshold = lower_bound(cv_results)
    candidate_idx = np.flatnonzero(cv_results['mean_test_score'] >= threshold)
    best_idx = candidate_idx[
        cv_results['param_reduce_dim__n_components'][candidate_idx].argmin()
    ]
    return best_idx


def grid_search_and_result(
        Xtrain: pd.DataFrame,
        ytrain: pd.Series,
        Xtest: pd.DataFrame,
        ytest: pd.Series,
        pipe: Pipeline,
        grid: dict,
        log_dir: str,
        score=None,
        verbose: int = 2,
        k: int = 5,
        fit_params: dict = None):
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
        fit_params: 训练时参数

    Returns: 训练好的GridSearchCV模型

    """
    scoring = score
    if scoring is None:
        scoring = {
            'f1': 'f1_macro',
            'accuracy': 'accuracy'
        }
    gsCV = GridSearchCV(
        estimator=pipe,
        cv=k, n_jobs=-1,
        param_grid=grid,
        scoring=scoring,
        refit='f1',
        verbose=verbose
    )
    if fit_params is None:
        gsCV.fit(Xtrain, ytrain)
    else:
        gsCV.fit(Xtrain, ytrain, **fit_params)
    dump(gsCV, log_dir + '/gsCV')
    dump(gsCV.best_estimator_, log_dir + '/best_model')
    file_prefix = log_dir + '/' + strftime("%Y_%m_%d_%H_%M_%S", localtime())
    file = open(file_prefix + '.log.txt', 'x')
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
        plot_confusion_matrix(cm, ['特别困难', '一般困难', '不困难'], file_prefix + '_train_cm.png')
        file.write('\ntrain_cm:\n')
        file.write(cm.__str__())
        cm = confusion_matrix(ytest, test_prediction)
        plot_confusion_matrix(cm, ['特别困难', '一般困难', '不困难'], file_prefix + '_test_cm.png')
        file.write('\ntest_cm:\n')
        file.write(cm.__str__())
    #         plot_roc(best_model, Xtest, ytest, file_prefix + '_roc.png')
    file.close()
    return gsCV


def do_decision_tree(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    训练决策树

    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from sklearn.tree import DecisionTreeClassifier
    if grid is None:
        grid = {
            'dt__criterion': ['gini', 'entropy'],
            'dt__max_features': ['auto', 'sqrt', 'log2'],
            'dt__class_weight': [None, 'balanced'],
            'dt__ccp_alpha': [0.0, 0.1],
            'dt__min_impurity_decrease': [0., 0.01],
            'dt__min_samples_leaf': [1, 5],
            'dt__min_samples_split': [2, 8],
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('dt', DecisionTreeClassifier())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_random_forest(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    训练随机森林

    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from sklearn.ensemble import RandomForestClassifier
    if grid is None:
        # raw grid
        # grid = {
        #     'rf__criterion': ['gini', 'entropy'],
        #     'rf__n_estimators': [100, 300, 600, 800, 1200],
        #     'rf__min_samples_split': [2, 5],  # 这里数字是随机给的，无根据
        #     'rf__min_samples_leaf': [1, 4],  # 这里数字是随机给的，无根据
        #     'rf__bootstrap': [True, False],
        #     'rf__min_impurity_decrease': [0., 0.01, 0.1],
        #     'rf__class_weight': ['balanced', 'balanced_subsample', None],
        #     'rf__warm_start': [True, False],
        #     'rf__oob_score': [True, False],
        #     'rf__ccp_alpha': [0., 0.1, 0.5]
        # }
        # fine grid
        grid = {
            'rf__criterion': ['gini', 'entropy'],
            'rf__n_estimators': [80, 100, 150, 200, 500],
            'rf__min_samples_split': [1, 2],  # 这里数字是随机给的，无根据
            'rf__min_samples_leaf': [1, 4],  # 这里数字是随机给的，无根据
            'rf__min_impurity_decrease': [0., 0.01, 0.1],
            'rf__warm_start': [True, False],
            'rf__oob_score': [True, False],
            'rf__ccp_alpha': [0., 0.1, 0.001]
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestClassifier(max_depth=None, n_jobs=-1))
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_svm(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    训练支持向量机

    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from sklearn.svm import SVC
    if grid is None:
        # rough grid
        # grid = {
        #     'SVM__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        #     'SVM__C': [0.01, 0.1, 0.5, 1, 5, 10, 100],
        #     'SVM__gamma': [0.0001, 0.001, 0.01, 'scale', 'auto'],
        #     'SVM__degree': [3, 5],
        #     'SVM__decision_function_shape': ['ovo', 'ovr'],
        #     'SVM__class_weight': [None, 'balanced'],
        #     'SVM__max_iter': [-1, 300],
        #     'SVM__break_ties': [True, False],
        #     'SVM__shrinking': [True, False]
        # }
        # fine grid
        grid = {
            'SVM__kernel': ['linear', 'rbf', 'poly'],
            'SVM__C': [0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.5, 2],
            'SVM__degree': [2, 3, 4],
            'SVM__gamma': [0.001, 'scale'],
            'SVM__decision_function_shape': ['ovo', 'ovr'],
            'SVM__break_ties': [True, False],
            'SVM__tol': [1e-2, 1e-3, 1e-4, 1e-5]
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('SVM', SVC(cache_size=500))
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_logistic(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    训练逻辑回归

    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from sklearn.linear_model import LogisticRegression
    if grid is None:
        grid = {
            'Logistic__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'Logistic__C': [0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 100, 1000],  # 这里数字是随机给的，无根据
            'Logistic__solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
            'Logistic__fit_intercept': [True, False],
            'Logistic__dual': [True, False],
            'Logistic__l1_ratio': [True, False],
            'Logistic__warm_start': [True, False],
            'Logistic__intercept_scaling': [0.01, 0.1, 0.5, 1, 2, 5, 10]
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('Logistic', LogisticRegression(n_jobs=-1, max_iter=500))
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_naive_bayes(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    训练朴素贝叶斯

    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from sklearn.naive_bayes import GaussianNB
    if grid is None:
        grid = {
            'NB__var_smoothing': [1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2, 1],
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('NB', GaussianNB())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_xgb(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """
    训练Xgboost

    Args:
        grid:超参数搜索空间的网格，不填则使用默认搜索空间
        dataset:输入数据集，将会按照0.7, 0.3比例分为训练集和测试集
        log_dir:输出结果文件的目录

    Returns:返回训练好的GridSearchCV模型

    """
    from xgboost import XGBClassifier
    if grid is None:
        grid = {
            'xgb__n_estimators': [80, 100, 150, 200, 400, 500, 600, 800],
            'xgb__max_depth': [6, 8, 10, 15, 20],
            'xgb__colsample_bytree': [0.8, 1],
            'xgb__learning_rate': [0.01, 0.1, 0.3],
            # 'xgb__n_estimators': [1]
        }
    train_param = {
        'xgb__early_stopping_rounds': 100
    }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('xgb', XGBClassifier(
                objective='multi：softmax',
                n_jobs=-1,
                booster='gbtree',
                verbosity=2,
                verbose=True
            )
        )
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    gscv = grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)
    best_model = gscv.best_estimator_
    file = open(log_dir + '/feature.txt', 'a')
    file.write('\nfeature importance\n')
    file.write(best_model['xgb'].feature_importances_.__str__())
    file.close()
    return gscv

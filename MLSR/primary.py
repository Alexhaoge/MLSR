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


def lower_bound(cv_results):
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


def best_low_complexity(cv_results):
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
        verbose: int = 2):
    """
    交叉验证网格搜索，测试集和训练集得分，混淆矩阵和ROC曲线绘制
    Args:
        Xtrain:
        ytrain:
        Xtest:
        ytest:
        pipe:
        grid:
        log_dir:
        score:
        verbose:

    Returns:

    """
    scoring = score
    if scoring is None:
        scoring = {
            'f1': 'f1_macro',
            'accuracy': 'accuracy'
        }
    gsCV = GridSearchCV(
        estimator=pipe,
        cv=5, n_jobs=-1, param_grid=grid,
        scoring=scoring,
        refit='f1',
        verbose=verbose
    )
    gsCV.fit(Xtrain, ytrain)
    dump(gsCV, log_dir + '/gsCV')
    dump(gsCV.best_estimator_, log_dir + '/best_model')
    file_prefix = log_dir + '/' + strftime("%Y_%m_%d_%H_%M_%S", localtime())
    file = open(file_prefix + '.log', 'x')
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

    Args:
        grid:
        dataset:
        log_dir:

    Returns:

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

    Args:
        grid:
        dataset:
        log_dir:

    Returns:

    """
    from sklearn.ensemble import RandomForestRegressor
    if grid is None:
        grid = {
            'rf__criterion': ['gini', 'entropy'],
            'rf__n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],  # 这里数字是随机给的，无根据
            'rf__max_features': ['auto', 'sqrt', 'log2'],
            'rf__max_depth': [int(x) for x in np.linspace(10, 110, num=11)],  # 这里应该加一个None，但不清楚怎么加；这里数字是随机给的，无根据
            'rf__min_samples_split': [2, 5, 10],  # 这里数字是随机给的，无根据
            'rf__min_samples_leaf': [1, 2, 4],  # 这里数字是随机给的，无根据
            'rf__bootstrap': [True, False],
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('rf', RandomForestRegressor())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_SVM(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """

    Args:
        grid:
        dataset:
        log_dir:

    Returns:

    """
    from sklearn.svm import SVC
    if grid is None:
        grid = {
            'SVM__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'SVM__C': [0.1, 1, 10, 100],  # 这里数字是随机给的，无根据
            'SVM__gamma': [1,0.1,0.01,0.001],  # 这里数字是随机给的，无根据
            'SVM__decision_function_shape': ['ovo', 'ovr'],
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('SVM', SVC())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_Logistic(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """

    Args:
        grid:
        dataset:
        log_dir:

    Returns:

    """
    from sklearn.linear_model import LogisticRegression
    if grid is None:
        grid = {
            'Logistic__penalty' : ['l1', 'l2'],
            'Logistic__C' : np.logspace(-4, 4, 20),  # 这里数字是随机给的，无根据
            'Logistic__solver' : ['lbfgs', 'liblinear'],
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('Logistic', LogisticRegression())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)


def do_Naive_Bayes(dataset: DataSet, log_dir: str = '../log', grid: dict = None):
    """

    Args:
        grid:
        dataset:
        log_dir:

    Returns:

    """
    from sklearn.naive_bayes import MultinomialNB
    if grid is None:
        grid = {
            'NB__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],  # 这里数字是随机给的，无根据
        }
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('NB', MultinomialNB())
    ])
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.features, dataset.label, train_size=0.7)
    return grid_search_and_result(Xtrain, ytrain, Xtest, ytest, pipe, grid, log_dir)
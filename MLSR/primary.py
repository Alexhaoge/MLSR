from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, plot_roc_curve
from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import strftime, localtime
from .data import DataSet


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


def plot_confusion_matrix(cm, classes, filename, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.show()


def plot_roc(model, X, y, filename):
    ax = plt.gca()
    dis = plot_roc_curve(model, X, y, ax=ax)
    dis.plot(ax=ax, alpha=0.8)
    plt.savefig(filename)
    plt.show()


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
    file_prefix = log_dir + strftime("/%Y_%m_%d_%H_%M_%S", localtime())
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
        cm = confusion_matrix(ytest, test_prediction)
        plot_confusion_matrix(cm, ['特别困难', '一般困难', '不困难'], file_prefix + '_test_cm.png')
        plot_roc(best_model, Xtest, ytest, file_prefix + '_roc.png')
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

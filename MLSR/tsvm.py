# coding:utf-8
import numpy as np
from numpy import ndarray, ones, hstack, vstack, arange, argmax
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import load, dump
# from sklearn.model_selection import train_test_split,cross_val_score


class TSVM(BaseEstimator, ClassifierMixin):
    """
    TSVM算法实现，参照《机器学习》（周志华） 13.3节
    代码修改自https://github.com/horcham/TSVM
    """

    def __init__(self, Cl: int = 1, Cu: float = 0.001,  kernel: str = 'linear'):
        """
        Initial TSVM
        Args:
            kernel: kernel of svm
            Cl:
            Cu:
        """
        self.Cl = Cl
        self.Cu = Cu
        self.kernel = kernel
        self.clf = SVC(C=1.5, kernel=self.kernel)

    def get_params(self, deep=True):
        """
        实现BaseEstimator的get_params接口
        """
        return {
            'Cl': self.Cl, 
            'Cu': self.Cu,
            'kernel': self.kernel,
            'clf': self.clf
        }

    def set_params(self, **parameters):
        """
        实现BaseEstimator的set_params接口
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def load(self, model_path):
        """
        Load TSVM from model_path

        Args:
            model_path: model path of TSVM
                        model should be svm in sklearn and saved by sklearn.externals.joblib
        """
        self.clf = load(model_path)

    def fit(self, X: ndarray, y: ndarray):
        """
        fit TSVM

        Args:
            X: Input data
                np.array, shape:[n, m], n: numbers of samples with labels, m: numbers of features
            y: labels of input
                np.array, shape:[n, ], n: numbers of samples with labels, -1 for unlabeled
        """
        # self.clf = SVC(C=1.5, kernel=self.kernel)
        X1 = X[y > -1, :]
        X2 = X[y == -1, :]
        Y1 = y[y > -1, :]
        # Y1 = Y1 * 2 - 1
        N = len(X1) + len(X2)
        sample_weight = ones(N)
        sample_weight[len(X1):] = self.Cu

        self.clf.fit(X1, Y1)
        Y2 = self.clf.predict(X2)
        X2_id = arange(len(X2))
        X3 = vstack([X1, X2])
        Y3 = hstack([Y1, Y2])

        while self.Cu < self.Cl:
            self.clf.fit(X3, Y3, sample_weight=sample_weight)
            while True:
                Y2_d = self.clf.decision_function(X2)    # linear: w^Tx + b
                epsilon = 1 - Y2 * Y2_d   # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                positive_max_id = positive_id[argmax(positive_set)]
                negative_max_id = negative_id[argmax(negative_set)]
                a, b = epsilon[positive_max_id], epsilon[negative_max_id]
                if a > 0 and b > 0 and a + b > 2.0:
                    Y2[positive_max_id] = Y2[positive_max_id] * -1
                    Y2[negative_max_id] = Y2[negative_max_id] * -1
                    Y3 = hstack([Y1, Y2])
                    self.clf.fit(X3, Y3, sample_weight=sample_weight)
                else:
                    break
            self.Cu = min(2*self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu

    def score(self, X: ndarray, Y: ndarray, sample_weight=None):
        """
        Calculate accuracy of TSVM by X, Y

        Args:
            X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features
            Y: labels of X
                np.array, shape:[n, ], n: numbers of samples
            sample_weight:

        Returns
        -------
        Accuracy of TSVM
                float
        """
        return self.clf.score(X, Y, sample_weight=sample_weight)

    def predict(self, X):
        """
        Feed X and predict Y by TSVM

        Args:
            X: Input data
                np.array, shape:[n, m], n: numbers of samples, m: numbers of features

        Returns
        -------
        labels of X
                np.array, shape:[n, ], n: numbers of samples
        """
        return self.clf.predict(X)

    def save(self, path):
        """
        Save TSVM to model_path

        Args:
            model_path: model path of TSVM
                        model should be svm in sklearn
        """
        dump(self.clf, path)


# if __name__ == '__main__':
#     import random, examples

#     my_random_generator = random.Random()
#     my_random_generator.seed(0)
#     # X_train_l, L_train_l, X_train_u, X_test, L_test = examples.get_moons_data(my_random_generator)
#     X_train_l, L_train_l, X_train_u, X_test, L_test = examples.get_gaussian_data(my_random_generator)

#     model = TSVM()
#     model.train(X_train_l, L_train_l, X_train_u)
#     Y_hat = model.predict(X_test)
#     # print("Y_hat",Y_hat)
#     accuracy = model.score(X_test, L_test)
#     print("accuracy",accuracy)




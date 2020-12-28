import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from .data import DataSet


def plot_confusion_matrix(cm, classes, filename, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    Args:
        cm: 混淆矩阵，numpy.ndarray
        classes: 类名
        filename: 保存文件路径
        title: 标题
        cmap: 使用的色系

    """
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
    """
    画roc图，不过sklearn只支持二分类roc，三分类画不了
    Args:
        model:
        X:
        y:
        filename:

    """
    ax = plt.gca()
    dis = plot_roc_curve(model, X, y, ax=ax)
    dis.plot(ax=ax, alpha=0.8)
    plt.savefig(filename)
    try:
        plt.show()
    except Exception as e:
        print(e.args)
    


def plot_tsne(data: DataSet, filename: str, n_iter: int = 1000):
    X = MinMaxScaler().fit_transform(data.features)
    X_embed = TSNE(n_components=3, n_iter=n_iter, init='random', n_jobs=-1).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X_embed[:, 0], X_embed[:, 1], X_embed[:, 2],
        c=data.label,
        cmap=plt.cm.Spectral
    )

    plt.legend(loc="best", markerscale=2., numpoints=2, scatterpoints=2, fontsize=12)
    plt.savefig(filename)
    try:
        plt.show()
    except Exception as e:
        print(e.args)


def plot_tsne_ssl(data: DataSet, filename: str, n_iter: int = 1000):
    X = MinMaxScaler().fit_transform(data.features)
    # X = data.features
    X_embed = TSNE(n_components=3, n_iter=n_iter, init='random', n_jobs=-1).fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = {-1: 'grey', 0: 'red', 1: 'yellow', 2: 'blue', 3: 'green'}
    label = {-1: '无标注', 0: '1级困难', 1: '2级困难', 2: '3级困难', 3: '4级困难'}
    for i in data.strong_label.unique():
        X_draw = X_embed[data.strong_label[data.strong_label == i].index.tolist()]
        ax.scatter(
            X_draw[:, 0], X_draw[:, 1], X_draw[:, 2],
            color=color[i],
            alpha=0.2 if i == -1 else 1,
            label=label[i],
            s=8 if i == -1 else 18
        )
    plt.legend(loc="best")
    plt.savefig(filename)
    try:
        plt.show()
    except Exception as e:
        print(e.args)

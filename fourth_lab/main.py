import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize

import utils
from data import Data


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def test_kernel_functions(data):
    C = 50.0
    svc = svm.SVC(kernel='linear', C=C).fit(data.train_data, data.train_classes)
    lin_svc = svm.LinearSVC(C=C).fit(data.train_data, data.train_classes)
    rbf_svc = svm.SVC(kernel='rbf', C=C).fit(data.train_data, data.train_classes)
    sigmoid_svc = svm.SVC(kernel='sigmoid', C=C).fit(data.train_data, data.train_classes)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(data.train_data, data.train_classes)

    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with rbf kernel',
              'SVC with sigmoid kernel',
              'SVC with poly kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, sigmoid_svc, poly_svc)):
        pred = clf.predict(data.test_data)
        print('Accuracy for {}: {:.2%}'.format(titles[i], metrics.accuracy_score(data.test_classes, pred)))


def linear_c_test(data):
    C_range = np.logspace(-2, 7, 10)
    param_grid = dict(C=C_range)

    scores = get_svc_accuracy(param_grid, len(C_range), 1, data)

    draw(scores, C_range, [1], 'c', '')


def poly_c_coef_test(data):
    C_range = np.logspace(-3, 4, 8)
    coef0_range = np.logspace(-4, 3, 8)
    param_grid = dict(coef0=coef0_range, C=C_range)

    scores = get_svc_accuracy(param_grid, len(C_range), len(coef0_range), data)

    draw(scores, coef0_range, C_range, 'coef0', 'C')


def poly_c_degre_test(data):
    C_range = np.logspace(-2, 5, 8)
    degree_range = np.linspace(1, 4.5, 8)
    param_grid = dict(degree=degree_range, C=C_range)

    scores = get_svc_accuracy(param_grid, len(C_range), len(degree_range), data)

    draw(scores, degree_range, C_range, 'degree', 'C')


def get_svc_accuracy(param_grid, fst_length, scnd_length, data):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(kernel='poly'), param_grid=param_grid, cv=cv)
    grid.fit(data.data, data.classes)

    return grid.cv_results_['mean_test_score'].reshape(scnd_length, fst_length)


def draw(scores, fst_range, scnd_range, fst_name, scnd_name):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.86))
    plt.xlabel(fst_name)
    plt.ylabel(scnd_name)
    plt.colorbar()
    plt.xticks(np.arange(len(fst_range)), fst_range, rotation=45)
    plt.yticks(np.arange(len(scnd_range)), scnd_range)
    plt.title('Validation accuracy')
    plt.show()

def main():
    file_data, file_classes = utils.load_data()
    data = Data(file_data, file_classes, 0.1,  0.7, 30)
    test_kernel_functions(data)
    linear_c_test(data)
    #poly_c_coef_test(data)
    #poly_c_degre_test(data)


main()
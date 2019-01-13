import numpy as np


def load_data():
    data = np.loadtxt('../../HAPT Data Set/Train/X_train.txt', delimiter=' ')
    classes = np.loadtxt('../../HAPT Data Set/Train/y_train.txt')
    return data, classes

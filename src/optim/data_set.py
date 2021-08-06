from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import argparse
from sklearn import datasets
import itertools
import numpy as np

def mnist_data(path):
    '''
    Return the MNIST data, fetches it and caches it using sklearn datasets

    Parameters:
    -----------
    path: Directory storing the training and test data

    Returns:
    -----------
    X_train: numpy.ndarray
        X training data
    y_train: numpy.ndarray
        y training labels
    X_test: numpy.ndarray
        X testing data
    y_test: numpy.ndarray
        y testing labels
    '''
    print('Fetching MNIST dataset... ')
    digits = datasets.fetch_mldata('MNIST original')

    X = digits.data
    Y = digits.target

    idx_all = np.random.RandomState(1).permutation(len(Y))
    idx_train = idx_all[:int(.9 * len(Y))]
    idx_test = idx_all[int(.9 *  len(Y)):]

    # TRAIN AND TEST DATA
    X_train, y_train, X_test, y_test = X[idx_train], Y[idx_train], X[idx_test], Y[idx_test]

    return X_train, y_train, X_test, y_test

def gz_data(path):
    '''
    Return the Galaxy Zoo dataset at the given path directory

    Parameters:
    -----------
    path: Directory storing the training and test data

    Returns:
    -----------
    X_train: numpy.ndarray
        X training data
    y_train: numpy.ndarray
        y training labels
    X_test: numpy.ndarray
        X testing data
    y_test: numpy.ndarray
        y testing labels
    '''

    index = 0

    Y = np.load(path + 'training.npy')
    Y = Y[:,index]
    X = np.load(path + 'solutions.npy')

    idx_all = np.random.RandomState(1).permutation(len(Y))
    idx_train = idx_all[:int(.9 * len(Y))]
    idx_test = idx_all[int(.9 *  len(Y)):]

    # TRAIN AND TEST DATA
    X_train, y_train, X_test, y_test = X[idx_train], Y[idx_train], X[idx_test], Y[idx_test]
    return X_train, y_train, X_test, y_test

def neal_data(path):
    '''
    Return the Radford Neal toy data specified at the given path

    Parameters:
    -----------
    path: Directory storing the training and test data

    Returns:
    -----------
    X_train: numpy.ndarray
        X training data
    y_train: numpy.ndarray
        y training labels
    X_test: numpy.ndarray
        X testing data
    y_test: numpy.ndarray
        y testing labels
    '''
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    class_names = iris.target_names

    idx_all = np.random.RandomState(1).permutation(len(Y))
    idx_train = idx_all[:int(.7 * len(Y))]
    idx_test = idx_all[int(.7 *  len(Y)):]

    # TRAIN AND TEST DATA
    X_train, y_train, X_test, y_test = X[idx_train], Y[idx_train], X[idx_test], Y[idx_test]
    return X_train, y_train, X_test, y_test

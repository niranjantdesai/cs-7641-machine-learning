from load_data import load_breast_cancer_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from util import init_weights, compute_loss, perturb_weights, plot_loss
import time
import matplotlib.pyplot as plt


if __name__ == '__main__':
    debug = False

    X, y = load_breast_cancer_data('../data/breast-cancer-wisconsin-data/data.csv')

    # Standardize data
    X = preprocessing.scale(X)

    # Split into training and test data. Use random_state to get the same results in every run
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=18)
    data_train = np.c_[X_train, y_train]
    data_test = np.c_[X_test, y_test]

    np.savetxt('../data/train.csv', data_train, delimiter=',')
    np.savetxt('../data/test.csv', data_test, delimiter=',')
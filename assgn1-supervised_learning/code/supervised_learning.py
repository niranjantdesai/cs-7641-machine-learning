from load_data import load_breast_cancer_data, load_mushroom_data, load_wine_quality_data

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt

import numpy as np

fig_path = '../figures/'

# Read data
#X, y = load_breast_cancer_data('../data/breast-cancer-wisconsin-data/data.csv')
#X, y = load_mushroom_data('../data/mushroom-classification/mushrooms.csv')
X, y = load_wine_quality_data('../data/wine-quality/winequality-white.csv')

# Standardize data
X = preprocessing.scale(X)

# Split into training and test data. Use random_state to get the same results in every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=18)

"""
Decision tree
"""
if False:
    clf_dt = tree.DecisionTreeClassifier(random_state=7)
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of decision tree without hyperparameter tuning is %.2f%%' % (dt_accuracy*100))

# Validation curve
if False:
    param_range_1 = np.arange(1, 21)
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=7), X_train, y_train,
                                                 param_name="max_depth", param_range=param_range_1, cv=5)

    plt.figure()
    plt.plot(param_range_1, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(param_range_1, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for decision tree')
    plt.xlabel('max_depth')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(fig_path + 'dt_validation_curve_1.png')

    param_range_2 = np.arange(2, 21)
    train_scores, test_scores = validation_curve(tree.DecisionTreeClassifier(random_state=7), X_train, y_train,
                                                 param_name="min_samples_split", param_range=param_range_2, cv=5)

    plt.figure()
    plt.plot(param_range_2, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(param_range_2, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Validation curve for decision tree')
    plt.xlabel('min_samples_split')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(fig_path + 'dt_validation_curve_2.png')

# Hyperparameter tuning
if False:
    tuned_params = {'max_depth' : param_range_1, 'min_samples_split' : param_range_2}
    clf_dt = GridSearchCV(tree.DecisionTreeClassifier(random_state=7), param_grid=tuned_params, cv=5)
    clf_dt.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(clf_dt.best_params_)
    y_pred = clf_dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of decision tree is %.2f%%' % (dt_accuracy * 100))

# Learning curve
if False:
    train_sizes = np.linspace(0.1, 1.0, 5)
    _, train_scores, test_scores = learning_curve(clf_dt, X_train, y_train, train_sizes=train_sizes, cv=5)

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    plt.title('Learning curve for decision tree')
    plt.xlabel('Number of training examples')
    plt.ylabel("Classification score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(fig_path + 'dt_learning_curve.png')

"""
Neural network
"""
if False:
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, early_stopping=True, validation_fraction=0.2)
    clf_nn.fit(X_train, y_train)
    y_pred = clf_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of neural network is %.2f%%' % (nn_accuracy * 100))

# Loss curves
if True:
    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1, warm_start=True)
    # Split validation set
    X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=18)
    num_epochs = 5000
    #train_loss = train_scores = val_scores = np.empty(num_epochs)
    train_loss = np.empty(num_epochs)
    train_scores = np.empty(num_epochs)
    val_scores = np.empty(num_epochs)
    for i in range(num_epochs):
        clf_nn.fit(X_train1, y_train1)
        train_loss[i] = clf_nn.loss_
        train_scores[i] = accuracy_score(y_train1, clf_nn.predict(X_train1))
        val_scores[i] = accuracy_score(y_val, clf_nn.predict(X_val))

    y_pred = clf_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of neural network is %.2f%%' % (nn_accuracy * 100))

    xrange = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(xrange, train_loss)
    plt.title('Training loss curve for neural network')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(fig_path + 'nn_train_loss.png')
    #plt.show()

    plt.figure()
    plt.plot(xrange, train_scores, label='Training score')
    plt.plot(xrange, val_scores, label='Validation score')
    plt.title('Training and validation score curve for neural network')
    plt.xlabel('Epochs')
    plt.ylabel("Classification score")
    plt.grid()
    plt.legend(loc="best")
    plt.savefig(fig_path + 'nn_score_curve.png')
    # plt.show()

    clf_nn.set_params(max_iter=1000)

"""
Boosting
"""
if False:
    clf_boosted = AdaBoostClassifier(random_state=7)
    clf_boosted.fit(X_train, y_train)
    y_pred = clf_boosted.predict(X_test)
    boosted_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of Adaboost is %.2f%%' % (boosted_accuracy * 100))

"""
SVM
"""
if False:
    svm_linear = svm.SVC(kernel='linear')
    svm_linear.fit(X_train, y_train)
    y_pred = svm_linear.predict(X_test)
    svm_linear_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM with linear kernel is %.2f%%' % (svm_linear_accuracy * 100))

    svm_poly = svm.SVC(kernel='poly')
    svm_poly.fit(X_train, y_train)
    y_pred = svm_poly.predict(X_test)
    svm_poly_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM with polynomial kernel is %.2f%%' % (svm_poly_accuracy * 100))

    svm_rbf = svm.SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    y_pred = svm_rbf.predict(X_test)
    svm_rbf_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of SVM with RBF kernel is %.2f%%' % (svm_rbf_accuracy * 100))

"""
kNN
"""
if False:
    for k in range(1, 11):
        clf_knn = KNeighborsClassifier(n_neighbors=k)
        clf_knn.fit(X_train, y_train)
        y_pred = clf_knn.predict(X_test)
        clf_knn_accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy of kNN with k = %d is %.2f%%' % (k, clf_knn_accuracy * 100))

pass
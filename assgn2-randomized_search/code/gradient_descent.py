from load_data import load_breast_cancer_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time


if __name__ == '__main__':
    debug = False

    X, y = load_breast_cancer_data('../data/breast-cancer-wisconsin-data/data.csv')
    print('Total number of examples in the dataset: %d' % X.shape[0])
    print('Fraction of positive examples: %.2f%%' % (y[y == 1].shape[0]/y.shape[0]*100.0))

    # Standardize data
    X = preprocessing.scale(X)

    # Split into training and test data. Use random_state to get the same results in every run
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=18)


    clf_nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1000)
    start = time.time()
    clf_nn.fit(X_train, y_train)
    end = time.time()
    print('Finished training in %f seconds' % (end - start))
    y_pred = clf_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
from load_data import load_breast_cancer_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from util import init_weights, compute_loss, perturb_weights, plot_loss
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

    # Initialize neural network
    nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1, warm_start=True)
    nn.fit(X_train, y_train)

    # Randomized search
    step_sz = 1e-1
    max_iters = 5000
    eps = 1e-5
    num_restarts = 10
    losses = np.empty(num_restarts)
    times = np.empty(num_restarts)
    coefs = []
    intercepts = []
    np.random.seed(7)  # seed NumPy's random number generator for reproducibility of results
    for run in range(num_restarts):

        # Initialize weights
        nn.coefs_, nn.intercepts_ = init_weights(X_train.shape[1], list(nn.hidden_layer_sizes))
        loss_next = compute_loss(X_train, y_train, nn)

        train_loss = []
        start = time.time()
        for it in range(max_iters):
            # Save current parameters
            coefs_prev = nn.coefs_
            intercepts_prev = nn.intercepts_
            loss_prev = loss_next

            if debug:
                print('Iteration %d' % it)
                print('Loss = ', loss_prev)
                print()

            # Update parameters
            nn.coefs_ = perturb_weights(nn.coefs_, step_sz)
            nn.intercepts_ = perturb_weights(nn.intercepts_, step_sz)

            # Keep the updated parameters only if the loss using them decreases
            loss_next = compute_loss(X_train, y_train, nn)
            if loss_next >= loss_prev:
                nn.coefs_ = coefs_prev
                nn.intercepts_ = intercepts_prev
                loss_next = loss_prev

            # train_loss[it] = loss_next
            train_loss.append(loss_next)

            diff = loss_prev - loss_next
            if diff > 0 and diff < eps:
                break

        end = time.time()
        times[run] = end - start

        losses[run] = train_loss[-1]
        coefs.append(nn.coefs_)
        intercepts.append(nn.intercepts_)

    # # Plot loss curve
    # plot_loss(train_loss, filename='../plots/nn_loss_rhc.png', title='Training loss curve: randomized hill climbing')

    # Find best weights
    idx = np.argmin(losses)
    nn.coefs_ = coefs[idx]
    nn.intercepts_ = intercepts[idx]

    # Plot losses across runs
    plot_loss(losses, filename='../plots/nn_runs_rhc.png', title='Losses for different runs with RHC', xlabel='Run',
              ylabel='Loss')

    # Find accuracy on the test set
    y_pred = nn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy on test data using randomized hill climbing is %.2f%%' % (test_accuracy * 100))

    # Timing information
    print('Average time per RHC run is %f seconds' % np.mean(times))

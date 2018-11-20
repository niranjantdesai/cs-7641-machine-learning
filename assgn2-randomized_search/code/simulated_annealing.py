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
    print('Total number of examples in the dataset: %d' % X.shape[0])
    print('Fraction of positive examples: %.2f%%' % (y[y == 1].shape[0]/y.shape[0]*100.0))

    # Standardize data
    X = preprocessing.scale(X)

    # Split into training and test data. Use random_state to get the same results in every run
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=18)

    ### Simulated annealing
    # Set params
    # T_init = 1000
    # T_min = 1e-6
    # decay = 0.95
    # num_iters = 100     # number of iterations for a given temperature
    # step_size = 0.1
    #
    # T = T_init
    # loss = []
    # while T > T_min:
    #     if debug:
    #         print()
    #         print('Temperature: ', T)
    #     for i in range(num_iters):
    #         # Save current parameters
    #         coefs_prev = nn.coefs_
    #         intercepts_prev = nn.intercepts_
    #         loss_prev = loss_next
    #
    #         if debug:
    #             print('Iteration # %d' % i)
    #             print('Loss = ', loss_prev)
    #
    #         # Update parameters
    #         nn.coefs_ = perturb_weights(nn.coefs_, step_size)
    #         nn.intercepts_ = perturb_weights(nn.intercepts_, step_size)
    #         loss_next = compute_loss(X_train, y_train, nn)
    #
    #         # Metropolis criterion for updating weights
    #         prob = np.exp((loss_prev - loss_next) / T)
    #         rand = np.random.rand()
    #         if loss_next < loss_prev or prob >= rand:
    #             pass
    #         else:
    #             nn.coefs_ = coefs_prev
    #             nn.intercepts_ = intercepts_prev
    #             loss_next = loss_prev
    #
    #         loss.append(loss_next)
    #
    #     T *= decay

    T_init = 1e5
    # decay = np.arange(0.25, 0.96, 0.1)
    decay = np.arange(0.75, 0.76, 0.1)
    num_iters = 5000  # number of iterations for a given temperature
    step_size = 0.1
    losses = np.empty(decay.size)
    test_accs = np.empty(decay.size)

    for idx, decay_rate in enumerate(decay):
        np.random.seed(7)  # seed NumPy's random number generator for reproducibility of results
        # Initialize neural network
        nn = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=7, max_iter=1, warm_start=True)
        nn.fit(X_train, y_train)

        # Initialize weights
        nn.coefs_, nn.intercepts_ = init_weights(X_train.shape[1], list(nn.hidden_layer_sizes))
        loss_next = compute_loss(X_train, y_train, nn)

        T = T_init
        loss = []
        start = time.time()
        for i in range(num_iters):
            # Save current parameters
            coefs_prev = nn.coefs_
            intercepts_prev = nn.intercepts_
            loss_prev = loss_next

            if debug:
                print('Iteration # %d' % i)
                print('Loss = ', loss_prev)

            # Update parameters
            nn.coefs_ = perturb_weights(nn.coefs_, step_size)
            nn.intercepts_ = perturb_weights(nn.intercepts_, step_size)
            loss_next = compute_loss(X_train, y_train, nn)

            # Metropolis criterion for updating weights
            prob = np.exp((loss_prev - loss_next) / T)
            rand = np.random.rand()
            if loss_next < loss_prev or prob >= rand:
                pass
            else:
                nn.coefs_ = coefs_prev
                nn.intercepts_ = intercepts_prev
                loss_next = loss_prev

            loss.append(loss_next)
            # diff = loss_prev - loss_next
            # if diff > 0 and diff < eps:
            #     break
            T *= decay_rate

        end = time.time()
        runtime = end - start

        losses[idx] = loss[-1]

        # # Plot loss
        plot_loss(loss, filename='../plots/nn_loss_sa_new.png', title='Training loss curve: simulated annealing')

        # Find accuracy on the test set
        y_pred = nn.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy on test data using simulated annealing is %.2f%%' % (test_accuracy * 100))
        test_accs[idx] = test_accuracy*100

        # # Timing information
        # print('Time taken to complete simulated annealing is %f seconds' % runtime)

    # Plot losses and test accuracies for different decay rates
    # fig, ax = plt.subplots()
    # plt.grid()
    # ax.plot(decay, losses, color='tab:blue', label='Training loss')
    # ax.set_xlabel('Temperature decay rate')
    # ax.set_ylabel('Loss')
    # ax2 = ax.twinx()
    # ax2.plot(decay, test_accs, color='tab:red', label='Test accuracy (%)')
    # ax2.set_ylabel('Accuracy (%)')
    # lines = ax.get_lines() + ax2.get_lines()
    # ax.legend(lines, [line.get_label() for line in lines])
    # ax.set_title('SA performance for different decay rates')
    # fig.tight_layout()
    # plt.savefig('../plots/nn_sa_decay_rates.png')
    # plt.figure()
    # plt.plot(decay, losses)
    # plt.title('SA performance for different decay rates')
    # plt.xlabel('Temperature decay rate')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.savefig('../plots/nn_sa_decay_rates.png')
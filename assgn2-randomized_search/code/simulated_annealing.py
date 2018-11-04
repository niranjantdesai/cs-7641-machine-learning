from load_data import load_breast_cancer_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from util import init_weights, compute_loss, perturb_weights, plot_loss


class SimAnnealStep:
    """
    Take a Monte Carlo step for simulated annealing by perturbing the weights of a neural network
    """
    def __init__(self, step_size=0.5):
        """
        :param step_size: initial step
        """
        self.step_size = step_size

    def __call__(self, x):
        """
        Randomly displaces coefficients and intercepts of a neural network
        :param x: ndarray containing lists comprising the coefficients and intercepts of a neural network
        :return: x_new: randomly displaced coefficients and intercepts
        """
        coeffs = x[0]
        intercepts = x[1]

        coeffs_new = perturb_weights(coeffs, self.step_size)
        intercepts_new = perturb_weights(intercepts, self.step_size)
        x_new = np.array([coeffs_new, intercepts_new])

        return x_new


def compute_loss_helper(weights, X, y_true, net):
    """
    Helper function to compute loss of a neural network for simulated annealing
    :param weights: ndarray of coefficients and biases
    :param X: data
    :param y_true: ground truth
    :param net: MLPClassifier object
    :return: loss: loss of a neural network
    """
    net.coefs_ = weights[:len(net.coefs_)]
    net.intercepts_ = weights[len(net.coefs_):]
    loss = compute_loss(X, y_true, net)

    return loss


if __name__ == '__main__':
    debug = True
    np.random.seed(7)   # seed NumPy's random number generator for reproducibility of results

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

    # Initialize weights
    nn.coefs_, nn.intercepts_ = init_weights(X_train.shape[1], list(nn.hidden_layer_sizes))
    initial_wts = np.array([nn.coefs_, nn.intercepts_])    # create a ndarray containing both coefficients and biases
    wts = initial_wts
    loss_prev = compute_loss_helper(wts, X_train, y_train, nn)

    ### Simulated annealing
    # Set params
    T_init = 2500
    T_min = 2.5
    decay = 0.95
    num_iters = 100     # number of iterations for a given temperature

    nn_takestep = SimAnnealStep()
    T = T_init
    loss =[]
    while T > T_min:
        if debug:
            print()
            print('Temperature: ', T)
        for i in range(num_iters):
            if debug:
                print('Iteration # %d' % i)
            wts_new = nn_takestep(wts)
            loss_new = compute_loss_helper(wts_new, X_train, y_train, nn)

            # Metropolis criterion for updating weights
            prob = np.exp((loss_new - loss) / T)
            rand = np.random.rand()
            if loss_new < loss_prev or prob >= rand:
                wts = wts_new
                loss_prev = loss_new

        T *= decay
        loss.append(loss_prev)

    # Plot loss
    plot_loss(loss, title='Training loss curve: simulated annealing')

    # Find accuracy on the test set
    y_pred = nn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy on test data using randomized hill climbing is %.2f%%' % (test_accuracy * 100))
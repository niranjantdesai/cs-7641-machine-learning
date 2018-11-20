from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt


def init_weights(n_features, hidden_layer_sizes):
    """
    Initializes weights of a neural network
    :param n_features: number of features in input
    :param hidden_layer_sizes: list of layer sizes
    :return: coeffs: coefficients
    :return: biases: intercepts or biases
    """

    layer_units = ([n_features] + hidden_layer_sizes + [1])
    n_layers_ = len(layer_units)

    # Initialize coefficient and intercept layers
    coeffs = []
    biases = []

    for i in range(n_layers_ - 1):
        coef_init, intercept_init = init_coef(layer_units[i], layer_units[i + 1])
        coeffs.append(coef_init)
        biases.append(intercept_init)

    return coeffs, biases


def init_coef(fan_in, fan_out):
    """
    Helper function for initializing weights of a neural network. Uses the initialization method recommended by Glorot
    et al.
    :param fan_in: number of neurons in the current layer
    :param fan_out: number of neurons in the next layer
    :return: coefficients and intercepts (biases)
    """

    factor = 2.
    init_bound = np.sqrt(factor / (fan_in + fan_out))
    rand_state = np.random.RandomState(seed=7)

    # Generate weights and bias:
    coef_init = rand_state.uniform(-init_bound, init_bound, (fan_in, fan_out))
    intercept_init = rand_state.uniform(-init_bound, init_bound, fan_out)
    return coef_init, intercept_init


def compute_loss(X, y_true, net):
    """
    Compute cross entropy loss of a neural network with regularization
    :param X: data
    :param y_true: ground truth
    :param net: MLPClassifier object
    :return: loss: sum of cross entropy loss and regularization loss
    """
    y_eval = net.predict_proba(X)
    cross_entropy_loss = log_loss(y_true, y_eval, eps=1e-10)
    values = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in net.coefs_]))
    reg_loss = 0.5 * net.alpha * values / y_true.shape[0]
    net_loss = cross_entropy_loss + reg_loss

    return net_loss


def perturb_weights(weights, step_size):
    """
    Randomly perturbs weights
    :param weights: list of weights
    :param step_size: maximum absolute perturbation
    :return: perturbed_wts: perturbed weights
    """
    perturbed_wts = []
    for layer_wts in weights:
        delta = np.reshape(np.random.uniform(-step_size, step_size, layer_wts.shape), layer_wts.shape)
        perturbed_wts.append(layer_wts + delta)

    return perturbed_wts


def plot_loss(loss, filename, title='Loss curve', xlabel='Iterations', ylabel='Loss'):
    """
    Plots loss curve
    :param loss: list/ndarray of losses at each iteration
    :param title: title of the plot
    :param xlabel: label of the X axis
    :param ylabel: label of the Y axis
    """
    loss = np.array(loss)
    xrange = np.arange(loss.size) + 1
    plt.figure()
    plt.plot(xrange, loss)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(filename)
    # plt.show()
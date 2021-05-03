import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from matplotlib.ticker import MaxNLocator


def plot_loss_metrics(history_file):
    """ Plots the loss function and metrics of a trained model.
    # Arguments
        history_file: value of the true dependent variable
    # Returns
        A plot with the loss and metric curves.
    """
    history = pickle.load(open(history_file, "rb"))
    loss, metric, val_loss, val_metric = islice(history.keys(), 4)
    n_epochs = len(history[loss])

    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(13, 8))

    ax1.set_title(loss)
    ax1.plot(np.arange(1, n_epochs + 1), history[loss], label='train')
    ax1.plot(np.arange(1, n_epochs + 1), history[val_loss], label='test')
    ax1.legend()

    ax2.set_title(metric)
    ax2.plot(np.arange(1, n_epochs + 1), history[metric], label='train')
    ax2.plot(np.arange(1, n_epochs + 1), history[val_metric], label='test')
    ax2.set_xlabel('Epochs')
    ax2.set_xlim((1, n_epochs + 1))
    xa = ax2.get_xaxis()
    xa.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()
    plt.savefig(history_file + '.png')
    plt.show()


plot_loss_metrics('history_no_testT1_stack')

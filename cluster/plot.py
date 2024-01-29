from utils import get_file_path
import matplotlib.pyplot as plt

def plot_2d(x, y, clusters=None, save=''):
    """
    use pca to plot in 2d given embeddings and optionally clusters
    """
    if not (clusters is None):
        plt.scatter(x, y, c=clusters, cmap='rainbow')
    else:
        plt.scatter(x, y)


    if save != '':
        save_path = get_file_path(save, 'plots')
        plt.savefig(save_path)

    plt.clf()

def plot_parameter_search(p_list, error_list, xlabel, ylabel, save=''):
    plt.plot(p_list, error_list, 'bx-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save != '':
        save_path = get_file_path(save, 'plots')
        plt.savefig(save_path)

    plt.clf()

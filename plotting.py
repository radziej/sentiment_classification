import os
import itertools

import matplotlib.pyplot as plt
import seaborn as sns


def palette():
    return itertools.cycle(sns.color_palette())


def save_figure(plot, name, suffix='.png', directory='plots'):
    '''Save pyplot/seaborn plot to the directory with the given name'''

    if hasattr(plot, 'savefig'):
        plot.savefig(os.path.join(directory, name + suffix))
    elif hasattr(plot, 'figure'):
        plot.figure.savefig(os.path.join(directory, name + suffix))
    else:
        raise ValueError('Attempted to save an object '
                         'that is not a pyplot/seaborn figure ' + str(plot))
    plt.close()


def confusion_heatmap(confusion_matrix, labels, normalize=False):
    if normalize:
        confusion_matrix = (confusion_matrix.astype('float') /
                            confusion_matrix.sum(axis=1)[:, np.newaxis])
    hm = sns.heatmap(confusion_matrix, annot=True,
                     xticklabels=labels,
                     yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

import os
import itertools

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def palette():
    return itertools.cycle(sns.color_palette())


def save_figure(plot, name, suffix='.pdf', directory='plots'):
    '''Save pyplot/seaborn plot to the directory with the given name'''

    plt.tight_layout()

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
    plt.figure()
    sns.heatmap(confusion_matrix, annot=True, xticklabels=labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, va='center')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')


def cross_validation_indices(cv, X, y, ax, n_splits, lw=10):
    '''Create a sample plot for indices of a cross-validation object.'''

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Paired)

    # ax.scatter(range(len(X)), [ii + 2.5] * len(X),
    #            c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+1.2, -.2], xlim=[0, len(X)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

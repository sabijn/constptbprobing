import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Helper function to plot confusion matrix
def plot_confusion_matrix(cm, labels, name):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix: relative shared levels',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(round(cm[i, j], 1)) if cm[i, j] > 0 else '0',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    plt.savefig(name)

def compute_confusion_matrix(path):
    df_dev = pd.read_csv(path, sep='\t')
    # read in test_shared_labels
    with open(test_shared_labels, 'r') as infile:
        shared_labels = infile.readlines()

    y_true = np.array([label.replace('\n', '') for sent in shared_labels for label in sent.split(' ')])
    y_pred = df_dev['pred_label'].to_numpy()

    cm_own = confusion_matrix(y_true, y_pred, normalize='true')

    return cm_own, np.unique(y_true)

if __name__ == '__main__':
    full_tree_path = 'parsing-as-pretraining/exp_trees/distilbert-base-uncased/concat_lev/results.p'
    test_results_tree = 'parsing-as-pretraining/exp_trees/distilbert-base-uncased/concat_lev/pred_labels_test.tsv'
    lca_path = 'exp_lca/distilbert-base-uncased/orig2orig_concat/results.p'
    test_shared_labels = 'parsing-as-pretraining/exp_trees/test_shared_levels.txt'

    # with open(full_tree_path,'rb') as infile:
    #     results = pickle.load(infile)

    cm, labels = compute_confusion_matrix(test_results_tree)
    plot_confusion_matrix(cm, labels, 'visualizations_own/confusion_matrix_lev_dev_own.png')




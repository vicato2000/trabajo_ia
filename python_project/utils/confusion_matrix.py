from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from common_path import ROOT_PATH
import os


def plot_confusion_matrix(true, pred, confusion_matrix_path, k,
                          improve_filter=False):
    matrix = confusion_matrix(y_true=true, y_pred=pred,
                              labels=['spam', 'no_spam'])

    cm_display = ConfusionMatrixDisplay(matrix,
                                        display_labels=['spam', 'no_spam'])
    cm_display.plot()
    plt.title('k = {}'.format(k))

    if not improve_filter:
        plt.savefig(
            '{}{}{}confusion_matrix_k{}'.format(ROOT_PATH,
                                                confusion_matrix_path,
                                                os.path.sep, k))
    else:
        plt.savefig(
            '{}{}{}confusion_matrix_k{}_improve_filter'
                .format(ROOT_PATH, confusion_matrix_path, os.path.sep, k))
    plt.show()

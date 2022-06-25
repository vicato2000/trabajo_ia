from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from common_path import ROOT_PATH
import os


def plot_confusion_matrix(true, pred, confusion_matrix_path, k,
                          improve_filter=False):
    """
    Función que muestra y guarda una matriz de confusión.

    :param list true: Lista de elementos correctamente clasificados.
    :param list pred: Lista de elementos supuestamente bien clasificados.
    :param pathlib.Path confusion_matrix_path: Ruta donde se va a guardar la
        imagen con la matriz de confusión.
    :param int k: Valor del hiperparámetro de suavizado o de KNN.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
    encuentra a False.
    """

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

from sklearn.feature_extraction.text import CountVectorizer
from python_project.utils.reader import read_emails, read_email, \
    get_dic_train_email
from python_project.utils.vocabulary import get_vocabulary
from common_path import ROOT_PATH
from pathlib import Path
import pickle as pk
import os
import numpy as np
from math import log
from python_project.utils.confusion_matrix import plot_confusion_matrix
from python_project.utils.improvements import improve

VOCABULARY_JSON = Path('python_project', 'utils', 'generated_documents',
                       'vocabulary.json')
NO_SPAM_TRAIN_PATH = Path('python_project', 'split_email_folder', 'train',
                          'legítimo')
SPAM_TRAIN_PATH = Path('python_project', 'split_email_folder', 'train',
                       'no_deseado')
VAL_LEGITIMO = Path('python_project', 'split_email_folder', 'val', 'legítimo')

VAL_SPAM = Path('python_project', 'split_email_folder', 'val', 'no_deseado')


def train_bag_of_words(k=1, improve_filter=False):
    """
    Función que entrena al modelo de Bolsa de Palabras.

    :param int k: Hiperparámetro de suavizado. Por defecto es 1.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: Una tupla que contiene los pesos del modelo entrenado con el
        conjunto de entrenamiento spam y no spam.
    """

    vocabulary_list = get_vocabulary(improve_filter)

    vectorizer_no_spam = CountVectorizer(stop_words='english',
                                         vocabulary=vocabulary_list,
                                         lowercase=True,
                                         analyzer='word')

    vectorizer_spam = CountVectorizer(stop_words='english',
                                      vocabulary=vocabulary_list,
                                      lowercase=True,
                                      analyzer='word')

    dic_corpus = get_corpus(True, improve_filter)

    corpus_no_spam = [dic_corpus[e] for e in dic_corpus.keys()
                      if e.startswith('no_spam')]
    corpus_spam = [dic_corpus[e] for e in dic_corpus.keys()
                   if e.startswith('spam')]

    matrix_no_spam = vectorizer_no_spam.fit_transform(corpus_no_spam)
    matrix_spam = vectorizer_spam.fit_transform(corpus_spam)

    array_no_spam = np.ndarray.tolist(matrix_no_spam.toarray())
    array_spam = np.ndarray.tolist(matrix_spam.toarray())

    vector_no_spam = list(map(sum, zip(*array_no_spam)))
    vector_spam = list(map(sum, zip(*array_spam)))

    total_no_spam = sum(vector_no_spam) + k * len(vocabulary_list)
    total_spam = sum(vector_spam) + k * len(vocabulary_list)

    vector_no_spam_softened = list(map(lambda x: (x + k) / total_no_spam,
                                       vector_no_spam))
    vector_spam_softened = list(map(lambda x: (x + k) /
                                              total_spam, vector_spam))

    if not improve_filter:
        vector_no_spam_softened_json = Path('python_project', 'bag_of_word',
                                            'generated_documents',
                                            'vector_no_spam_softened_k{}.json'
                                            .format(k))
        vector_spam_softened_json = Path('python_project', 'bag_of_word',
                                         'generated_documents',
                                         'vector_spam_softened_k{}.json'
                                         .format(k))
    else:
        vector_no_spam_softened_json = Path('python_project', 'bag_of_word',
                                            'generated_documents',
                                            'vector_no_spam_softened_k{}'
                                            '_improve_filter.json'
                                            .format(k))
        vector_spam_softened_json = Path('python_project', 'bag_of_word',
                                         'generated_documents',
                                         'vector_spam_softened_k{}'
                                         '_improve_filter.json'
                                         .format(k))
    with open(ROOT_PATH + '{}'.format(vector_no_spam_softened_json),
              'wb') as f:
        pk.dump(vector_no_spam_softened, f)

    with open(ROOT_PATH + '{}'.format(vector_spam_softened_json), 'wb') as f:
        pk.dump(vector_spam_softened, f)

    return vector_no_spam_softened, vector_spam_softened


def classify_email_bow(email_path, k=1, improve_filter=False):
    """
    Función que clasifica un email como spam o no_spam.

    :param str email_path: Ruta donde se encuentra el email a clasificar.
    :param int k: Hiperparámetro de suavizado.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: La clase en la que se ha clasificado.
    """

    email = read_email(email_path)

    if improve_filter:
        email = improve(email)

    vector_spam_softened, vector_no_spam_softened = \
        get_vectors_softened(k, improve_filter)

    vocabulary_list = get_vocabulary(improve_filter)

    vectorizer = CountVectorizer(stop_words='english',
                                 vocabulary=vocabulary_list,
                                 lowercase=True,
                                 analyzer='word')

    new_matrix = vectorizer.fit_transform([email])
    new_vector = np.ndarray.tolist(new_matrix.toarray())

    total_train_spam = len(os.listdir(ROOT_PATH +
                                      '{}'.format(SPAM_TRAIN_PATH)))

    total_train_no_spam = len(os.listdir(ROOT_PATH +
                                         '{}'.format(NO_SPAM_TRAIN_PATH)))

    total = total_train_no_spam + total_train_spam

    p_spam_list = [new_vector[0][i] * log(vector_spam_softened[i])
                   for i in range(len(vector_spam_softened))]

    p_spam = log(total_train_spam / total) + sum(p_spam_list)

    p_no_spam_list = [new_vector[0][i] * log(vector_no_spam_softened[i])
                      for i in range(len(vector_no_spam_softened))]

    p_no_spam = log(total_train_no_spam / total) + sum(p_no_spam_list)

    return 'spam' if p_spam > p_no_spam else 'no_spam'


def classify_emails_bow(folder_path, val_emails=None, k=1,
                        improve_filter=False):
    """
    Función que clasifica los emails de prueba como spam o no_spam.

    :param str folder_path: Ruta de la carpeta a clasificar. Se usaría en el
        caso de que val_emails fuera None
    :param list[str] val_emails: Lista cuerpos de emails a clasificar. Por
        defecto es None.
    :param int k: Hiperparámetro de suavizado.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: Lista con las clases en las que se ha clasificado los emails.
    """

    emails = None

    if val_emails is None:
        emails = read_emails(folder_path)
    else:
        emails = val_emails

    vector_spam_softened, vector_no_spam_softened = \
        get_vectors_softened(k, improve_filter)

    vocabulary_list = get_vocabulary(improve_filter)

    vectorizer = CountVectorizer(stop_words='english',
                                 vocabulary=vocabulary_list,
                                 lowercase=True,
                                 analyzer='word')

    new_matrix = vectorizer.fit_transform(emails)
    new_vector = np.ndarray.tolist(new_matrix.toarray())

    total_train_spam = len(os.listdir(ROOT_PATH +
                                      '{}'.format(SPAM_TRAIN_PATH)))

    total_train_no_spam = len(os.listdir(ROOT_PATH +
                                         '{}'.format(NO_SPAM_TRAIN_PATH)))

    total = total_train_no_spam + total_train_spam

    result = []

    for v in new_vector:
        p_spam_list = [v[i] * log(vector_spam_softened[i])
                       for i in range(len(vector_spam_softened))]

        p_spam = log(total_train_spam / total) + sum(p_spam_list)

        p_no_spam_list = [v[i] * log(vector_no_spam_softened[i])
                          for i in range(len(vector_no_spam_softened))]

        p_no_spam = log(total_train_no_spam / total) + sum(p_no_spam_list)

        result.append('spam' if p_spam > p_no_spam else 'no_spam')

    return result


def get_vectors_softened(k=1, improve_filter=False):
    """
    Función que genera o obtine los pesos del modelo entrenado dependiendo de
    si están o no generados sus archivos json.

    :param int k: Hiperparámetro de suavizado.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: Una tupla que contiene los pesos del modelo entrenado con el
        conjunto de entrenamiento spam y no spam.
    """

    if not improve_filter:
        vector_no_spam_softened_json_path = Path('python_project',
                                                 'bag_of_word',
                                                 'generated_documents',
                                                 'vector_no_spam_softened_k{}'
                                                 '.json'.format(k))
        vector_spam_softened_json_path = Path('python_project', 'bag_of_word',
                                              'generated_documents',
                                              'vector_spam_softened_k{}.json'
                                              .format(k))
    else:
        vector_no_spam_softened_json_path = Path('python_project',
                                                 'bag_of_word',
                                                 'generated_documents',
                                                 'vector_no_spam_softened_k{}'
                                                 '_improve_filter.json'
                                                 .format(k))
        vector_spam_softened_json_path = Path('python_project', 'bag_of_word',
                                              'generated_documents',
                                              'vector_spam_softened_k{}'
                                              '_improve_filter.json'
                                              .format(k))

    vector_no_spam_softened_json = os.path.exists(
        ROOT_PATH + '{}'.format(vector_no_spam_softened_json_path))

    vector_spam_softened_json = os.path.exists(
        ROOT_PATH + '{}'.format(vector_spam_softened_json_path))

    vector_spam_softened = None
    vector_no_spam_softened = None

    if vector_spam_softened_json and vector_no_spam_softened_json:
        with open(ROOT_PATH + '{}'.format(vector_no_spam_softened_json_path),
                  'rb') as f:
            vector_no_spam_softened = pk.load(f)

        with open(ROOT_PATH + '{}'.format(vector_spam_softened_json_path),
                  'rb') as f:
            vector_spam_softened = pk.load(f)
    else:
        vector_no_spam_softened, vector_spam_softened = \
            train_bag_of_words(k, improve_filter)

    return vector_spam_softened, vector_no_spam_softened


def generate_confusion_matrix(k=1, improve_filter=False):
    """
    Función que genera las matrices de confusión de los correos de
    entrenamiento para el modelo de Bolsa de Palabras.

    :param int k: Hiperparámetro de suavizado.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    """

    confusion_matrix_path = Path('python_project', 'bag_of_word',
                                 'generated_documents')

    dic_emails = get_corpus(False, improve_filter)

    list_spam_emails = [dic_emails[e] for e in dic_emails.keys()
                        if e.startswith('spam')]

    pred_spam = classify_emails_bow('', list_spam_emails, k, improve_filter)
    true_spam = ['spam' for i in range(len(pred_spam))]

    list_no_spam_emails = [dic_emails[e] for e in dic_emails.keys()
                           if e.startswith('no_spam')]

    pred_no_spam = classify_emails_bow('', list_no_spam_emails, k,
                                       improve_filter)
    true_no_spam = ['no_spam' for i in range(len(pred_no_spam))]

    pred = pred_spam + pred_no_spam
    true = true_spam + true_no_spam

    plot_confusion_matrix(true, pred, confusion_matrix_path, k, improve_filter)


def get_corpus(train=True, improve_filter=False):
    """
    Función que obtiene el corpus de entrenamiento o de prueba.

    :param bool train: Leer la carpeta de entrenamiento o de prueba.
        Por defecto se lee la de entrenamiento.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False. Por defecto es 1.
    :return: Diccionario donde la clave es la clase correcta a la que
        pertenece ese email, spam o no_spam, y el valor es el cuerpo del email.
    :rtype: dict[str,str]
    """

    return get_dic_train_email(train, improve_filter)

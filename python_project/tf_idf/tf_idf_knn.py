import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from python_project.utils.confusion_matrix import plot_confusion_matrix
from python_project.utils.reader import get_dic_train_email, read_email
from sklearn.metrics.pairwise import cosine_similarity
from common_path import ROOT_PATH
from pathlib import Path
from collections import Counter
import pickle as pk
from python_project.utils.improvements import improve
import re
from python_project.utils.vocabulary import get_vocabulary

VOCABULARY_JSON = Path('python_project', 'utils', 'generated_documents',
                       'vocabulary.json')

TF_IDF_VECTOR_JSON = Path('python_project', 'tf_idf', 'generated_documents',
                          'tf_idf_vector.json')

TF_IDF_VECTORS_JSON = Path('python_project', 'tf_idf',
                           'generated_documents',
                           'tf_idf_vectors.json')

TF_IDF_VECTOR_JSON_IMPROVED = Path('python_project', 'tf_idf',
                                   'generated_documents',
                                   'tf_idf_vector_improved.json')

TF_IDF_VECTORS_JSON_IMPROVED = Path('python_project', 'tf_idf',
                                    'generated_documents',
                                    'tf_idf_vectors'
                                    '_improved.json')


def tf_idf_vector(improve_filter=False):
    """
    Función donde se entrena al modelo TF-IDF a partir del conjunto de
    entrenamiento.

    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: Una tupla donde se devuelve el TfIdfVectorizer entrenado y una
        matriz con los pesos de TF-IDF del conjunto de entrenamiento.
    :rtype: tuple[TfidfVectorizer, scipy.sparse.csr.csr_matrix]
    """

    corpus = get_corpus(True, improve_filter)

    vocabulary_list = get_vocabulary(improve_filter)

    vector = TfidfVectorizer(vocabulary=vocabulary_list,
                             stop_words='english',
                             lowercase=True)
    if improve_filter:
        corpus = {e: improve(corpus[e]) for e in corpus}

        vectors = vector.fit_transform(corpus.values())

        with open(ROOT_PATH + '{}'.format(TF_IDF_VECTOR_JSON_IMPROVED), 'wb') \
                as f:
            pk.dump(vector, f)

        with open(ROOT_PATH +
                  '{}'.format(TF_IDF_VECTORS_JSON_IMPROVED), 'wb') \
                as f:
            pk.dump(vectors, f)
    else:
        vectors = vector.fit_transform(corpus.values())

        with open(ROOT_PATH + '{}'.format(TF_IDF_VECTOR_JSON), 'wb') as f:
            pk.dump(vector, f)

        with open(ROOT_PATH + '{}'.format(TF_IDF_VECTORS_JSON), 'wb') \
                as f:
            pk.dump(vectors, f)

    return vector, vectors


def classify_email(email_path, k=1, improve_filter=False):
    """
    Función que clasifica un email como spam o no_spam.

    :param str email_path: Ruta del email a clasificar.
    :param int k: Valor del parámetro k de KNN. Por defecto es 1.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: La clase en la que se clasifica, spam o no spam.
    :rtype: str
    """

    email = read_email(email_path)

    if improve_filter:
        email = improve(email)

    vector, vectors = get_tf_idf_vector()
    corpus = get_corpus(True, improve_filter)

    new_vector = vector.transform([email])

    m_similitary = cosine_similarity(vectors, new_vector)

    dic_m_similitary = {}
    i = 0
    for e in corpus.keys():
        dic_m_similitary[e] = m_similitary[i]
        i += 1

    classes = sorted(dic_m_similitary.keys(),
                     key=lambda x: list(dic_m_similitary[x])[0])[-k:]
    classes = list(map(map_name_to_class, classes.copy()))

    return Counter(classes).most_common()[0][0]


def classify_emails(k=1, improve_filter=False):
    """
        Función que clasifica los emails de prueba como spam o no_spam.

        :param int k: Valor del parámetro k de KNN. Por defecto es 1.
        :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto
            se encuentra a False. Por defecto es 1.
        :return: Un diccionario en que las claves son las clases correctas, y
            los valores son las clases predichas.
        :rtype: dict[str,str]
        """

    dic = get_corpus(False, improve_filter)

    if improve_filter:
        dic = {e: improve(dic[e]) for e in dic.keys()}

    vector, vectors = get_tf_idf_vector(improve_filter)
    corpus = get_corpus(True, improve_filter)

    new_vector = vector.transform(dic.values())

    m_similitary = cosine_similarity(new_vector, vectors)

    dic_m_similitary = {}
    i = 0
    for e in dic.keys():
        dic_m_similitary[e] = m_similitary[i]
        i += 1

    dic_index = {}
    for e in dic.keys():
        res = sorted(range(len(dic_m_similitary[e])),
                     key=lambda index: dic_m_similitary[e][index])[-k:]
        dic_index[e] = res

    dic_classes = {}
    for e in dic_index.keys():
        classes = map(map_name_to_class, [list(corpus.keys())[i]
                                          for i in dic_index[e]])
        dic_classes[e] = classes

    dic_class = {e: Counter(dic_classes[e]).most_common()[0][0]
                 for e in dic_classes.keys()}

    return dic_class


def generate_confusion_matrix(k, improve_filter=False):

    """
    Función que genera las matrices de confusión de los correos de
    entrenamiento para el modelo TF-IDF.

    :param int k: Valor del parámetro k de KNN. Por defecto es 1.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    """

    confusion_matrix_path = Path('python_project', 'tf_idf',
                                 'generated_documents')

    dic_class = classify_emails(k, improve_filter)

    true = [map_name_to_class(c) for c in dic_class.keys()]
    pred = [c for c in dic_class.values()]

    plot_confusion_matrix(true, pred, confusion_matrix_path, k, improve_filter)


def get_tf_idf_vector(improve_filter=False):
    """
    Función que lee o genera el vector TfIdfVectoricer y la matriz de pesos
    dependiendo de si se han generado o no sus archivos json previamente.

    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False. Por defecto es 1.
    :return: Una tupla donde se devuelve el TfIdfVectorizer entrenado y una
        matriz con los pesos de TF-IDF del conjunto de entrenamiento.
    :rtype: tuple[TfidfVectorizer, scipy.sparse.csr.csr_matrix]
    """

    if improve_filter:
        vector_path = os.path \
            .exists(ROOT_PATH + '{}'.format(TF_IDF_VECTOR_JSON_IMPROVED))
        vectors_path = os.path \
            .exists(ROOT_PATH + '{}'
                    .format(TF_IDF_VECTORS_JSON_IMPROVED))

        if vector_path and vectors_path:
            with open(ROOT_PATH + '{}'.format(TF_IDF_VECTORS_JSON_IMPROVED),
                      'rb') as f:
                vectors = pk.load(f)
            with open(ROOT_PATH + '{}'.format(TF_IDF_VECTOR_JSON_IMPROVED),
                      'rb') as f:
                vector = pk.load(f)
        else:
            vector, vectors = tf_idf_vector(improve_filter)
    else:
        vector_path = os.path.exists(
            ROOT_PATH + '{}'.format(TF_IDF_VECTOR_JSON))
        vectors_path = os.path.exists(
            ROOT_PATH + '{}'.format(TF_IDF_VECTORS_JSON))

        if vector_path and vectors_path:
            with open(ROOT_PATH + '{}'.format(TF_IDF_VECTORS_JSON), 'rb') as f:
                vectors = pk.load(f)
            with open(ROOT_PATH + '{}'.format(TF_IDF_VECTOR_JSON), 'rb') as f:
                vector = pk.load(f)
        else:
            vector, vectors = tf_idf_vector(improve_filter)

    return vector, vectors


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


def map_name_to_class(name):
    """
    Función que mapean los string del tipo 'no_spam_N', o 'spam_N', como
    'no_spam', o 'spam'.

    :param string name: Cade de texto que que sigue la expresión regular
        '_\d+'. Por ejemplo: 'no_spam_1678' o 'spam_2'.
    :return: Cadena de texto mapeada.
    :rtype: str
    """

    return re.sub(r'_\d+', '', name)

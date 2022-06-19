import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from python_project.utils.confusion_matrix import plot_confusion_matrix
from python_project.utils.reader import read_split_email_folder, read_email
from sklearn.metrics.pairwise import cosine_similarity
from python_project.tf_idf.tf_idf_knn import map_name_to_class
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


def classify_email(email_path, k=7, improve_filter=False):
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


def classify_emails(k=7, improve_filter=False):
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
    confusion_matrix_path = Path('python_project', 'tf_idf',
                                 'generated_documents')

    dic_class = classify_emails(k, improve_filter)

    true = [map_name_to_class(c) for c in dic_class.keys()]
    pred = [c for c in dic_class.values()]

    plot_confusion_matrix(true, pred, confusion_matrix_path, k, improve_filter)


def get_tf_idf_vector(improve_filter=False):
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
    return read_split_email_folder(train, improve_filter)


# no_spam_N --> no_spam // spam_N --> spam
def map_name_to_class(name):
    return re.sub(r'_\d+', '', name)


if __name__ == '__main__':
    # print(classify_email(ROOT_PATH + '\\python_project'
    #                                  '\\split_email_folder\\val'
    #                                  '\\legítimo\\1'))

    for k in range(1, 20):
        if k % 2 != 0:
            generate_confusion_matrix(k)

from sklearn.feature_extraction.text import TfidfVectorizer
from python_project.utils.confusion_matrix import plot_confusion_matrix
from python_project.utils.reader import read_split_email_folder, read_email
from sklearn.metrics.pairwise import cosine_similarity
from python_project.tf_idf.tf_idf_knn import map_name_to_class
from common_path import ROOT_PATH
from matplotlib import pyplot as plt
from pathlib import Path
from collections import Counter
import pickle as pk
import os


VOCABULARY_JSON = Path('python_project', 'utils', 'generated_documents',
                       'vocabulary.json')


def tf_idf_vector():
    corpus = read_split_email_folder()

    vocabulary_list = None

    with open(ROOT_PATH + '{}'.format(VOCABULARY_JSON), 'rb') as f:
        vocabulary_list = pk.load(f)

    vector = TfidfVectorizer(vocabulary=vocabulary_list,
                             stop_words='english',
                             lowercase=True)

    vectors = vector.fit_transform(corpus.values())

    return corpus, vector, vectors


def classify_email(email_path):
    k = 7

    email = read_email(email_path)

    corpus, vector, vectors = tf_idf_vector()

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


def classify_emails(k):
    dic = read_split_email_folder(train=False)

    corpus, vector, vectors = tf_idf_vector()

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


def generate_confusion_matrix(k):
    confusion_matrix_path = Path('python_project', 'tf_idf',
                                 'generated_documents')

    dic_class = classify_emails(k)

    true = [map_name_to_class(c) for c in dic_class.keys()]
    pred = [c for c in dic_class.values()]

    plot_confusion_matrix(true, pred, confusion_matrix_path, k)


if __name__ == '__main__':
    # print(classify_email(ROOT_PATH + '\\python_project'
    #                                  '\\split_email_folder\\val'
    #                                  '\\legítimo\\1'))

    for k in range(1, 20):
        if k % 2 != 0:
            generate_confusion_matrix(k)

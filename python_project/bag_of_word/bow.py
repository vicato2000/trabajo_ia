from sklearn.feature_extraction.text import CountVectorizer
from python_project.utils.reader import read_emails, read_email
from python_project.utils.vocabulary import generate_vocabulary
from common_path import ROOT_PATH
from pathlib import Path
import pickle as pk
import os
import numpy as np
from math import log
from python_project.utils.confusion_matrix import plot_confusion_matrix

VOCABULARY_JSON = Path('python_project', 'utils', 'generated_documents',
                       'vocabulary.json')
NO_SPAM_TRAIN_PATH = Path('python_project', 'split_email_folder', 'train',
                          'legítimo')
SPAM_TRAIN_PATH = Path('python_project', 'split_email_folder', 'train',
                       'no_deseado')
VAL_LEGITIMO = Path('python_project', 'split_email_folder', 'val', 'legítimo')

VAL_SPAM = Path('python_project', 'split_email_folder', 'val', 'no_deseado')


def train_bag_of_words(k=1):
    vocabulary_json = os.path.exists(ROOT_PATH + '{}'.format(VOCABULARY_JSON))

    vocabulary_list = None

    if vocabulary_json:
        with open(ROOT_PATH + '{}'.format(VOCABULARY_JSON), 'rb') as f:
            vocabulary_list = pk.load(f)
    else:
        vocabulary_list = generate_vocabulary()

    vectorizer_no_spam = CountVectorizer(stop_words='english',
                                         vocabulary=vocabulary_list,
                                         lowercase=True,
                                         analyzer='word')

    vectorizer_spam = CountVectorizer(stop_words='english',
                                      vocabulary=vocabulary_list,
                                      lowercase=True,
                                      analyzer='word')

    corpus_no_spam = read_emails(ROOT_PATH + '{}'.format(NO_SPAM_TRAIN_PATH))
    corpus_spam = read_emails(ROOT_PATH + '{}'.format(SPAM_TRAIN_PATH))

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

    vector_no_spam_softened_json = Path('python_project', 'bag_of_word',
                                        'generated_documents',
                                        'vector_no_spam_softened_k{}.json'
                                        .format(k))
    vector_spam_softened_json = Path('python_project', 'bag_of_word',
                                     'generated_documents',
                                     'vector_spam_softened_k{}.json'
                                     .format(k))

    with open(ROOT_PATH + '{}'.format(vector_no_spam_softened_json),
              'wb') as f:
        pk.dump(vector_no_spam_softened, f)

    with open(ROOT_PATH + '{}'.format(vector_spam_softened_json), 'wb') as f:
        pk.dump(vector_spam_softened, f)

    return vector_no_spam_softened, vector_spam_softened


def classify_email_bow(email_path):
    email = read_email(email_path)

    vector_spam_softened, vector_no_spam_softened = get_vectors_softened()

    vocabulary_list = get_vocabulary()

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


def classify_emails_bow(folder_path, k=1):
    emails = read_emails(folder_path)

    vector_spam_softened, vector_no_spam_softened = get_vectors_softened(k)

    vocabulary_list = get_vocabulary()

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


def get_vocabulary():
    vocabulary_json = os.path.exists(ROOT_PATH + '{}'.format(VOCABULARY_JSON))

    vocabulary_list = None

    if vocabulary_json:
        with open(ROOT_PATH + '{}'.format(VOCABULARY_JSON), 'rb') as f:
            vocabulary_list = pk.load(f)
    else:
        vocabulary_list = generate_vocabulary()

    sorted(vocabulary_list)

    return vocabulary_list


def get_vectors_softened(k=1):
    vector_no_spam_softened_json = Path('python_project', 'bag_of_word',
                                        'generated_documents',
                                        'vector_no_spam_softened_k{}.json'
                                        .format(k))
    vector_spam_softened_json = Path('python_project', 'bag_of_word',
                                     'generated_documents',
                                     'vector_spam_softened_k{}.json'
                                     .format(k))

    vector_no_spam_softened_json = os.path.exists(
        ROOT_PATH + '{}'.format(vector_no_spam_softened_json))

    vector_spam_softened_json = os.path.exists(
        ROOT_PATH + '{}'.format(vector_spam_softened_json))

    vector_spam_softened = None
    vector_no_spam_softened = None

    if vector_spam_softened_json and vector_no_spam_softened_json:
        with open(ROOT_PATH + '{}'.format(vector_no_spam_softened_json),
                  'rb') as f:
            vector_no_spam_softened = pk.load(f)

        with open(ROOT_PATH + '{}'.format(vector_spam_softened_json),
                  'rb') as f:
            vector_spam_softened = pk.load(f)
    else:
        vector_no_spam_softened, vector_spam_softened = train_bag_of_words()

    return vector_spam_softened, vector_no_spam_softened


def generate_confusion_matrix(k=1):
    confusion_matrix_path = Path('python_project', 'bag_of_word',
                                 'generated_documents')

    pred_spam = classify_emails_bow(ROOT_PATH + '{}'.format(VAL_SPAM), k)
    true_spam = ['spam' for i in range(len(pred_spam))]

    pred_no_spam = classify_emails_bow(ROOT_PATH + '{}'.format(VAL_LEGITIMO),
                                       k)
    true_no_spam = ['no_spam' for i in range(len(pred_no_spam))]

    pred = pred_spam + pred_no_spam
    true = true_spam + true_no_spam

    plot_confusion_matrix(true, pred, confusion_matrix_path, k)


if __name__ == '__main__':
    # print(classify_email_bow(ROOT_PATH + '\\python_project'
    #                                      '\\split_email_folder\\val'
    #                                      '\\legítimo\\1'))

    for k in range(1, 20):
        generate_confusion_matrix(k)

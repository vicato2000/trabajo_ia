import pickle

from sklearn.feature_extraction.text import CountVectorizer
from python_project.utils.reader import read_emails, read_email
from python_project.utils.vocabulary import generate_vocabulary
from common_path import ROOT_PATH
from pathlib import Path
import pickle as pk
import os
import numpy as np
from math import log

VOCABULARY_JSON = Path('python_project', 'utils', 'generated_documents',
                       'vocabulary.json')
NO_SPAM_TRAIN_PATH = Path('python_project', 'split_email_folder', 'train',
                          'legítimo')
SPAM_TRAIN_PATH = Path('python_project', 'split_email_folder', 'train',
                       'no_deseado')
VECTOR_NO_SPAM_SOFTENED_JSON = Path('python_project', 'bag_of_word',
                                    'generated_documents',
                                    'vector_no_spam_softened.json')
VECTOR_SPAM_SOFTENED_JSON = Path('python_project', 'bag_of_word',
                                 'generated_documents',
                                 'vector_spam_softened.json')


def train_bag_of_words():
    k = 1

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

    with open(ROOT_PATH + '{}'.format(VECTOR_NO_SPAM_SOFTENED_JSON),
              'wb') as f:
        pk.dump(vector_no_spam_softened, f)

    with open(ROOT_PATH + '{}'.format(VECTOR_SPAM_SOFTENED_JSON), 'wb') as f:
        pk.dump(vector_spam_softened, f)

    return vector_no_spam_softened, vector_spam_softened


def classify_email_bow(email_path):
    k = 1

    email = read_email(email_path)

    vector_no_spam_softened_json = os.path.exists(
        ROOT_PATH + '{}'.format(VECTOR_NO_SPAM_SOFTENED_JSON))

    vector_spam_softened_json = os.path.exists(
        ROOT_PATH + '{}'.format(VECTOR_SPAM_SOFTENED_JSON))

    vector_spam_softened = None
    vector_no_spam_softened = None

    if vector_spam_softened_json and vector_no_spam_softened_json:
        with open(ROOT_PATH + '{}'.format(VECTOR_NO_SPAM_SOFTENED_JSON),
                  'rb') as f:
            vector_no_spam_softened = pk.load(f)

        with open(ROOT_PATH + '{}'.format(VECTOR_SPAM_SOFTENED_JSON),
                  'rb') as f:
            vector_spam_softened = pk.load(f)
    else:
        vector_no_spam_softened, vector_spam_softened = train_bag_of_words()

    vocabulary_json = os.path.exists(ROOT_PATH + '{}'.format(VOCABULARY_JSON))

    vocabulary_list = None

    if vocabulary_json:
        with open(ROOT_PATH + '{}'.format(VOCABULARY_JSON), 'rb') as f:
            vocabulary_list = pk.load(f)
    else:
        vocabulary_list = generate_vocabulary()

    sorted(vocabulary_list)

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

    p_spam = log(total_train_spam/total) + sum(p_spam_list)

    p_no_spam_list = [new_vector[0][i] * log(vector_no_spam_softened[i])
                      for i in range(len(vector_no_spam_softened))]

    p_no_spam = log(total_train_no_spam/total) + sum(p_no_spam_list)

    return 'spam' if p_spam > p_no_spam else 'no_spam'


def count_different():
    count_no_spam = 0

    emails = os.listdir('../split_email_folder/val/legítimo')

    n_spam = len(emails)

    for e in emails:
        c = classify_email_bow('../split_email_folder/val/legítimo/{}'.format(str(e)))
        if c != 'no_spam':
            count_no_spam += 1
        print(n_spam)
        n_spam -= 1

    count_spam = 0

    emails_2 = os.listdir('../split_email_folder/val/no_deseado')

    spam = len(emails_2)

    for e in emails_2:
        c = classify_email_bow('../split_email_folder/val/no_deseado/{}'.format(str(e)))
        if c != 'spam':
            count_spam += 1
        print(spam)
        spam -= 1

    return count_no_spam ,count_spam

if __name__ == '__main__':
    # print(classify_email_bow(ROOT_PATH + '\\python_project'
    #                                      '\\split_email_folder\\val'
    #                                      '\\legítimo\\1'))

    print(count_different())
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.reader import read_emails, read_email
from utils.vocabulary import generate_vocabulary
from sklearn import neighbors
from common_path import ROOT_PATH
import pandas as pd
from matplotlib import pyplot


def training_tf_idf():
    corpus = get_corpus(ROOT_PATH + '/Enron-Spam/no_deseado',
                        ROOT_PATH + '/Enron-Spam/legítimo')

    vocabulary = generate_vocabulary()

    vector = TfidfVectorizer(vocabulary=vocabulary,
                             stop_words='english',
                             lowercase=True)

    vectors = vector.fit_transform(corpus)

    dense = vectors.todense()

    pyplot.show(dense)

    return vectors


def get_corpus(file_spam_email_path, file_no_spam_email_path):
    result = []

    for e in [file_spam_email_path, file_no_spam_email_path]:
        result.extend(read_emails(e))

    return result


def classify_email(email_path):
    pass


if __name__ == '__main__':
    training_tf_idf()

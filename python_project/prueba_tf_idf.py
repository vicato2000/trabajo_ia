from sklearn.feature_extraction.text import TfidfVectorizer
from python_project.utils.reader import read_emails, read_email
from python_project.utils.vocabulary import generate_vocabulary
from common_path import ROOT_PATH
import re
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

VOCABULARY = generate_vocabulary()
TF_IDF_VECTOR = TfidfVectorizer(vocabulary=VOCABULARY,
                                stop_words='english',
                                lowercase=True)


def dic_tf_idf():
    corpus = read_split_email_folder()

    TF_IDF_VECTOR.fit_transform(corpus)

    return {k: TF_IDF_VECTOR.transform([corpus[k]]).todense()
            for k in corpus.keys()}


def classify_email(email_path, k):
    tf_idf = dic_tf_idf()

    email = read_email(email_path)

    vector = TF_IDF_VECTOR.transform([email])

    similitary = {k: cosine_similarity(np.array(tf_idf[k]),
                                       vector.toarray())[0][0]
                  for k in tf_idf.keys()}

    sorted_similitary = sorted(similitary.items(), key=lambda x: x[1],
                               reverse=True)

    classes = map(map_name_to_class, [sorted_similitary[i][0]
                                      for i in range(0, k)])

    counter = Counter(classes)

    return counter.most_common()[0][0]


# no_spam_N --> no_spam // spam_N --> spam
def map_name_to_class(name):
    return re.sub(r'_\d+', '', name)


def read_split_email_folder():
    result = {}

    train_email_no_spam = read_emails(
        ROOT_PATH + '/python_project/split_email_folder/train/legítimo')

    result.update({'no_spam_{}'.format(e): train_email_no_spam[e] for e in
                   range(0, len(train_email_no_spam))})

    train_email_spam = read_emails(
        ROOT_PATH + '/python_project/split_email_folder/train/no_deseado')

    result.update({'spam_{}'.format(e): train_email_spam[e] for e in
                   range(0, len(train_email_spam))})

    return result


if __name__ == '__main__':
    val = classify_email('./split_email_folder/val/no_deseado/2', 7)

    print(val)

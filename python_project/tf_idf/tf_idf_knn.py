from sklearn.feature_extraction.text import TfidfVectorizer
from python_project.utils.reader import read_split_email_folder, read_email
from python_project.utils.vocabulary import generate_vocabulary
from sklearn.metrics.pairwise import cosine_similarity
import os.path
import re
import numpy as np
from collections import Counter
import pickle as pk
from common_path import ROOT_PATH


def generate_tf_idf_vector():
    vocabulary_json = os.path.exists(ROOT_PATH + '/python_project/utils'
                                                 '/generated_documents'
                                                 '/vocabulary.json')

    vocabulary_list = None

    if vocabulary_json:
        with open(ROOT_PATH + '/python_project/utils/generated_documents'
                              '/vocabulary.json', 'rb') as f:
            vocabulary_list = pk.load(f)
    else:
        vocabulary_list = generate_vocabulary()

    tf_idf_vector = TfidfVectorizer(vocabulary=vocabulary_list,
                                    stop_words='english',
                                    lowercase=True)

    with open(ROOT_PATH + '/python_project/tf_idf/generated_documents'
                          '/tf_idf_vector.json', 'wb') as f:
        pk.dump(tf_idf_vector, f)

    return tf_idf_vector


def dic_tf_idf():
    result = {}

    tf_idf_vector_json = os.path.exists(ROOT_PATH + '/python_project/tf_idf'
                                                    '/generated_documents'
                                                    '/tf_idf_vector.json')

    tf_idf_vector = None

    if tf_idf_vector_json:
        with open(ROOT_PATH + '/python_project/tf_idf/generated_documents'
                              '/tf_idf_vector.json', 'rb') as f:
            tf_idf_vector = pk.load(f)
    else:
        tf_idf_vector = generate_tf_idf_vector()

    corpus = read_split_email_folder()

    tf_idf_vector.fit_transform(corpus.values())

    with open(ROOT_PATH + '/python_project/tf_idf/generated_documents'
                          '/tf_idf_vector_training.json', 'wb') as f:
        pk.dump(tf_idf_vector, f)

    result.update({k: tf_idf_vector.transform([corpus[k]]).todense()
                   for k in corpus.keys()})

    with open(ROOT_PATH + '/python_project/tf_idf/generated_documents'
                          '/dic_tf_idf.json', 'wb') as f:
        pk.dump(result, f)

    return result, tf_idf_vector


def classify_email(email_path, k):
    dic_tf_idf_json = os.path.exists(ROOT_PATH + '/python_project/tf_idf'
                                                 '/generated_documents'
                                                 '/dic_tf_idf.json')

    tf_idf_vector_training_json = os.path.exists(
        ROOT_PATH + '/python_project/tf_idf/generated_documents'
                    '/tf_idf_vector_training.json')

    tf_idf = None
    tf_idf_vector = None

    if dic_tf_idf_json and tf_idf_vector_training_json:
        with open(ROOT_PATH + '/python_project/tf_idf/generated_documents'
                              '/dic_tf_idf.json', 'rb') as f:
            tf_idf = pk.load(f)

        with open(ROOT_PATH + '/python_project/tf_idf/generated_documents'
                              '/tf_idf_vector_training.json', 'rb') as f:
            tf_idf_vector = pk.load(f)
    else:
        tf_idf, tf_idf_vector = dic_tf_idf()

    email = read_email(email_path)

    vector = tf_idf_vector.transform([email])

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


if __name__ == '__main__':
    val = classify_email(ROOT_PATH + '/python_project/split_email_folder/val'
                                     '/no_deseado/472', 7)

    print(val)

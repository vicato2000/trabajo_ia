from sklearn.feature_extraction.text import TfidfVectorizer
from utils.reader import read_emails, read_email
from utils.vocabulary import generate_vocabulary
from sklearn import neighbors


def training_tf_idf(file_spam_email_path, file_no_spam_email_path, file_email):
    corpus = get_corpus(file_spam_email_path, file_no_spam_email_path)

    vocabulary = generate_vocabulary(file_email)

    vectorizer = TfidfVectorizer(vocabulary=vocabulary, stop_words='english',
                                 lowercase=True)
    x_vector = vectorizer.fit_transform(corpus)

    return x_vector


def get_corpus(file_spam_email_path, file_no_spam_email_path):
    result = []

    for e in [file_spam_email_path, file_no_spam_email_path]:
        result.extend(read_emails(e))

    return result


def classify_email(email_in, vector_tf_idf, file_email):

    vocabulary = generate_vocabulary(file_email)

    clasif_kNN = neighbors.KNeighborsClassifier(n_neighbors=5)
    clasif_kNN.fit(vector_tf_idf, vocabulary)

    vector_to_classify = training_tf_idf.transform(read_email(email_in))

    print(clasif_kNN.predict(vector_to_classify))

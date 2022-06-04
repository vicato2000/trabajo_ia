from nltk.lm.vocabulary import Vocabulary
from nltk.corpus import stopwords
import itertools
from nltk import download
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.data import load


def extract_vocabulary(emails_list):
    emails = [e for i, e in emails_list]

    for e in emails:
        e.rstrip()



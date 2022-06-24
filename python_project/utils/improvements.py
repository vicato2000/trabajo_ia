import unicodedata
import nltk
import inflect
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import contractions

nltk.download('punkt', download_dir='.')
nltk.download('stopwords', download_dir='.')


# Eliminación de ruido

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def replace_contractions(text):
    """
    Función que reemplaza las contracciones del inglés.

    :param str text: Cadena de texto de entrada.
    :return: La cadena de texto sin contracciones.
    :rtype: str
    """
    return contractions.fix(text)


def denoise_text(text):
    text = strip_html(text)
    text = replace_contractions(text)
    text = remove_between_square_brackets(text)
    return text


# Tokenización

def tokenize(text):
    return nltk.word_tokenize(text)


# Normalización

def remove_non_ascii(words):

    """
    Función que elimina los caracteres no ASCII.

    :param list[str] words: Lista de palabras tokenizadas.
    :return: Lista de palabras tokenizadas sin caracteres no ASCII.
    :rtype: list[str]
    """

    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word) \
            .encode('ascii', 'ignore') \
            .decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):

    """
    Funcion que convierte todos los caracteres a minuscula de una lista de
    palabras tokenizadas.

    :param list[str] words: Lista de palabras tokenizadas.
    :return: Lista de palabras tokenizadas con todos sus caracteres en
    minúscula.
    :rtype: list[str]
    """

    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):

    """
    Función que elimina los signos de puntuación de una lista de palabras
    tokenizadas.

    :param list[str] words: Lista de palabras tokenizadas.
    :return: Lista de palabras tokenizadas sin signos de puntuación.
    :rtype: list[str]
    """

    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):

    """
    Función que reemplaza las ocurrencias de números enteros, de no más de
    20 dígitos, de una lista de palabras tokenizadas en su representación
    textual.

    :param list[str] words: Lista de palabras tokenizadas.
    :return: Lista de palabras tokenizadas.
    :rtype: list[str]
    """

    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit() and len(word) <= 20:
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):

    """
    Función que elimina las 'stop words' de una lista de palabras tokenizadas.

    :param list[str] words: Lista de palabras tokenizadas.
    :return: Lista de palabras tokenizadas sin las 'stop words'.
    :rtype: list[str]
    """

    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):

    """
    Función que radicaliza (stemming) las palabras de una lista tokenizada.

    :param list[str] words: Lista de palabras tokenizadas.
    :return: Lista de palabras tokenizadas.
    :rtype: list[str]
    """

    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """
       Función que lematiza los verbos de una lista tokenizada.

       :param list[str] words: Lista de palabras tokenizadas.
       :return: Lista de palabras tokenizadas.
       :rtype: list[str]
       """

    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


def improve(text):
    result = denoise_text(text)
    result = tokenize(result)
    result = normalize(result)

    return ' '.join(result)

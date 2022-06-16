from bs4 import BeautifulSoup
import re
import contractions


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def denoise_text(text):
    text = strip_html(text)
    text = replace_contractions(text)
    text = remove_between_square_brackets(text)
    return text

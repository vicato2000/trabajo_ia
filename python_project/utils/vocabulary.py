from python_project.utils.clean_email import denoise_text
from python_project.utils.reader import read_emails
import re
import os
from common_path import ROOT_PATH
import pickle as pk
from pathlib import Path

ENRON_SPAM_LEGITIMO_PATH = Path('Enron-Spam', 'legítimo')
READ_EMAILS_LEGITIMO_TXT = Path('python_project', 'utils',
                                'generated_documents',
                                'read_emails_legítimo.txt')
ENRON_SPAM_NO_DESEADO_PATH = Path('Enron-Spam', 'no_deseado')
READ_EMAILS_NO_DESEADO_TXT = Path('python_project', 'utils',
                                  'generated_documents',
                                  'read_emails_no_deseado.txt')
VOCABULARY_JSON = Path('python_project', 'utils', 'generated_documents',
                       'vocabulary.json')


def generate_vocabulary():
    result = set()

    email_vocabulary(ROOT_PATH + '{}'.format(ENRON_SPAM_LEGITIMO_PATH),
                     ROOT_PATH + '{}'.format(READ_EMAILS_LEGITIMO_TXT))

    email_vocabulary(ROOT_PATH + '{}'.format(ENRON_SPAM_NO_DESEADO_PATH),
                     ROOT_PATH + '{}'.format(READ_EMAILS_NO_DESEADO_TXT))

    vocabulary = set()

    for file_path in [ROOT_PATH + '{}'.format(READ_EMAILS_LEGITIMO_TXT),
                      ROOT_PATH + '{}'.format(READ_EMAILS_NO_DESEADO_TXT)]:
        with open(file_path,
                  mode='r',
                  encoding='utf-8') as line:
            vocabulary.update(re.split(' |/|, |. ', line.read()))

    for e in vocabulary.copy():
        if bool(re.search(r'\W|\d|-|_', e)):
            vocabulary.remove(e)

    result.update(set(v.lower() for v in vocabulary))

    with open(ROOT_PATH + '{}'.format(VOCABULARY_JSON), 'wb') as f:
        pk.dump(result, f)

    return result


def email_vocabulary(file_email_in, file_out):
    email_list = read_emails(file_email_in)

    file = ''

    for i in range(0, len(email_list) - 1):
        file += '{}'.format(denoise_text('{}'.format(email_list[i]))) + '\n'

    if os.path.exists(file_out):
        os.remove(file_out)

    with open(file_out,
              mode='w',
              encoding='utf-8') as f:

        f.write(file)

from python_project.utils.clean_email import denoise_text
from python_project.utils.reader import read_emails
import re
import os


class FileEmail:
    def __init__(self, file_legitimate_email_in, file_legitimate_out,
                 file_spam_email_in, file_spam_out):
        self.legitimate_email_vocabulary_in = file_legitimate_email_in
        self.legitimate_email_vocabulary_out = file_legitimate_out
        self.spam_email_vocabulary_in = file_spam_email_in
        self.spam_email_vocabulary_out = file_spam_out


def generate_vocabulary(file_email):
    email_vocabulary(file_email.legitimate_email_vocabulary_in,
                     file_email.legitimate_email_vocabulary_out)

    email_vocabulary(file_email.spam_email_vocabulary_in,
                     file_email.spam_email_vocabulary_out)

    vocabulary = set()

    for file_path in [file_email.legitimate_email_vocabulary_out,
                      file_email.spam_email_vocabulary_out]:
        with open(file_path,
                  mode='r',
                  encoding='utf-8') as line:
            vocabulary.update(re.split(' |/|, |. ', line.read()))

    for e in vocabulary.copy():
        if bool(re.search(r'\W|\d|-|_', e)):
            vocabulary.remove(e)

    return set(v.lower() for v in vocabulary)


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

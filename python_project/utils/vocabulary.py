from python_project.utils.clean_email import denoise_text
from python_project.utils.reader import read_emails
import re
import os
from common_path import ROOT_PATH
import pickle as pk


def generate_vocabulary():

    result = set()

    email_vocabulary(ROOT_PATH + '/Enron-Spam/legítimo',
                     ROOT_PATH + '/python_project/utils/generated_documents'
                                 '/read_emails_legítimo.txt')

    email_vocabulary(ROOT_PATH + '/Enron-Spam/no_deseado',
                     ROOT_PATH + '/python_project/utils/generated_documents'
                                 '/read_emails_no_deseado.txt')

    vocabulary = set()

    for file_path in [ROOT_PATH + '/python_project/utils/generated_documents'
                                  '/read_emails_legítimo.txt',
                      ROOT_PATH + '/python_project/utils/generated_documents'
                                  '/read_emails_no_deseado.txt']:
        with open(file_path,
                  mode='r',
                  encoding='utf-8') as line:
            vocabulary.update(re.split(' |/|, |. ', line.read()))

    for e in vocabulary.copy():
        if bool(re.search(r'\W|\d|-|_', e)):
            vocabulary.remove(e)

    result.update(set(v.lower() for v in vocabulary))

    with open(ROOT_PATH + '/python_project/utils/generated_documents'
                          '/vocabulary.json', 'wb') as f:
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


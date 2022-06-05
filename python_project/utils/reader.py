from email.parser import Parser
from email import policy
import email as em
import os
import json


def read_email(email_path):
    with open(email_path, 'r', encoding='latin1') as email:
        msg = em.message_from_string(email.read(), policy=policy.default)

        if msg.is_multipart():
            return msg.get_body()
        else:
            return msg.get_payload()


def read_emails(emails_folder_path):
    folder_files_list = [int(file) for file in os.listdir(emails_folder_path)]
    folder_files_list.sort()

    return [read_email('{}/{}'.format(emails_folder_path, e)) for e in folder_files_list]


if __name__ == '__main__':
    email_list = read_emails('/home/vicato/IA/trabajo_ia/Enron-Spam/no_deseado')

    file = ''

    for i in range(0, len(email_list) - 1):
        file += '{}'.format(email_list[i]) + '\n'

    f = open('../documentos_prueba/read_emails_prueba.txt', 'w')
    f.write(file)

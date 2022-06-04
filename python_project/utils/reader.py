from email.parser import Parser
from email.policy import default
import os


def read_email(email_path):
    email = open(email_path, 'r')

    headers = Parser(policy=default).parsestr(email.read())

    return 'body-{}'.format(email_path.split('/')[-1]), headers.get_payload()


def read_emails(emails_folder_path):
    folder_files_list = [int(file) for file in os.listdir(emails_folder_path)]
    folder_files_list.sort()

    print(folder_files_list)

    return [read_email('{}/{}'.format(emails_folder_path, e)) for e in folder_files_list]


if __name__ == '__main__':
    email_list = read_emails('/home/vicato/IA/trabajo_ia/Enron-Spam/legítimo')

    file = ''

    for i in range(0, len(email_list) - 1):
        file += email_list[i][1] + '\n'
        file += '---------------------------------------------------------------------\n'

    f = open('../documentos_prueba/read_emails_prueba.txt', 'w')
    f.write(file)

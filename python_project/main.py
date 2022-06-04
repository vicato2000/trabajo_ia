from prueba_seleccion_vocabulario import *
from utils.reader import *

if __name__ == '__main__':
    email_list = read_emails('/home/vicato/IA/trabajo_ia/Enron-Spam/legítimo')

    # file = ''
    #
    # for i in range(0, len(email_list) - 1):
    #     file += email_list[i][1] + '\n'
    #     file += '---------------------------------------------------------------------\n'
    #
    # f = open('../documentos_prueba/read_emails_prueba.txt', 'w')
    # f.write(file)

    extract_vocabulary(email_list)
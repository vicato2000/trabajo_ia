from prueba_seleccion_vocabulario import *
from utils.reader import *
from utils.clean_code import *
import os

if __name__ == '__main__':

    email_list = read_emails('/home/vicato/IA/trabajo_ia/Enron-Spam/no_deseado')

    file = ''

    for i in range(0, len(email_list) - 1):
        file += '{}'.format(denoise_text('{}'.format(email_list[i]))) + '\n'

    f = open('/home/vicato/IA/trabajo_ia/python_project/documentos_prueba/read_emails_prueba.txt', 'w')
    f.write(file)

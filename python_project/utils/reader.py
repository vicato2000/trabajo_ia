from email.policy import default
import email as em
import os
from common_path import ROOT_PATH
from pathlib import Path
import pickle as pk
from python_project.utils.improvements import improve

TRAIN_LEGITIMO = Path('python_project', 'split_email_folder', 'train',
                      'legítimo')

TRAIN_SPAM = Path('python_project', 'split_email_folder', 'train',
                  'no_deseado')

VAL_LEGITIMO = Path('python_project', 'split_email_folder', 'val', 'legítimo')

VAL_SPAM = Path('python_project', 'split_email_folder', 'val', 'no_deseado')

DIC_TRAIN_EMAIL_PATH = Path('python_project', 'utils', 'generated_documents',
                            'dic_train_email.json')

DICT_VAL_EMAIL_PATH = Path('python_project', 'utils', 'generated_documents',
                           'dic_val_email.json')

DIC_TRAIN_EMAIL_IMPROVE_PATH = Path('python_project', 'utils',
                                    'generated_documents',
                                    'dic_train_email_improve.json')

DICT_VAL_EMAIL_IMPROVE_PATH = Path('python_project', 'utils',
                                   'generated_documents',
                                   'dic_val_email_improve.json')


def read_email(email_path):

    """
    Función que lee el cuerpo.

    :param str email_path: Ruta del email.
    :return: Cadena de texto con el cuerpo del email, en el caso de ser
        multipar devuelve todas las partes concatenadas.
    :rtype: str
    """

    with open(email_path, 'r', encoding='utf-8', errors='ignore') as email:
        msg = em.message_from_string(email.read(), policy=default)

        def _get_body_multipart(emailobj):
            body = ''

            if emailobj.is_multipart():
                for e in emailobj.get_payload():
                    body += _get_body_multipart(e)
                    body += '\n'
            else:
                body += emailobj.get_payload()
                body += '\n'

            return body

        return '{}'.format(_get_body_multipart(msg))


def read_emails(emails_folder_path):

    """
    Función que lee todos los cuerpos de los emails que se encuentran en una
    carpeta.

    :param str emails_folder_path: Ruta de la carpeta donde se encuentran los
        correos que se quieren leer.
    :return: Lista compuesta por los cuerpos de los mensajes.
    :rtype: list[str]
    """

    folder_files_list = [int(file) for file in os.listdir(emails_folder_path)]
    folder_files_list.sort()

    return [read_email('{}'.format(os.path.join(emails_folder_path, str(e))))
            for e in folder_files_list]


def read_split_email_folder(path_legitimo, path_spam):

    """
    Función que lee cuerpos de los emails de dos carpetas. Se usa para leer
    todos los emails de entrenamiento o de los de prueba.

    :param str path_legitimo: Ruta de la carpeta donde se encuentren los emails
        legítimos.
    :param str path_spam: Ruta de la carpeta donde se encuentren los emails
        no_deseados.
    :return: Diccionario donde la clave es la clase correcta a la que
        pertenece ese email, spam o no_spam, y el valor es el cuerpo del email.
    :rtype: dict[str,str]
    """

    result = {}

    email_no_spam = read_emails(path_legitimo)
    result.update({'no_spam_{}'.format(e): email_no_spam[e] for e in
                   range(0, len(email_no_spam))})

    email_spam = read_emails(path_spam)
    result.update({'spam_{}'.format(e): email_spam[e] for e in
                   range(0, len(email_spam))})

    return result


def get_dic_train_email(train=True, improve_filter=False):

    """
    Función que genera o lee los cuerpos de los emails de entrenamiento o de
    prueba dependiendo de si existen previamente su archivo json o no.

    :param bool train: Indica si tiene que leer la carpeta de entrenamiento o
        la carpeta de prueba. Por defecto se lee la de entrenamiento.
    :param bool improve_filter: Usar o no técnicas de mejoras. Por defecto se
        encuentra a False.
    :return: Diccionario donde la clave es la clase correcta a la que
        pertenece ese email, spam o no_spam, y el valor es el cuerpo del email.
    :rtype: dict[str,str]
    """

    if improve_filter:
        if train:
            path_legitimo = ROOT_PATH + '{}'.format(TRAIN_LEGITIMO)
            path_spam = ROOT_PATH + '{}'.format(TRAIN_SPAM)

            dic_improved_path = ROOT_PATH + '{}'.format(
                DIC_TRAIN_EMAIL_IMPROVE_PATH)
        else:
            path_legitimo = ROOT_PATH + '{}'.format(VAL_LEGITIMO)
            path_spam = ROOT_PATH + '{}'.format(VAL_SPAM)

            dic_improved_path = ROOT_PATH + '{}'.format(
                DICT_VAL_EMAIL_IMPROVE_PATH)

        if os.path.exists(dic_improved_path):
            resul = {}

            with open(dic_improved_path, 'rb') as f:
                result = pk.load(f)
        else:
            dic_email = read_split_email_folder(path_legitimo, path_spam)

            result = {k: improve(dic_email[k]) for k in dic_email.keys()}

            with open(dic_improved_path, 'wb') as f:
                pk.dump(result, f)

    else:
        if train:
            path_legitimo = ROOT_PATH + '{}'.format(TRAIN_LEGITIMO)
            path_spam = ROOT_PATH + '{}'.format(TRAIN_SPAM)

            dic_improved_path = ROOT_PATH + '{}'.format(
                DIC_TRAIN_EMAIL_PATH)
        else:
            path_legitimo = ROOT_PATH + '{}'.format(VAL_LEGITIMO)
            path_spam = ROOT_PATH + '{}'.format(VAL_SPAM)

            dic_improved_path = ROOT_PATH + '{}'.format(
                DICT_VAL_EMAIL_PATH)

        if os.path.exists(dic_improved_path):
            resul = {}

            with open(dic_improved_path, 'rb') as f:
                result = pk.load(f)
        else:
            result = read_split_email_folder(path_legitimo, path_spam)

            with open(dic_improved_path, 'wb') as f:
                pk.dump(result, f)

    return result

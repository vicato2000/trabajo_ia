from email.policy import default
import email as em
import os
from common_path import ROOT_PATH
from pathlib import Path

TRAIN_LEGITIMO = Path('python_project', 'split_email_folder', 'train',
                      'legítimo')

TRAIN_SPAM = Path('python_project', 'split_email_folder', 'train',
                  'no_deseado')

VAL_LEGITIMO = Path('python_project', 'split_email_folder', 'val', 'legítimo')

VAL_SPAM = Path('python_project', 'split_email_folder', 'val', 'no_deseado')


def read_email(email_path):
    with open(email_path, 'r', encoding='utf-8', errors='ignore') as email:
        msg = em.message_from_string(email.read(), policy=default)

        if msg.is_multipart():

            msg_multipart = em.message_from_string('{}'.format(
                msg.get_payload()))

            return msg_multipart.get_payload()[0]
        else:
            return msg.get_payload()


def read_emails(emails_folder_path):
    folder_files_list = [int(file) for file in os.listdir(emails_folder_path)]
    folder_files_list.sort()

    return [read_email('{}'.format(os.path.join(emails_folder_path, str(e))))
            for e in folder_files_list]


def read_split_email_folder(train=None):
    result = {}

    path_legitimo = '{}'.format(TRAIN_LEGITIMO) \
        if train or train is None else '{}'.format(VAL_LEGITIMO)

    path_spam = '{}'.format(TRAIN_SPAM) \
        if train or train is None else '{}'.format(VAL_SPAM)

    email_no_spam = read_emails(ROOT_PATH + path_legitimo)

    result.update({'no_spam_{}'.format(e): email_no_spam[e] for e in
                   range(0, len(email_no_spam))})

    email_spam = read_emails(ROOT_PATH + path_spam)

    result.update({'spam_{}'.format(e): email_spam[e] for e in
                   range(0, len(email_spam))})

    return result

from email.policy import default
import email as em
import os
from common_path import ROOT_PATH


def read_email(email_path):
    with open(email_path, 'r', encoding='latin1') as email:
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

    return [read_email('{}/{}'.format(emails_folder_path, e))
            for e in folder_files_list]


def read_split_email_folder():
    result = {}

    train_email_no_spam = read_emails(
        ROOT_PATH + '/python_project/split_email_folder/train/legítimo')

    result.update({'no_spam_{}'.format(e): train_email_no_spam[e] for e in
                   range(0, len(train_email_no_spam))})

    train_email_spam = read_emails(
        ROOT_PATH + '/python_project/split_email_folder/train/no_deseado')

    result.update({'spam_{}'.format(e): train_email_spam[e] for e in
                   range(0, len(train_email_spam))})



    return result


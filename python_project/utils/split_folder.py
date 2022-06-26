import splitfolders
from common_path import ROOT_PATH
from pathlib import Path

ENRON_SPAM_PATH = Path('Enron-Spam')
SPLIT_EMAIL_FOLDER_PATH = Path('python_project', 'split_email_folder')


def split_enron_spam_folder():
    path_in = ROOT_PATH + '{}'.format(ENRON_SPAM_PATH)
    path_out = ROOT_PATH + '{}'.format(SPLIT_EMAIL_FOLDER_PATH)

    splitfolders.ratio(path_in, path_out, ratio=(0.8, 0.2))

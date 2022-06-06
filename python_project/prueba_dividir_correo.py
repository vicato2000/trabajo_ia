import splitfolders
import os


def split_email_folder(folder_path_in, folder_path_out):
    if os.path.exists(folder_path_out):
        os.rmdir(folder_path_out)
        os.mkdir(folder_path_out)

    splitfolders.ratio(folder_path_in, output=folder_path_out,
                       seed=1337, ratio=(.8, .2), group_prefix=None,
                       move=False)


if __name__ == '__main__':
    split_email_folder('../Enron-Spam/', './split_email_folder/')

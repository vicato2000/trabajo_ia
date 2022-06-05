from python_project.utils.clean_email import denoise_text
from python_project.utils.reader import read_emails


def generate_vocabulary():
    # TODO
    pass


def spam_vocabulary():
    email_list = read_emails('../Enron-Spam/no_deseado')

    file = ''

    for i in range(0, len(email_list) - 1):
        file += '{}'.format(denoise_text('{}'.format(email_list[i]))) + '\n'

    f = open('generated_documents/read_emails_no_deseado.txt', 'w')
    f.write(file)


def no_spam_vocabulary():
    email_list = read_emails('../Enron-Spam/legítimo')

    file = ''

    for i in range(0, len(email_list) - 1):
        file += '{}'.format(denoise_text('{}'.format(email_list[i]))) + '\n'

    f = open('generated_documents/read_emails_legítimo.txt', 'w')
    f.write(file)

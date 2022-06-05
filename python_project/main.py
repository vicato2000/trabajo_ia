from python_project.utils.vocabulary import generate_vocabulary, FileEmail

if __name__ == '__main__':
    file_email = FileEmail(file_spam_email_in='../Enron-Spam/no_deseado',
                           file_spam_out='./utils/generated_documents/'
                                         'read_emails_no_deseado.txt',
                           file_legitimate_email_in='../Enron-Spam/legítimo',
                           file_legitimate_out='./utils/generated_documents/'
                                               'read_emails_legitimos.txt')

    vocabulary = generate_vocabulary(file_email)

    print(vocabulary)

from utils.vocabulary import generate_vocabulary, FileEmail
from prueba_tf_idf import training_tf_idf, classify_email

if __name__ == '__main__':
    file_email = FileEmail(file_spam_email_in='../Enron-Spam/no_deseado',
                           file_spam_out='./utils/generated_documents/'
                                         'read_emails_no_deseado.txt',
                           file_legitimate_email_in='../Enron-Spam/legítimo',
                           file_legitimate_out='./utils/generated_documents/'
                                               'read_emails_legitimos.txt')

    vector_tf_idf = training_tf_idf('./split_email_folder/train/no_deseado',
                                    './split_email_folder/train/legítimo',
                                    file_email)

    classify_email('./split_email_folder/val/no_deseado/2',
                   vector_tf_idf,
                   file_email)

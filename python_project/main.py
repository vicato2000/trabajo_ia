from prueba_tf_idf import training_tf_idf, classify_email

if __name__ == '__main__':

    vector_tf_idf = training_tf_idf('./split_email_folder/train/no_deseado',
                                    './split_email_folder/train/legítimo',
                                    file_email)

    classify_email('./split_email_folder/val/no_deseado/2',
                   vector_tf_idf,
                   file_email)

from python_project.tf_idf.tf_idf_knn import classify_email


def es_mensaje_no_deseado(email_path):
    result = classify_email(email_path, k=1, improve_filter=False)

    return True if result == 'spam' else False

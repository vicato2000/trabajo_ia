from email.parser import BytesParser, Parser
from email.policy import default
from utils import reader


def read_email(email_path):
    email = open(email_path, 'r')

    headers = Parser(policy=default).parsestr(email.read())

    return {'to': headers['to'],
            'from': headers['from'],
            'subject': headers['subject'],
            'body': headers.get_payload()}


if __name__ == '__main__':
    print(reader.read_email('/home/vicato/IA/trabajo_ia/Enron-Spam/no_deseado/7'))

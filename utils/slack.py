import requests
from socket import gethostbyname, gethostname


def send_info_to_slack(msg, webhook_url):
    """
    Send slack info message

    :param msg: message
    :return: True or False
    """

    data = {
        'attachments': [
            {
                'pretext': 'IP: {}\nInfo message:'.format(
                    gethostbyname(gethostname())),
                'color': '#0000FF',
                'text': msg,
            }
        ]
    }
    return send_msg_to_slack(data, webhook_url)


def send_error_to_slack(msg, webhook_url):
    """
    Send slack error message

    :param msg: message
    :return: True or False
    """
    data = {
        'attachments': [
            {
                'pretext': 'IP: {}\nError message:'.format(
                    gethostbyname(gethostname())),
                'color': '#FF0000',
                'text': msg,
            }
        ]
    }
    return send_msg_to_slack(data, webhook_url)


def send_msg_to_slack(data, webhook_url):
    """
    Send slack message

    :param data: slack message format, dict type
    :return: True or False
    """

    requests.post(webhook_url, json=data)

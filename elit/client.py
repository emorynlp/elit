# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 20:00
import json
from typing import Any, Dict

from elit.common.document import Document
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Client(object):

    def __init__(self, url: str, auth: str = None) -> None:
        super().__init__()
        self.url = url
        self.auth = auth

    def parse(self, text: str) -> Document:
        response = self._send_get_json(self.url + '/parse', {'text': text})
        return Document(response)

    def _send_post(self, url, form: Dict[str, Any]):
        request = Request(url, urlencode(form).encode())
        return urlopen(request).read().decode()

    def _send_post_json(self, url, form: Dict[str, Any]):
        return json.loads(self._send_post(url, form))

    def _send_get(self, url, form: Dict[str, Any]):
        request = Request(url + '?' + urlencode(form))
        return urlopen(request).read().decode()

    def _send_get_json(self, url, form: Dict[str, Any]):
        return json.loads(self._send_get(url, form))


def main():
    text = "Jobs and Wozniak co-founded Apple in 1976 to sell Wozniak's Apple I personal computer. " \
           "Together the duo gained fame and wealth a year later with the Apple II. "
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(text))


if __name__ == '__main__':
    main()
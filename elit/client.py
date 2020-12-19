import json
from typing import Any, Dict, List, Union
from elit.common.document import Document
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Client(object):

    def __init__(self, url: str, auth: str = None) -> None:
        super().__init__()
        self.url = url
        self.auth = auth

    def parse(self,
              text: Union[str, List[str]] = None,
              tokens: List[List[str]] = None,
              models=("lem", "pos", "ner", "con", "dep", "srl", "amr"),
              language='en',
              verbose=True,
              ) -> Document:
        assert text or tokens, 'At least one of text or tokens has to be specified.'
        response = self._send_post_json(self.url + '/parse', {
            'text': text,
            'tokens': tokens,
            'models': models,
            'language': language,
            'verbose': verbose
        })
        return Document(response)

    def _send_post(self, url, form: Dict[str, Any]):
        request = Request(url, json.dumps(form).encode())
        return urlopen(request).read().decode()

    def _send_post_json(self, url, form: Dict[str, Any]):
        return json.loads(self._send_post(url, form))

    def _send_get(self, url, form: Dict[str, Any]):
        request = Request(url + '?' + urlencode(form))
        return urlopen(request).read().decode()

    def _send_get_json(self, url, form: Dict[str, Any]):
        return json.loads(self._send_get(url, form))


def main():
    # _test_raw_text()
    # _test_sents()
    _test_tokens()


def _test_raw_text():
    text = "Emory NLP is a research lab in Atlanta, GA. " \
           "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(text))


def _test_sents():
    text = ["Emory NLP is a research lab in Atlanta, GA. ",
            "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."]
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(text))


def _test_tokens():
    tokens = [
        ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
        ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", ".", "Dr.", "Choi", "is", "a", "professor",
         "at", "Emory", "University", "."]
    ]
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(tokens=tokens, models=['ner', 'srl', 'dep'], verbose=True))


if __name__ == '__main__':
    main()

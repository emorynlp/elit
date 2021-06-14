import json
from typing import Any, Dict, List, Union, Optional
from elit.common.document import Document
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class Client(object):
    coref_context_keys = ('input_ids', 'sentence_map', 'subtoken_map', 'mentions', 'uttr_start_idx', 'speaker_ids')

    def __init__(self, url: str, auth: str = None) -> None:
        super().__init__()
        self.url = url
        self.auth = auth

    @classmethod
    def create_coref_context_from_online_output(cls, coref_output: dict) -> Optional[dict]:
        """ Create context (from previous online output) to online coreference input with 1+ turns """
        if coref_output is None:
            return None
        return {k: coref_output[k] for k in cls.coref_context_keys}

    def parse(self,
              text: Union[str, List[str]] = None,
              tokens: List[List[str]] = None,
              models=("lem", "pos", "ner", "con", "dep", "srl", "amr", "dcr", "ocr"),
              speaker_ids: Union[int, List[int]] = None,
              genre: str = None,
              coref_context: dict = None,
              return_coref_prob: bool = False,
              language='en',
              verbose=True,
              ) -> Document:
        assert text or tokens, 'At least one of text or tokens has to be specified.'
        response = self._send_post_json(self.url + '/parse', {
            'text': text,
            'tokens': tokens,
            'models': models,
            'speaker_ids': speaker_ids,
            'genre': genre,
            'coref_context': coref_context,
            'return_coref_prob': return_coref_prob,
            'language': language,
            'verbose': verbose
        })
        return Document(response)

    def _send_post(self, url, form: Dict[str, Any]):
        request = Request(url, json.dumps(form).encode())
        request.add_header('Content-Type', 'application/json')
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


def _test_doc_coref():
    # Can be either raw text, sents, or tokens
    text = 'Pfizer said last week it may need the U.S. government to help it secure some components needed to ' \
           'make the vaccine. While the company halved its 2020 production target due to manufacturing issues, ' \
           'it said last week its manufacturing is running smoothly now. The government also has the option to ' \
           'acquire up to an additional 400 million doses of the vaccine.'
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(text=text, models=['dcr']))


def _test_online_coref():
    # Can be either raw text, sents, or tokens
    nlp = Client('http://0.0.0.0:8000')
    doc = nlp.parse(text='I read an article today. It is about US politics.',
                    speaker_ids=1, coref_context=None, models=['ocr'])
    print(doc)

    context = nlp.create_coref_context_from_online_output(doc['ocr'])
    doc = nlp.parse(text='What does it say about US politics?',
                    speaker_ids=2, coref_context=context, models=['ocr'])
    print(doc)

    context = nlp.create_coref_context_from_online_output(doc['ocr'])
    doc = nlp.parse(text='It talks about the US presidential election.',
                    speaker_ids=1, coref_context=context, models=['ocr'])
    print(doc)

    context = nlp.create_coref_context_from_online_output(doc['ocr'])
    doc = nlp.parse(text='The presidential election is indeed interesting.',
                    speaker_ids=2, coref_context=context, models=['ocr'])
    print(doc)


if __name__ == '__main__':
    main()

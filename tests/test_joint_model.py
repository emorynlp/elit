# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 19:21
import unittest
from elit.server.en import en_services
from elit.server.format import Input


class TestJointModel(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_text_input(self):
        text = "Emory NLP is a research lab in Atlanta, GA. "
        "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
        doc = en_services.parser.parse([Input(text=text)])[0]
        print(doc)

    def test_sents_input(self):
        text = ["Emory NLP is a research lab in Atlanta, GA.",
                "It is founded by Jinho D. Choi in 2014.",
                'Dr. Choi is a professor at Emory University.']
        doc = en_services.parser.parse([Input(text=text)])[0]
        print(doc)

    def test_tokens_input(self):
        tokens = [
            "yes i do what 's your job".split(),
        ]
        doc = en_services.parser.parse([Input(tokens=tokens)])[0]
        print(doc)


if __name__ == '__main__':
    unittest.main()

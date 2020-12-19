# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 19:21
import unittest


class TestTransformerNamedEntityRecognizer(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_joint_model(self):
        from elit.server.en import en
        from elit.server.format import Input
        text = "Emory NLP is a research lab in Atlanta, GA. "
        "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."

        doc = en.parse([Input(text=text)])[0]
        print(doc)


if __name__ == '__main__':
    unittest.main()

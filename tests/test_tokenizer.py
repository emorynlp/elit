# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 19:21
import unittest

from elit.components.tokenizer import EnglishTokenizer


class TestTokenizer(unittest.TestCase):
    def test_tokenize(self):
        tokenizer = EnglishTokenizer()
        text = "Emory NLP is a research lab in Atlanta, GA. It is founded by Jinho D. Choi in 2014. Dr. Choi is a " \
               "professor at Emory University. "
        print(tokenizer.tokenize(text))

    def test_segment(self):
        tokenizer = EnglishTokenizer()
        print(tokenizer.segment(
            ['Emory', 'NLP', 'is', 'a', 'research', 'lab', 'in', 'Atlanta', ',', 'GA', '.', 'It', 'is', 'founded', 'by',
             'Jinho', 'D.', 'Choi', 'in', '2014', '.', 'Dr.', 'Choi', 'is', 'a', 'professor', 'at', 'Emory',
             'University', '.']))


if __name__ == '__main__':
    unittest.main()

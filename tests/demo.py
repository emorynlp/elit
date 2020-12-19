# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 19:21
from elit.server.en import en

text = [
    "Emory NLP is a research lab in Atlanta, GA. "
    "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
]
docs = en.parse(text)
for doc in docs:
    print(doc)

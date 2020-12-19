# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-13 20:21
from typing import List

import elit
from elit.common.document import Document
from elit.components.tokenizer import EnglishTokenizer
from elit.pretrained.mtl import LEM_POS_NER_DEP_SDP_CON_AMR_ELECTRA_BASE_EN
from elit.server.format import Input
from elit.server.service import Service

tokenizer = EnglishTokenizer()


def eos(text: List[str]) -> List[List[str]]:
    results = []
    for doc in text:
        tokens = tokenizer.tokenize(doc)
        sents = tokenizer.segment(tokens)
        results.append(['\t'.join(x) for x in sents])
    return results


def tokenize(sents: List[str]) -> List[List[str]]:
    return [tokenizer.tokenize(x) for x in sents]


model = elit.load(LEM_POS_NER_DEP_SDP_CON_AMR_ELECTRA_BASE_EN)
en = Service(
    model,
    eos,
    tokenize
)


def main():
    text = [
        "Emory NLP is a research lab in Atlanta, GA. "
        "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
    ]
    input = Input(text=text)
    input.models = ['lem']
    docs = en.parse([input])
    for doc in docs:
        print(doc)


if __name__ == '__main__':
    main()

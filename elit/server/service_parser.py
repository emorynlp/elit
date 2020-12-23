# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-14 12:05
from collections import defaultdict
from typing import List, Callable

from elit.common.document import Document
from elit.server.format import Input


class Service(object):

    def __init__(self,
                 model: Callable[[List[List[str]], List[str]], Document],
                 eos: Callable[[List[str]], List[List[str]]] = None,
                 tokenizer: Callable[[List[str]], List[List[str]]] = None) -> None:
        super().__init__()
        self.model = model
        self.eos = eos
        self.tokenizer = tokenizer

    def split_sents(self, text: List[str]) -> List[List[str]]:
        assert self.eos, 'eos is required to perform sentence split'
        return self.eos(text)

    def tokenize(self, docs: List[List[str]]) -> List[List[List[str]]]:
        assert self.tokenizer, 'tokenizer is required to perform tokenization'
        results = []
        tokens = self.tokenizer(sum(docs, []))
        for doc in docs:
            results.append([])
            results[-1].extend(tokens[:len(doc)])
            del tokens[:len(doc)]
        return results

    def parse_sents(self, sents: List[List[str]], tasks: List[str] = None) -> Document:
        return self.model(sents, tasks=tasks)

    def parse(self, inputs: List[Input]) -> List[Document]:
        needs_split = []
        input_ids = []
        for i, input in enumerate(inputs):
            if isinstance(input.text, str):
                needs_split.append(input.text)
                input_ids.append(i)
        for i, sents in zip(input_ids, self.split_sents(needs_split)):
            inputs[i].text = sents

        needs_tokenize = []
        input_ids = []
        for i, input in enumerate(inputs):
            if not input.tokens:
                needs_tokenize.append(input.text)
                input_ids.append(i)
        for i, tokens in zip(input_ids, self.tokenize(needs_tokenize)):
            inputs[i].tokens = tokens

        # We shall group by models
        inputs_by_tasks = defaultdict(list)
        for i, input in enumerate(inputs):
            tasks = tuple(sorted(input.models))
            inputs_by_tasks[tasks].append(i)

        results = [Document() for _ in inputs]
        for tasks, input_ids in inputs_by_tasks.items():
            group_inputs = [inputs[i] for i in input_ids]
            group_tokens = sum([input.tokens for input in group_inputs], [])
            annotations = self.parse_sents(group_tokens, tasks)
            for k, v in annotations.items():
                # fit ELIT standard
                if k == 'ner':
                    for j, s in enumerate(v):
                        v[j] = [x[1:] + x[:1] for x in s]
                elif k == 'srl':
                    for _v in v:
                        for j, s in enumerate(_v):
                            _v[j] = [x[1:] + x[:1] for x in s]
                elif k == 'dep':
                    for j, s in enumerate(v):
                        v[j] = [(x[0] - 1, x[1]) for x in s]
            for i, input in zip(input_ids, group_inputs):
                for k, v in annotations.items():
                    results[i][k] = v[:len(input.tokens)]
                    if k == 'ner':
                        if not input.verbose:
                            for j, s in enumerate(results[i][k]):
                                results[i][k][j] = [x[:-1] for x in s]
                    elif k == 'srl':
                        if not input.verbose:
                            for _v in v:
                                for j, s in enumerate(_v):
                                    _v[j] = [x[:-1] for x in s]
                    del v[:len(input.tokens)]

        return results

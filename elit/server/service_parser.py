# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-14 12:05
from collections import defaultdict
from typing import List, Callable

from elit.common.document import Document
from elit.server.format import Input
from elit.server.service_tokenizer import ServiceTokenizer
from elit.server.en_util import eos, tokenize


class ServiceParser(object):

    def __init__(self,
                 service_tokenizer: ServiceTokenizer,
                 model: Callable[[List[List[str]], List[str]], Document]) -> None:
        super().__init__()
        self.service_tokenizer = ServiceTokenizer(eos, tokenize) if service_tokenizer is None else service_tokenizer
        self.model = model

    def parse_sents(self, sents: List[List[str]], tasks: List[str] = None) -> Document:
        return self.model(sents, tasks=tasks)

    def parse(self, inputs: List[Input]) -> List[Document]:
        self.service_tokenizer.tokenize_inputs(inputs)  # no effects (read-only) in server pipeline

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

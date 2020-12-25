# ========================================================================
# Copyright 2020 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

# -*- coding:utf-8 -*-
# Author: hankcs, Liyan Xu
from typing import List, Callable, Union

from elit.server.format import Input


class ServiceTokenizer:

    def __init__(self,
                 eos: Callable[[List[str]], List[List[str]]] = None,
                 tokenizer: Callable[[List[str]], List[List[str]]] = None) -> None:
        super().__init__()
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

    def tokenize_inputs(self, inputs: Union[Input, List[Input]]) -> Union[Input, List[Input]]:
        single_input = False
        if isinstance(inputs, Input):
            single_input = True
            inputs = [inputs]

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

        return inputs[0] if single_input else inputs

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
from typing import List

from elit.components.tokenizer import EnglishTokenizer

tokenizer = EnglishTokenizer()


def eos(text: List[str]) -> List[List[str]]:
    results = []
    for doc in text:
        tokens = tokenizer.tokenize(doc)
        sents = tokenizer.segment(tokens)
        results.append(['\0'.join(x) for x in sents])
    return results


def tokenize(sents: List[str]) -> List[List[str]]:
    # Since text from user seldom contains \0, so we use \0 to indicate a pre-tokenized sentence
    return [x.split('\0') if '\0' in x else tokenizer.tokenize(x) for x in sents]

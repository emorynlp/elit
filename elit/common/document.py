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
# Author: hankcs
import json
from typing import List

from phrasetree.tree import Tree
from elit.common.structure import SerializableDict
from elit.utils.util import collapse_json


def tree_to_list(T):
    return [T.label(), [tree_to_list(t) if isinstance(t, Tree) else t for t in T]]


def list_to_tree(L):
    if isinstance(L, str):
        return L
    while len(L) == 1:
        L = L[0]
        if isinstance(L, str):
            return L
    return Tree(L[0], [list_to_tree(child) for child in L[1]])


class Sentence(SerializableDict):
    KEY_WORDS = 'words'
    KEY_POS = 'pos'
    KEY_NER = 'ner'

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.update(kwargs)

    @property
    def words(self) -> List[str]:
        return self.get(Sentence.KEY_WORDS)

    @words.setter
    def words(self, words: List[str]):
        self[Sentence.KEY_WORDS] = words


class Document(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if not v:
                continue
            if k == 'con':
                if isinstance(v, Tree) or isinstance(v[0], Tree):
                    continue
                flat = isinstance(v[0], str)
                if flat:
                    v = [v]
                ls = []
                for each in v:
                    if not isinstance(each, Tree):
                        ls.append(list_to_tree(each))
                if flat:
                    ls = ls[0]
                self[k] = ls
            elif k == 'amr':
                from stog.utils import penman
                from stog.data.dataset_readers.amr_parsing.amr import AMRGraph
                if isinstance(v, AMRGraph) or isinstance(v[0], AMRGraph):
                    continue
                flat = isinstance(v[0][0], str)
                if flat:
                    v = [v]
                graphs = [AMRGraph(penman.Graph(triples)) for triples in v]
                if flat:
                    graphs = graphs[0]
                self[k] = graphs

    def to_json(self, ensure_ascii=False, indent=2) -> str:
        d = self.to_dict()
        text = json.dumps(d, ensure_ascii=ensure_ascii, indent=indent, default=lambda o: repr(o))
        text = collapse_json(text, 4)
        return text

    def to_dict(self):
        d = dict(self)
        for k, v in self.items():
            if not v:
                continue
            if k == 'con':
                if not isinstance(v, Tree) and not isinstance(v[0], Tree):
                    continue
                flat = isinstance(v, Tree)
                if flat:
                    v = [v]
                ls = []
                for each in v:
                    if isinstance(each, Tree):
                        ls.append(tree_to_list(each))
                if flat:
                    ls = ls[0]
                d[k] = ls
            elif k == 'amr':
                from stog.data.dataset_readers.amr_parsing.amr import AMRGraph
                flat = isinstance(v, AMRGraph)
                if flat:
                    v: List[AMRGraph] = [v]
                triples = [x.triples() for x in v]
                if flat:
                    triples = triples[0]
                d[k] = triples
        return d

    def __str__(self) -> str:
        return self.to_json()

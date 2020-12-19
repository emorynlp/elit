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
from typing import List, Union, Tuple

from elit.common.structure import SerializableDict
from elit.utils.io_util import get_resource, TimingFileIterator
from elit.utils.log_util import logger, markdown_table


class CoNLLWord(SerializableDict):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, deprel=None, phead=None,
                 pdeprel=None):
        """CoNLL format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf

        Parameters
        ----------
        id : int
            Token counter, starting at 1 for each new sentence.
        form : str
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        cpos : str
            Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
        pos : str
            Fine-grained part-of-speech tag, where the tagset depends on the treebank.
        feats : str
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or an underscore if not available.
        head : Union[int, List[int]]
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        deprel : Union[str, List[str]]
            Dependency relation to the HEAD.
        phead : int
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        pdeprel : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = sanitize_conll_int_value(id)
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = sanitize_conll_int_value(head)
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        if isinstance(self.head, list):
            return '\n'.join('\t'.join(['_' if v is None else v for v in values]) for values in [
                [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                 None if head is None else str(head), deprel, self.phead, self.pdeprel] for head, deprel in
                zip(self.head, self.deprel)
            ])
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  None if self.head is None else str(self.head), self.deprel, self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        return list(f for f in
                    [self.form, self.lemma, self.cpos, self.pos, self.feats, self.head, self.deprel, self.phead,
                     self.pdeprel] if f)

    def get_pos(self):
        """
        Get the precisest pos for this word.

        Returns: self.pos or self.cpos

        """
        return self.pos or self.cpos


class CoNLLUWord(SerializableDict):
    def __init__(self, id: Union[int, str], form, lemma=None, upos=None, xpos=None, feats=None, head=None, deprel=None,
                 deps=None,
                 misc=None):
        """CoNLL-U format template, see https://universaldependencies.org/format.html

        Parameters
        ----------
        id : Union[int, str]
            Token counter, starting at 1 for each new sentence.
        form : Union[str, None]
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        upos : str
            Universal part-of-speech tag.
        xpos : str
            Language-specific part-of-speech tag; underscore if not available.
        feats : str
            List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
        head : int
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        deprel : str
            Dependency relation to the HEAD.
        deps : Union[List[Tuple[int, str], str]
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        misc : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = sanitize_conll_int_value(id)
        self.form = form
        self.upos = upos
        self.xpos = xpos
        if isinstance(head, list):
            assert deps is None, 'When head is a list, deps has to be None'
            assert isinstance(deprel, list), 'When head is a list, deprel has to be a list'
            assert len(deprel) == len(head), 'When head is a list, deprel has to match its length'
            deps = list(zip(head, deprel))
            head = None
            deprel = None
        self.head = sanitize_conll_int_value(head)
        self.deprel = deprel
        self.lemma = lemma
        self.feats = feats
        if deps == '_':
            deps = None
        if isinstance(deps, str):
            self.deps = []
            for pair in deps.split('|'):
                h, r = pair.split(':')
                h = int(h)
                self.deps.append((h, r))
        else:
            self.deps = deps
        self.misc = misc

    def __str__(self):
        deps = self.deps
        if not deps:
            deps = None
        else:
            deps = '|'.join(f'{h}:{r}' for h, r in deps)
        values = [str(self.id), self.form, self.lemma, self.upos, self.xpos, self.feats,
                  str(self.head) if self.head is not None else None, self.deprel, deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

    @property
    def nonempty_fields(self):
        return list(f for f in
                    [self.form, self.lemma, self.upos, self.xpos, self.feats, self.head, self.deprel, self.deps,
                     self.misc] if f)

    def get_pos(self):
        """
        Get the precisest pos for this word.

        Returns: self.xpos or self.upos

        """
        return self.xpos or self.upos


class CoNLLSentence(list):
    def __init__(self, words=None):
        """A list of ConllWord"""
        super().__init__()
        if words:
            self.extend(words)

    def __str__(self):
        return '\n'.join([word.__str__() for word in self])

    @staticmethod
    def from_str(conll: str, conllu=False):
        """Build a CoNLLSentence from CoNLL-X format str

        Args:
          conll(str): CoNLL-X format string
          CoNLL-U(bool): 
          conll: str: 
          conllu:  (Default value = False) True to create words in CoNLLUWord representation

        Returns:

        
        """
        words: List[CoNLLWord] = []
        prev_id = None
        for line in conll.strip().split('\n'):
            if line.startswith('#'):
                continue
            cells = line.split('\t')
            cells = [None if c == '_' else c for c in cells]
            if '-' in cells[0]:
                continue
            cells[0] = int(cells[0])
            cells[6] = int(cells[6])
            if cells[0] != prev_id:
                words.append(CoNLLUWord(*cells) if conllu else CoNLLWord(*cells))
            else:
                if isinstance(words[-1].head, list):
                    words[-1].head.append(cells[6])
                    words[-1].deprel.append(cells[7])
                else:
                    words[-1].head = [words[-1].head] + [cells[6]]
                    words[-1].deprel = [words[-1].deprel] + [cells[7]]
            prev_id = cells[0]
        if conllu:
            for word in words:  # type: CoNLLUWord
                if isinstance(word.head, list):
                    assert not word.deps
                    word.deps = list(zip(word.head, word.deprel))
                    word.head = None
                    word.deprel = None
        return CoNLLSentence(words)

    @staticmethod
    def from_file(path: str, conllu=False):
        """

        Args:
          path: 
          conllu:  (Default value = False)
          path: str: 

        Returns:

        
        """
        with open(path) as src:
            return [CoNLLSentence.from_str(x, conllu) for x in src.read().split('\n\n') if x.strip()]

    @staticmethod
    def from_dict(d: dict, conllu=False):
        if conllu:
            headings = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        else:
            headings = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
        words: List[Union[CoNLLWord, CoNLLUWord]] = []
        for cells in zip(*list(d[f] for f in headings)):
            words.append(CoNLLUWord(*cells) if conllu else CoNLLWord(*cells))
        return CoNLLSentence(words)

    def to_markdown(self, headings='auto') -> str:
        cells = [str(word).split('\t') for word in self]
        if headings == 'auto':
            if isinstance(self[0], CoNLLWord):
                headings = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
            else:  # conllu
                headings = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
                for each in cells:
                    # if '|' in each[8]:
                    # each[8] = f'`{each[8]}`'
                    each[8] = each[8].replace('|', '⎮')
        alignment = [('^', '>'), ('^', '<'), ('^', '<'), ('^', '<'), ('^', '<'), ('^', '<'), ('^', '>'), ('^', '<'),
                     ('^', '<'), ('^', '<')]
        text = markdown_table(headings, cells, alignment=alignment)
        return text


def collapse_enhanced_empty_nodes(sent: list):
    collapsed = []
    for cells in sent:
        if isinstance(cells[0], float):
            id = cells[0]
            head, deprel = cells[8].split(':', 1)
            for x in sent:
                arrows = [s.split(':', 1) for s in x[8].split('|')]
                arrows = [(head, f'{head}:{deprel}>{r}') if h == str(id) else (h, r) for h, r in arrows]
                arrows = sorted(arrows)
                x[8] = '|'.join(f'{h}:{r}' for h, r in arrows)
            sent[head][7] += f'>{cells[7]}'
        else:
            collapsed.append(cells)
    return collapsed


def read_conll(filepath: Union[str, TimingFileIterator], underline_to_none=False, enhanced_collapse_empty_nodes=False):
    sent = []
    if isinstance(filepath, str):
        filepath: str = get_resource(filepath)
        if filepath.endswith('.conllu') and enhanced_collapse_empty_nodes is None:
            enhanced_collapse_empty_nodes = True
        src = open(filepath, encoding='utf-8')
    else:
        src = filepath
    for idx, line in enumerate(src):
        if line.startswith('#'):
            continue
        line = line.strip()
        cells = line.split('\t')
        if line and cells:
            if enhanced_collapse_empty_nodes and '.' in cells[0]:
                cells[0] = float(cells[0])
                cells[6] = None
            else:
                if '-' in cells[0] or '.' in cells[0]:
                    # sent[-1][1] += cells[1]
                    continue
                cells[0] = int(cells[0])
                if cells[6] != '_':
                    try:
                        cells[6] = int(cells[6])
                    except ValueError:
                        cells[6] = 0
                        logger.exception(f'Wrong CoNLL format {filepath}:{idx + 1}\n{line}')
            if underline_to_none:
                for i, x in enumerate(cells):
                    if x == '_':
                        cells[i] = None
            sent.append(cells)
        else:
            if enhanced_collapse_empty_nodes:
                sent = collapse_enhanced_empty_nodes(sent)
            yield sent
            sent = []

    if sent:
        if enhanced_collapse_empty_nodes:
            sent = collapse_enhanced_empty_nodes(sent)
        yield sent

    src.close()


def sanitize_conll_int_value(value: Union[str, int]):
    if value is None or isinstance(value, int):
        return value
    if value == '_':
        return None
    return int(value)

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
import itertools
from collections import Counter
from typing import Union, List, Callable

from elit.common.dataset import TransformDataset
from elit.utils.io_util import TimingFileIterator
from elit.utils.log_util import cprint
from elit.utils.string_util import ispunct

SETIMES2_EN_HR_SENTENCES_HOME = 'https://schweter.eu/cloud/nn_eos/SETIMES2.en-hr.sentences.tar.xz'
SETIMES2_EN_HR_HR_SENTENCES_TRAIN = SETIMES2_EN_HR_SENTENCES_HOME + '#SETIMES2.en-hr.hr.sentences.train'
SETIMES2_EN_HR_HR_SENTENCES_DEV = SETIMES2_EN_HR_SENTENCES_HOME + '#SETIMES2.en-hr.hr.sentences.dev'
SETIMES2_EN_HR_HR_SENTENCES_TEST = SETIMES2_EN_HR_SENTENCES_HOME + '#SETIMES2.en-hr.hr.sentences.test'

EUROPARL_V7_DE_EN_EN_SENTENCES_HOME = 'http://schweter.eu/cloud/nn_eos/europarl-v7.de-en.en.sentences.tar.xz'
EUROPARL_V7_DE_EN_EN_SENTENCES_TRAIN = EUROPARL_V7_DE_EN_EN_SENTENCES_HOME + '#europarl-v7.de-en.en.sentences.train'
EUROPARL_V7_DE_EN_EN_SENTENCES_DEV = EUROPARL_V7_DE_EN_EN_SENTENCES_HOME + '#europarl-v7.de-en.en.sentences.dev'
EUROPARL_V7_DE_EN_EN_SENTENCES_TEST = EUROPARL_V7_DE_EN_EN_SENTENCES_HOME + '#europarl-v7.de-en.en.sentences.test'


class SentenceBoundaryDetectionDataset(TransformDataset):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 # char_vocab=None,
                 append_after_sentence=None,
                 eos_chars=None,
                 eos_char_min_freq=200,
                 eos_char_is_punct=True,
                 window_size=5,
                 **kwargs
                 ) -> None:
        # self.char_vocab = char_vocab
        self.eos_char_is_punct = eos_char_is_punct
        self.append_after_sentence = append_after_sentence
        self.window_size = window_size
        self.eos_chars = eos_chars
        self.eos_char_min_freq = eos_char_min_freq
        super().__init__(data, transform, cache)

    def load_file(self, filepath: str):
        f = TimingFileIterator(filepath)
        sents = []
        eos_offsets = []
        offset = 0
        for line in f:
            if not line.strip():
                continue
            line = line.rstrip('\n')
            eos_offsets.append(offset + len(line.rstrip()) - 1)
            offset += len(line)
            if self.append_after_sentence:
                line += self.append_after_sentence
                offset += len(self.append_after_sentence)
            f.log(line)
            sents.append(line)
        f.erase()
        corpus = list(itertools.chain.from_iterable(sents))

        if self.eos_chars:
            if not isinstance(self.eos_chars, set):
                self.eos_chars = set(self.eos_chars)
        else:
            eos_chars = Counter()
            for i in eos_offsets:
                eos_chars[corpus[i]] += 1
            self.eos_chars = set(k for (k, v) in eos_chars.most_common() if
                                 v >= self.eos_char_min_freq and (not self.eos_char_is_punct or ispunct(k)))
            cprint(f'eos_chars = [yellow]{self.eos_chars}[/yellow]')

        eos_index = 0
        eos_offsets = [i for i in eos_offsets if corpus[i] in self.eos_chars]
        window_size = self.window_size
        for i, c in enumerate(corpus):
            if c in self.eos_chars:
                window = corpus[i - window_size: i + window_size + 1]
                label_id = 1. if eos_offsets[eos_index] == i else 0.
                if label_id > 0:
                    eos_index += 1
                yield {'char': window, 'label_id': label_id}
        assert eos_index == len(eos_offsets), f'{eos_index} != {len(eos_offsets)}'

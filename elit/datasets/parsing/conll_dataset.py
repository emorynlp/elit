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
from typing import Union, List, Callable, Dict

from elit.common.constant import ROOT, EOS, BOS
from elit.common.dataset import TransformDataset
from elit.components.parsers.conll import read_conll
from elit.utils.io_util import TimingFileIterator
from elit.utils.log_util import flash


class CoNLLParsingDataset(TransformDataset):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None,
                 prune: Callable[[Dict[str, List[str]]], bool] = None) -> None:
        self._prune = prune
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath):
        if filepath.endswith('.conllu'):
            # See https://universaldependencies.org/format.html
            field_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                           'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        else:
            field_names = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                           'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
        fp = TimingFileIterator(filepath)
        for idx, sent in enumerate(read_conll(fp)):
            sample = {}
            for i, field in enumerate(field_names):
                sample[field] = [cell[i] for cell in sent]
            if not self._prune or not self._prune(sample):
                yield sample
            fp.log(f'{idx + 1} samples [blink][yellow]...[/yellow][/blink]')

    def __len__(self) -> int:
        return len(self.data)


def append_bos(sample: dict, pos_key='CPOS', bos=ROOT) -> dict:
    """

    Args:
        sample:
        pos_key:
        bos: A special token inserted to the head of tokens.

    Returns:

    """
    sample['token'] = [bos] + sample['FORM']
    if pos_key in sample:
        sample['pos'] = [ROOT] + sample[pos_key]
    if 'HEAD' in sample:
        sample['arc'] = [0] + sample['HEAD']
        sample['rel'] = sample['DEPREL'][:1] + sample['DEPREL']
    return sample


def append_bos_eos(sample: dict) -> dict:
    sample['token'] = [BOS] + sample['FORM'] + [EOS]
    if 'CPOS' in sample:
        sample['pos'] = [BOS] + sample['CPOS'] + [EOS]
    if 'HEAD' in sample:
        sample['arc'] = [0] + sample['HEAD'] + [0]
        sample['rel'] = sample['DEPREL'][:1] + sample['DEPREL'] + sample['DEPREL'][:1]
    return sample


def get_sibs(sample: dict) -> dict:
    heads = sample.get('arc', None)
    if heads:
        sibs = [-1] * len(heads)
        for i in range(1, len(heads)):
            hi = heads[i]
            for j in range(i + 1, len(heads)):
                hj = heads[j]
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i] = j
                    else:
                        sibs[j] = i
                    break
        sample['sib_id'] = [0] + sibs[1:]
    return sample

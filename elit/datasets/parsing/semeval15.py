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
import warnings

from elit.common.constant import ROOT, PAD
from elit.components.parsers.conll import CoNLLSentence


def unpack_deps_to_head_deprel(sample: dict, pad_rel=None, arc_key='arc', rel_key='rel'):
    if 'DEPS' in sample:
        deps = ['_'] + sample['DEPS']
        sample[arc_key] = arc = []
        sample[rel_key] = rel = []
        for each in deps:
            arc_per_token = [False] * len(deps)
            rel_per_token = [None] * len(deps)
            if each != '_':
                for ar in each.split('|'):
                    a, r = ar.split(':')
                    a = int(a)
                    arc_per_token[a] = True
                    rel_per_token[a] = r
                    if not pad_rel:
                        pad_rel = r
            arc.append(arc_per_token)
            rel.append(rel_per_token)
        if not pad_rel:
            pad_rel = PAD
        for i in range(len(rel)):
            rel[i] = [r if r else pad_rel for r in rel[i]]
    return sample


def append_bos_to_form_pos(sample, pos_key='CPOS'):
    sample['token'] = [ROOT] + sample['FORM']
    if pos_key in sample:
        sample['pos'] = [ROOT] + sample[pos_key]
    return sample


def merge_head_deprel_with_2nd(sample: dict):
    if 'arc' in sample:
        arc_2nd = sample['arc_2nd']
        rel_2nd = sample['rel_2nd']
        for i, (arc, rel) in enumerate(zip(sample['arc'], sample['rel'])):
            if i:
                if arc_2nd[i][arc] and rel_2nd[i][arc] != rel:
                    sample_str = CoNLLSentence.from_dict(sample, conllu=True).to_markdown()
                    warnings.warn(f'The main dependency conflicts with 2nd dependency at ID={i}, ' \
                                  'which means joint mode might not be suitable. ' \
                                  f'The sample is\n{sample_str}')
                arc_2nd[i][arc] = True
                rel_2nd[i][arc] = rel
    return sample

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
from typing import List

from elit.common.transform import TransformList
from elit.components.parsers.udify.lemma_edit import gen_lemma_rule, apply_lemma_rule
from elit.components.taggers.transformers.transformer_tagger import TransformerTagger


def add_lemma_rules_to_sample(sample: dict):
    if 'tag' in sample and 'lemma' not in sample:
        lemma_rules = [gen_lemma_rule(word, lemma)
                       if lemma != "_" else "_"
                       for word, lemma in zip(sample['token'], sample['tag'])]
        sample['lemma'] = sample['tag'] = lemma_rules
    return sample


class TransformerLemmatizer(TransformerTagger):
    def build_dataset(self, data, transform=None, **kwargs):
        if not isinstance(transform, list):
            transform = TransformList()
        transform.append(add_lemma_rules_to_sample)
        return super().build_dataset(data, transform, **kwargs)

    def prediction_to_human(self, pred, vocab: List[str], batch, token=None):
        if token is None:
            token = batch['token']
        rules = super().prediction_to_human(pred, vocab, batch)
        for token_per_sent, rule_per_sent in zip(token, rules):
            lemma_per_sent = [apply_lemma_rule(t, r) for t, r in zip(token_per_sent, rule_per_sent)]
            yield lemma_per_sent

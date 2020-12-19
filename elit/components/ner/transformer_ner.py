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
import functools
from typing import Union, List

import torch

from elit.common.dataset import SamplerBuilder
from elit.components.taggers.transformers.transformer_tagger import TransformerTagger
from elit.metrics.chunking.sequence_labeling import get_entities
from elit.metrics.f1 import F1
from elit.utils.io_util import prune_ner_tagset
from elit.utils.string_util import guess_delimiter
from elit.utils.util import merge_locals_kwargs


class TransformerNamedEntityRecognizer(TransformerTagger):
    def build_metric(self, **kwargs):
        return F1()

    # noinspection PyMethodOverriding
    def update_metrics(self, metric, logits, y, mask, batch, prediction):
        for p, g in zip(prediction, self.tag_to_span(batch['tag'])):
            pred = set(p)
            gold = set(g)
            metric(pred, gold)

    # noinspection PyMethodOverriding
    def decode_output(self, logits, mask, batch, model=None):
        output = super().decode_output(logits, mask, batch, model)
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        prediction = self.id_to_tags(output, [len(x) for x in batch['token']])
        return self.tag_to_span(prediction)

    @staticmethod
    def tag_to_span(batch_tags):
        batch = []
        for tags in batch_tags:
            batch.append(get_entities(tags))
        return batch

    def decorate_spans(self, spans, batch):
        batch_ner = []
        delimiter_in_entity = self.config.get('delimiter_in_entity', ' ')
        for spans_per_sent, tokens in zip(spans, batch[self.config.token_key]):
            ner_per_sent = []
            for label, start, end in spans_per_sent:
                ner_per_sent.append((delimiter_in_entity.join(tokens[start:end]), label, start, end))
            batch_ner.append(ner_per_sent)
        return batch_ner

    def generate_prediction_filename(self, tst_data, save_dir):
        return super().generate_prediction_filename(tst_data.replace('.tsv', '.txt'), save_dir)

    def prediction_to_human(self, pred, vocab, batch):
        return self.decorate_spans(pred, batch)

    def input_is_flat(self, tokens):
        return tokens and isinstance(tokens, list) and isinstance(tokens[0], str)

    def fit(self, trn_data, dev_data, save_dir, transformer,
            delimiter_in_entity=None,
            average_subwords=False,
            word_dropout: float = 0.2,
            hidden_dropout=None,
            layer_dropout=0,
            scalar_mix=None,
            mix_embedding: int = 0,
            grad_norm=5.0,
            lr=5e-5,
            transformer_lr=None,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            crf=False,
            secondary_encoder=None,
            reduction='sum',
            batch_size=32,
            sampler_builder: SamplerBuilder = None,
            epochs=3,
            tagset=None,
            token_key=None,
            delimiter=None,
            max_seq_len=None,
            sent_delimiter=None,
            char_level=False,
            hard_constraint=False,
            transform=None,
            logger=None,
            devices: Union[float, int, List[int]] = None,
            **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, trn, logger, **kwargs):
        super().build_vocabs(trn, logger, **kwargs)
        if self.config.get('delimiter_in_entity', None) is None:
            # Check the first sample to guess the delimiter between tokens in a NE
            tokens = trn[0][self.config.token_key]
            delimiter_in_entity = guess_delimiter(tokens)
            logger.info(f'Guess the delimiter between tokens in named entity could be [blue]"{delimiter_in_entity}'
                        f'"[/blue]. If not, specify `delimiter_in_entity` in `fit()`')
            self.config.delimiter_in_entity = delimiter_in_entity

    def build_dataset(self, data, transform=None, **kwargs):
        dataset = super().build_dataset(data, transform, **kwargs)
        if isinstance(data, str):
            tagset = self.config.get('tagset', None)
            if tagset:
                dataset.append_transform(functools.partial(prune_ner_tagset, tagset=tagset))
        return dataset

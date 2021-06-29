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
import logging
from typing import Dict, Any, Union, Iterable, List

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from elit.common.constant import EOS
from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict, TransformList
from elit.components.mtl.tasks import Task
from elit.components.parsers.biaffine.biaffine_dep import BiaffineDependencyParser
from elit.components.parsers.biaffine.biaffine_model import BiaffineDecoder
from elit.datasets.parsing.conll_dataset import append_bos
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.time_util import CountdownTimer
from elit.utils.util import merge_locals_kwargs


class BiaffineDependencyParsing(Task, BiaffineDependencyParser):
    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=2e-3, separate_optimizer=False,
                 cls_is_bos=True,
                 sep_is_eos=False,
                 punct=False,
                 tree=False,
                 proj=False,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 mu=.9,
                 nu=.9,
                 epsilon=1e-12,
                 decay=.75,
                 decay_steps=5000,
                 use_pos=False,
                 max_seq_len=None,
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        BiaffineDependencyParser.update_metric(self, *prediction, batch['arc'], batch['rel_id'], output[1],
                                               batch.get('punct_mask', None), metric, batch)

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder, **kwargs) -> Union[Dict[str, Any], Any]:
        (arc_scores, rel_scores), mask = output
        return BiaffineDependencyParser.decode(self, arc_scores, rel_scores, mask, batch)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        (arc_scores, rel_scores), mask = output
        return BiaffineDependencyParser.compute_loss(self, arc_scores, rel_scores, batch['arc'], batch['rel_id'], mask,
                                                     criterion,
                                                     batch)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineDecoder(encoder_size, self.config.n_mlp_arc, self.config.n_mlp_rel, self.config.mlp_dropout,
                               len(self.vocabs.rel))

    def build_metric(self, **kwargs):
        return BiaffineDependencyParser.build_metric(self, **kwargs)

    def build_dataloader(self, data, transform: TransformList = None, training=False, device=None,
                         logger: logging.Logger = None, gradient_accumulation=1, **kwargs) -> DataLoader:
        transform.insert(0, append_bos)
        dataset = BiaffineDependencyParser.build_dataset(self, data, transform)
        if isinstance(data, str):
            dataset.purge_cache()
        if self.vocabs.mutable:
            BiaffineDependencyParser.build_vocabs(self, dataset, logger, transformer=True)
        if dataset.cache:
            timer = CountdownTimer(len(dataset))
            BiaffineDependencyParser.cache_dataset(self, dataset, timer, training, logger)
        max_seq_len = self.config.get('max_seq_len', None)
        if max_seq_len and isinstance(data, str):
            dataset.prune(lambda x: len(x['token_input_ids']) > 510, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(
                self.compute_lens(data, dataset, length_field='FORM'),
                shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset,
            pad=self.get_pad_dict())

    def feed_batch(self, h: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        logits = super().feed_batch(h, batch, mask, decoder)
        mask = mask.clone()
        mask[:, 0] = 0
        return logits, mask

    def build_optimizer(self, decoder: torch.nn.Module, **kwargs):
        config = self.config
        optimizer = Adam(decoder.parameters(),
                         config.lr,
                         (config.mu, config.nu),
                         config.epsilon)
        scheduler = ExponentialLR(optimizer, config.decay ** (1 / config.decay_steps))
        return optimizer, scheduler

    def input_is_flat(self, data) -> bool:
        return BiaffineDependencyParser.input_is_flat(self, data, self.config.use_pos)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        arcs, rels = prediction
        arcs = arcs[:, 1:]  # Skip the ROOT
        rels = rels[:, 1:]
        arcs = arcs.tolist()
        rels = rels.tolist()
        vocab = self.vocabs['rel'].idx_to_token
        for arcs_per_sent, rels_per_sent, tokens in zip(arcs, rels, batch['token']):
            tokens = tokens[1:]
            sent_len = len(tokens)
            result = list(zip(arcs_per_sent[:sent_len], [vocab[r] for r in rels_per_sent[:sent_len]]))
            result = [(x[0] - 1, x[1]) for x in result]
            yield result

    def build_samples(self, inputs, cls_is_bos=False, sep_is_eos=False):
        return [{'FORM': token + ([EOS] if sep_is_eos else [])} for token in inputs]

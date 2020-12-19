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

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict, TransformList
from elit.components.mtl.tasks import Task
from elit.components.parsers.biaffine.biaffine_model import BiaffineDecoder
from elit.components.parsers.biaffine.biaffine_sdp import BiaffineSemanticDependencyParser
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.time_util import CountdownTimer
from elit.utils.util import merge_locals_kwargs


class BiaffineSemanticDependencyParsing(Task, BiaffineSemanticDependencyParser):
    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=2e-3, separate_optimizer=False,
                 punct=False,
                 tree=True,
                 pad_rel=None,
                 apply_constraint=False,
                 single_root=True,
                 no_zero_head=None,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 mu=.9,
                 nu=.9,
                 epsilon=1e-12,
                 decay=.75,
                 decay_steps=5000,
                 cls_is_bos=True,
                 use_pos=False,
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        BiaffineSemanticDependencyParser.update_metric(self, *prediction, batch['arc'], batch['rel_id'], output[1],
                                                       output[-1], metric, batch)

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder, **kwargs) -> Union[Dict[str, Any], Any]:
        (arc_scores, rel_scores), mask, punct_mask = output
        return BiaffineSemanticDependencyParser.decode(self, arc_scores, rel_scores, mask, batch)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        (arc_scores, rel_scores), mask, punct_mask = output
        return BiaffineSemanticDependencyParser.compute_loss(self, arc_scores, rel_scores, batch['arc'],
                                                             batch['rel_id'], mask, criterion,
                                                             batch)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineDecoder(encoder_size, self.config.n_mlp_arc, self.config.n_mlp_rel, self.config.mlp_dropout,
                               len(self.vocabs.rel))

    def build_metric(self, **kwargs):
        return BiaffineSemanticDependencyParser.build_metric(self, **kwargs)

    def build_dataloader(self, data, transform: TransformList = None, training=False, device=None,
                         logger: logging.Logger = None, gradient_accumulation=1, **kwargs) -> DataLoader:
        if isinstance(data, list):
            data = BiaffineSemanticDependencyParser.build_samples(self, data, self.config.use_pos)
        dataset = BiaffineSemanticDependencyParser.build_dataset(self, data, transform)
        if isinstance(data, str):
            dataset.purge_cache()
        if self.vocabs.mutable:
            BiaffineSemanticDependencyParser.build_vocabs(self, dataset, logger, transformer=True)
        if dataset.cache:
            timer = CountdownTimer(len(dataset))
            BiaffineSemanticDependencyParser.cache_dataset(self, dataset, timer, training, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset,
            pad=self.get_pad_dict())

    def feed_batch(self, h: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        logits = super().feed_batch(h, batch, mask, decoder)
        arc_scores = logits[0]
        mask = mask.clone()
        mask[:, 0] = 0
        mask = self.convert_to_3d_mask(arc_scores, mask)
        punct_mask = self.convert_to_3d_puncts(batch.get('punct_mask', None), mask)
        return logits, mask, punct_mask

    def build_optimizer(self, decoder: torch.nn.Module, **kwargs):
        config = self.config
        optimizer = Adam(decoder.parameters(),
                         config.lr,
                         (config.mu, config.nu),
                         config.epsilon)
        scheduler = ExponentialLR(optimizer, config.decay ** (1 / config.decay_steps))
        return optimizer, scheduler

    def input_is_flat(self, data) -> bool:
        return BiaffineSemanticDependencyParser.input_is_flat(self, data, self.config.use_pos)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        arcs, rels = prediction
        arcs = arcs[:, 1:, :]  # Skip the ROOT
        rels = rels[:, 1:, :]
        arcs = arcs.tolist()
        rels = rels.tolist()
        vocab = self.vocabs['rel'].idx_to_token
        for arcs_per_sent, rels_per_sent, tokens in zip(arcs, rels, batch['token']):
            tokens = tokens[1:]
            sent_len = len(tokens)
            result = []
            for a, r in zip(arcs_per_sent[:sent_len], rels_per_sent[:sent_len]):
                heads = [i for i in range(sent_len + 1) if a[i]]
                deprels = [vocab[r[i]] for i in range(sent_len + 1) if a[i]]
                result.append(list(zip(heads, deprels)))
            yield result

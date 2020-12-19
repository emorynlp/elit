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
from typing import Union, List, Dict, Any, Iterable, Callable

import torch
from torch.utils.data import DataLoader

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict
from elit.components.mtl.tasks import Task
from elit.components.srl.span_rank.span_rank import SpanRankingSemanticRoleLabeler
from elit.components.srl.span_rank.span_ranking_srl_model import SpanRankingSRLDecoder
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.util import merge_locals_kwargs


class SpanRankingSemanticRoleLabeling(Task, SpanRankingSemanticRoleLabeler):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None, use_raw_hidden_states=False,
                 lr=1e-3, separate_optimizer=False,
                 lexical_dropout=0.5,
                 dropout=0.2,
                 span_width_feature_size=20,
                 ffnn_size=150,
                 ffnn_depth=2,
                 argument_ratio=0.8,
                 predicate_ratio=0.4,
                 max_arg_width=30,
                 mlp_label_size=100,
                 enforce_srl_constraint=False,
                 use_gold_predicates=False,
                 doc_level_offset=True,
                 use_biaffine=False,
                 loss_reduction='mean',
                 with_argument=' ',
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self, data, transform: Callable = None, training=False, device=None,
                         logger: logging.Logger = None, gradient_accumulation=1, **kwargs) -> DataLoader:
        dataset = self.build_dataset(data, isinstance(data, list), logger, transform)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        return SpanRankingSemanticRoleLabeler.update_metrics(self, batch, {'prediction': prediction},
                                                             tuple(metric.values()))

    def decode_output(self,
                      output: Dict[str, Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder, **kwargs) -> Union[Dict[str, Any], Any]:
        return SpanRankingSemanticRoleLabeler.decode_output(self, output, batch)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return output['loss']

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return SpanRankingSRLDecoder(encoder_size, len(self.vocabs.srl_label), self.config)

    def build_metric(self, **kwargs):
        predicate_f1, end_to_end_f1 = SpanRankingSemanticRoleLabeler.build_metric(self, **kwargs)
        return MetricDict({'predicate': predicate_f1, 'e2e': end_to_end_f1})

    def build_criterion(self, **kwargs):
        pass

    def input_is_flat(self, data) -> bool:
        return SpanRankingSemanticRoleLabeler.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        return SpanRankingSemanticRoleLabeler.format_dict_to_results(batch['token'], prediction, exclusive_offset=True,
                                                                     with_predicate=True,
                                                                     with_argument=self.config.get('with_argument',
                                                                                                   ' '),
                                                                     label_first=True)

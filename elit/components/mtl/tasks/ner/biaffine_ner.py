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
from copy import copy
from typing import Callable, Dict, Any, Union, Iterable, List

import torch
from torch.utils.data import DataLoader

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict, TransformList
from elit.components.mtl.tasks import Task
from elit.components.ner.biaffine_ner.biaffine_ner import BiaffineNamedEntityRecognizer
from elit.components.ner.biaffine_ner.biaffine_ner_model import BiaffineNamedEntityRecognitionDecoder
from elit.datasets.ner.ontonotes import unpack_ner
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.util import merge_locals_kwargs


class BiaffineNamedEntityRecognition(Task, BiaffineNamedEntityRecognizer):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None, use_raw_hidden_states=False,
                 lr=None, separate_optimizer=False,
                 doc_level_offset=True, is_flat_ner=True, tagset=None, ret_tokens=' ',
                 ffnn_size=150, loss_reduction='mean', **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        BiaffineNamedEntityRecognizer.update_metrics(self, batch, prediction, metric)

    def decode_output(self,
                      output: Dict[str, Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder,
                      **kwargs) -> Union[Dict[str, Any], Any]:
        return self.get_pred_ner(batch['token'], output['candidate_ner_scores'])

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return output['loss']

    def build_dataloader(self, data,
                         transform: TransformList = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        transform = copy(transform)
        transform.append(unpack_ner)
        dataset = BiaffineNamedEntityRecognizer.build_dataset(self, data, self.vocabs, transform)
        if self.vocabs.mutable:
            BiaffineNamedEntityRecognizer.build_vocabs(self, dataset, logger, self.vocabs)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineNamedEntityRecognitionDecoder(encoder_size, self.config.ffnn_size, len(self.vocabs.label),
                                                     self.config.loss_reduction)

    def build_metric(self, **kwargs):
        return BiaffineNamedEntityRecognizer.build_metric(self, **kwargs)

    def input_is_flat(self, data) -> bool:
        return BiaffineNamedEntityRecognizer.input_is_flat(data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        results = []
        BiaffineNamedEntityRecognizer.prediction_to_result(batch['token'], prediction, results,
                                                           ret_tokens=self.config.get('ret_tokens', ' '))
        return results

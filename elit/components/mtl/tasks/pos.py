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
from typing import Dict, Any, Union, Iterable, Callable, List

import torch
from torch.utils.data import DataLoader

from elit.common.dataset import SamplerBuilder, PadSequenceDataLoader
from elit.common.transform import VocabDict
from elit.components.mtl.tasks import Task
from elit.components.taggers.transformers.transformer_tagger import TransformerTagger
from elit.layers.crf.crf import CRF
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.utils.util import merge_locals_kwargs


class LinearCRFDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_labels,
                 crf=False,
                 crf_constraints=None) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, crf_constraints) if crf else None

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        return self.classifier(contextualized_embeddings)


class TransformerTagging(Task, TransformerTagger):

    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=1e-3,
                 separate_optimizer=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 delimiter=None,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False,
                 tagging_scheme=None,
                 crf=False,
                 token_key='token', **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self,
                         data,
                         transform: Callable = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         cache=False,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        args = dict((k, self.config[k]) for k in
                    ['delimiter', 'max_seq_len', 'sent_delimiter', 'char_level', 'hard_constraint'])
        dataset = self.build_dataset(data, cache=cache, transform=transform, **args)
        dataset.append_transform(self.vocabs)
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset, 'token_input_ids', 'token'),
                                                     shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return TransformerTagger.compute_loss(self, criterion, output, batch['tag_id'], batch['mask'])

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder,
                      **kwargs) -> Union[Dict[str, Any], Any]:
        return TransformerTagger.decode_output(self, output, mask, batch, decoder)

    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        return TransformerTagger.update_metrics(self, metric, output, batch['tag_id'], batch['mask'])

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return LinearCRFDecoder(encoder_size, len(self.vocabs['tag']), self.config.crf)

    def build_metric(self, **kwargs):
        return TransformerTagger.build_metric(self, **kwargs)

    def input_is_flat(self, data) -> bool:
        return TransformerTagger.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> Union[List, Dict]:
        return TransformerTagger.prediction_to_human(self, prediction, self.vocabs['tag'].idx_to_token, batch)

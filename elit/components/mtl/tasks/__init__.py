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
import os
import warnings
from abc import ABC, abstractmethod
from copy import copy
from typing import Callable, Dict, Any, Union, Iterable, List

import torch
from torch.utils.data import DataLoader

from elit.common.constant import BOS, EOS
from elit.common.dataset import SamplerBuilder, SortingSamplerBuilder, TransformDataset, KMeansSamplerBuilder
from elit.common.document import Document
from elit.common.structure import ConfigTracker
from elit.common.torch_component import TorchComponent
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.metrics.metric import Metric
from elit.metrics.mtl import MetricDict
from elit.transform.transformer_tokenizer import TransformerSequenceTokenizer
from elit.utils.time_util import CountdownTimer


class Task(ConfigTracker, TorchComponent, ABC):
    # noinspection PyMissingConstructor
    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=None,
                 separate_optimizer=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 **kwargs) -> None:
        """
        A task in the multi-task learning framework

        Args:
            trn: Path to training set.
            dev: Path to dev set.
            tst: Path to test set.
            sampler_builder: A builder which builds a sampler.
            dependencies: Its dependencies on other tasks.
            scalar_mix: A builder which builds a `ScalarMixWithDropout` object.
            use_raw_hidden_states: Whether use raw hidden states from transformer without any pooling.
            lr: Learning rate for this task.
            separate_optimizer: Use customized separate optimizer for this task.
            **kwargs: Not used.
        """
        ConfigTracker.__init__(self, locals())
        for f, n in zip([trn, dev, tst], ['trn', 'dev', 'tst']):
            if f and os.path.isfile(f):  # anonymize local file names
                self.config.pop(n)
        self.separate_optimizer = separate_optimizer
        self.lr = lr
        self.use_raw_hidden_states = use_raw_hidden_states
        if sampler_builder is None:
            sampler_builder = SortingSamplerBuilder(batch_size=32)
        self.sampler_builder: Union[SortingSamplerBuilder, KMeansSamplerBuilder] = sampler_builder
        self.dependencies = dependencies
        self.tst = tst
        self.dev = dev
        self.trn = trn
        self.scalar_mix = scalar_mix
        self.cls_is_bos = cls_is_bos
        self.sep_is_eos = sep_is_eos

    @property
    def name(self) -> str:
        if isinstance(self.group, str):
            return self.group
        return '/'.join(self.group)

    @abstractmethod
    def build_dataloader(self,
                         data,
                         transform: Callable = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         cache=False,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        """
        Build a dataloader for training or evaluation.

        Args:
            data:
            transform: The transform from MTL, which is usually [TransformerSequenceTokenizer, FieldLength('token')]
            training: Whether this method is called on training set.
            device: The device dataloader is intended to work with.
            logger: Logger for printing message indicating progress.
            cache: Whether the dataloader should be cached.
            gradient_accumulation: Gradient accumulation to be passed to sampler builder.
            **kwargs: Additional experimental arguments.
        """
        pass

    def build_optimizer(self, decoder: torch.nn.Module, **kwargs):
        pass

    def build_batch_wise_scheduler(self, decoder: torch.nn.Module, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion,
                     ) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        pass

    @abstractmethod
    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any], decoder: torch.nn.Module, **kwargs) -> Union[Dict[str, Any], Any]:
        pass

    @abstractmethod
    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        pass

    # noinspection PyMethodOverriding
    @abstractmethod
    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_metric(self, **kwargs):
        pass

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, output=False, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, **kwargs):
        pass

    # noinspection PyMethodMayBeStatic
    def compute_lens(self, data: Union[List[Dict[str, Any]], str], dataset: TransformDataset,
                     input_ids='token_input_ids', length_field='token'):
        """

        Args:
            data: Samples to be measured or path to dataset during training time.
            dataset: During training time, use this dataset to measure the length of each sample inside.
            input_ids: Field name corresponds to input ids.
            length_field: Fall back to this field during prediction as input_ids may not be generated yet.

        Returns:

            Length list of this samples

        """
        if isinstance(data, str):
            if not dataset.cache:
                warnings.warn(f'Caching for the dataset is not enabled, '
                              f'try `dataset.purge_cache()` if possible. The dataset is {dataset}.')
            timer = CountdownTimer(len(dataset))
            for each in dataset:
                timer.log('Preprocessing and caching samples [blink][yellow]...[/yellow][/blink]')
            timer.erase()
            return [len(x[input_ids]) for x in dataset]
        return [len(x[length_field]) for x in data]

    def feed_batch(self,
                   h: torch.FloatTensor,
                   batch: Dict[str, torch.Tensor],
                   mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        return decoder(h, batch=batch, mask=mask)

    def input_is_flat(self, data) -> bool:
        """
        Check whether the data is flat

        Returns:
            bool: is_flat
        """
        raise NotImplementedError(
            '`input_is_flat()` needs to be implemented for the task component to accept raw input from user.'
        )

    @abstractmethod
    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def transform_batch(self,
                        batch: Dict[str, Any],
                        results: Dict[str, Any] = None,
                        cls_is_bos=False,
                        sep_is_eos=False) -> Dict[str, Any]:
        """
        Let the task transform the batch

        Args:
            batch:
            cls_is_bos:
            sep_is_eos:

        Returns:

        """
        if cls_is_bos != self.cls_is_bos or sep_is_eos != self.sep_is_eos:
            batch = copy(batch)
            tokens = []
            for sent in batch['token']:
                if cls_is_bos:
                    if not self.cls_is_bos:
                        sent = sent[1:]
                elif self.cls_is_bos:
                    sent = [BOS] + sent
                if sep_is_eos:
                    if not self.sep_is_eos:
                        sent = sent[:-1]
                elif self.sep_is_eos:
                    sent = sent + [EOS]
                tokens.append(sent)
            delta = len(tokens[0]) - len(batch['token'][0])
            batch['token_length'] = batch['token_length'] + delta
            batch['token'] = tokens
        return batch

    # noinspection PyMethodMayBeStatic
    def build_samples(self, inputs, cls_is_bos=False, sep_is_eos=False):
        if cls_is_bos:
            inputs = [[BOS] + x for x in inputs]
        if sep_is_eos:
            inputs = [x + [EOS] for x in inputs]
        return [{'token': token} for token in inputs]

    def build_tokenizer(self, tokenizer: TransformerSequenceTokenizer):
        if tokenizer.cls_is_bos != self.cls_is_bos or tokenizer.sep_is_eos != self.sep_is_eos:
            tokenizer = copy(tokenizer)
            tokenizer.cls_is_bos = self.cls_is_bos
            tokenizer.sep_is_eos = self.sep_is_eos
        return tokenizer

    # noinspection PyMethodMayBeStatic
    def finalize_document(self, doc: Document, task_name: str):
        pass

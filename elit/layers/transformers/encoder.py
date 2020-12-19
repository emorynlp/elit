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
from typing import Union, Dict, Any

import torch
from torch import nn
from elit.layers.dropout import WordDropout
from elit.layers.scalar_mix import ScalarMixWithDropout, ScalarMixWithDropoutBuilder
from elit.layers.transformers.pt_imports import PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoModel, \
    AutoTokenizer, AutoModel_
from elit.layers.transformers.utils import transformer_encode


class TransformerEncoder(nn.Module):
    def __init__(self,
                 transformer: Union[PreTrainedModel, str],
                 transformer_tokenizer: PreTrainedTokenizer,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout=None,
                 max_sequence_length=None,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 trainable=True,
                 training=True) -> None:
        """

        Parameters
        ----------
        transformer
        transformer_tokenizer
        average_subwords
        scalar_mix:
            If scalar_mix is int, it means this encoder will output hidden states from scalar_mix-th to the last layer.
            For example, `scalar_mix=0` means to output hidden states from all layers.
        word_dropout
        hidden_dropout
        layer_dropout
        max_sequence_length
        training
        """
        super().__init__()
        self.ret_raw_hidden_states = ret_raw_hidden_states
        self.max_sequence_length = max_sequence_length
        self.average_subwords = average_subwords
        if word_dropout:
            oov = transformer_tokenizer.mask_token_id
            if isinstance(word_dropout, list):
                word_dropout, replacement = word_dropout
                if replacement == 'unk':
                    # Electra English has to use unk
                    oov = transformer_tokenizer.unk_token_id
                elif replacement == 'mask':
                    # UDify uses [MASK]
                    oov = transformer_tokenizer.mask_token_id
                else:
                    oov = replacement
            pad = transformer_tokenizer.pad_token_id
            cls = transformer_tokenizer.cls_token_id
            sep = transformer_tokenizer.sep_token_id
            excludes = [pad, cls, sep]
            self.word_dropout = WordDropout(p=word_dropout, oov_token=oov, exclude_tokens=excludes)
        else:
            self.word_dropout = None
        if isinstance(transformer, str):
            output_hidden_states = scalar_mix is not None
            if transformer_args is None:
                transformer_args = dict()
            transformer_args['output_hidden_states'] = output_hidden_states
            transformer = AutoModel_.from_pretrained(transformer, training=training or not trainable,
                                                     **transformer_args)
        self.transformer = transformer
        if not trainable:
            transformer.requires_grad_(False)

        if isinstance(scalar_mix, ScalarMixWithDropoutBuilder):
            self.scalar_mix: ScalarMixWithDropout = scalar_mix.build()
        else:
            self.scalar_mix = None

    def forward(self, input_ids: torch.LongTensor, attention_mask=None, token_type_ids=None, token_span=None, **kwargs):
        if self.word_dropout:
            input_ids = self.word_dropout(input_ids)

        x = transformer_encode(self.transformer,
                               input_ids,
                               attention_mask,
                               token_type_ids,
                               token_span,
                               layer_range=self.scalar_mix.mixture_range if self.scalar_mix else 0,
                               max_sequence_length=self.max_sequence_length,
                               average_subwords=self.average_subwords,
                               ret_raw_hidden_states=self.ret_raw_hidden_states)
        if self.ret_raw_hidden_states:
            x, raw_hidden_states = x
        if self.scalar_mix:
            x = self.scalar_mix(x)
        if self.ret_raw_hidden_states:
            # noinspection PyUnboundLocalVariable
            return x, raw_hidden_states
        return x

    @staticmethod
    def build_transformer(config, training=True) -> PreTrainedModel:
        kwargs = {}
        if config.scalar_mix and config.scalar_mix > 0:
            kwargs['output_hidden_states'] = True
        transformer = AutoModel_.from_pretrained(config.transformer, training=training, **kwargs)
        return transformer

    @staticmethod
    def build_transformer_tokenizer(config_or_str, use_fast=True, do_basic_tokenize=True) -> PreTrainedTokenizer:
        if isinstance(config_or_str, str):
            transformer = config_or_str
        else:
            transformer = config_or_str.transformer
        if use_fast and not do_basic_tokenize:
            warnings.warn('`do_basic_tokenize=False` might not work when `use_fast=True`')
        return AutoTokenizer.from_pretrained(transformer, use_fast=use_fast, do_basic_tokenize=do_basic_tokenize)

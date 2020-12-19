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
from typing import Optional, Union, List, Any, Dict

import torch
from torch import nn

from elit.common.structure import AutoConfigurable
from elit.layers.embeddings.embedding import Embedding
from elit.layers.scalar_mix import ScalarMixWithDropoutBuilder
from elit.layers.transformers.encoder import TransformerEncoder
from elit.layers.transformers.pt_imports import PreTrainedTokenizer, AutoConfig
from elit.transform.transformer_tokenizer import TransformerSequenceTokenizer


class ContextualWordEmbeddingModule(TransformerEncoder):
    def __init__(self,
                 field: str,
                 transformer: str,
                 transformer_tokenizer: PreTrainedTokenizer,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout=None,
                 max_sequence_length=None,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 trainable=True,
                 training=True) -> None:
        super().__init__(transformer, transformer_tokenizer, average_subwords, scalar_mix, word_dropout,
                         max_sequence_length, ret_raw_hidden_states, transformer_args, trainable,
                         training)
        self.field = field

    # noinspection PyMethodOverriding
    # noinspection PyTypeChecker
    def forward(self, batch: dict, mask=None, **kwargs):
        input_ids: torch.LongTensor = batch[f'{self.field}_input_ids']
        token_span: torch.LongTensor = batch.get(f'{self.field}_token_span', None)
        # input_device = input_ids.device
        # this_device = self.get_device()
        # if input_device != this_device:
        #     input_ids = input_ids.to(this_device)
        #     token_span = token_span.to(this_device)
        # We might want to apply mask here
        output: Union[torch.Tensor, List[torch.Tensor]] = super().forward(input_ids, token_span=token_span, **kwargs)
        # if input_device != this_device:
        #     if isinstance(output, torch.Tensor):
        #         output = output.to(input_device)
        #     else:
        #         output = [x.to(input_device) for x in output]
        return output

    def get_output_dim(self):
        return self.transformer.config.hidden_size

    def get_device(self):
        device: torch.device = next(self.parameters()).device
        return device


class ContextualWordEmbedding(Embedding, AutoConfigurable):
    def __init__(self, field: str,
                 transformer: str,
                 average_subwords=False,
                 scalar_mix: Union[ScalarMixWithDropoutBuilder, int] = None,
                 word_dropout=None,
                 max_sequence_length=None,
                 truncate_long_sequences=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 ret_token_span=True,
                 ret_subtokens=False,
                 ret_subtokens_group=False,
                 ret_prefix_mask=False,
                 ret_raw_hidden_states=False,
                 transformer_args: Dict[str, Any] = None,
                 use_fast=True,
                 do_basic_tokenize=True,
                 trainable=True) -> None:
        super().__init__()
        self.truncate_long_sequences = truncate_long_sequences
        self.transformer_args = transformer_args
        self.trainable = trainable
        self.ret_subtokens_group = ret_subtokens_group
        self.ret_subtokens = ret_subtokens
        self.ret_raw_hidden_states = ret_raw_hidden_states
        self.sep_is_eos = sep_is_eos
        self.cls_is_bos = cls_is_bos
        self.max_sequence_length = max_sequence_length
        self.word_dropout = word_dropout
        self.scalar_mix = scalar_mix
        self.average_subwords = average_subwords
        self.transformer = transformer
        self.field = field
        self._transformer_tokenizer = TransformerEncoder.build_transformer_tokenizer(self.transformer,
                                                                                     use_fast=use_fast,
                                                                                     do_basic_tokenize=do_basic_tokenize)
        self._tokenizer_transform = TransformerSequenceTokenizer(self._transformer_tokenizer,
                                                                 field,
                                                                 truncate_long_sequences=truncate_long_sequences,
                                                                 ret_prefix_mask=ret_prefix_mask,
                                                                 ret_token_span=ret_token_span,
                                                                 cls_is_bos=cls_is_bos,
                                                                 sep_is_eos=sep_is_eos,
                                                                 ret_subtokens=ret_subtokens,
                                                                 ret_subtokens_group=ret_subtokens_group,
                                                                 max_seq_length=self.max_sequence_length
                                                                 )

    def transform(self, **kwargs) -> TransformerSequenceTokenizer:
        return self._tokenizer_transform

    def module(self, training=True, **kwargs) -> Optional[nn.Module]:
        return ContextualWordEmbeddingModule(self.field,
                                             self.transformer,
                                             self._transformer_tokenizer,
                                             self.average_subwords,
                                             self.scalar_mix,
                                             self.word_dropout,
                                             self.max_sequence_length,
                                             self.ret_raw_hidden_states,
                                             self.transformer_args,
                                             self.trainable,
                                             training=training)

    def get_output_dim(self):
        config = AutoConfig.from_pretrained(self.transformer)
        return config.hidden_size

    def get_tokenizer(self):
        return self._transformer_tokenizer


def find_transformer(embed: nn.Module):
    if isinstance(embed, ContextualWordEmbeddingModule):
        return embed
    if isinstance(embed, nn.ModuleList):
        for child in embed:
            found = find_transformer(child)
            if found:
                return found

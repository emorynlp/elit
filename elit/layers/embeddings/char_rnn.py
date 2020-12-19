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
from typing import Optional, Callable, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from elit.common.structure import AutoConfigurable
from elit.common.transform import VocabDict, ToChar
from elit.common.vocab import Vocab
from elit.layers.embeddings.embedding import Embedding, EmbeddingDim


class CharRNN(nn.Module, EmbeddingDim):
    def __init__(self,
                 field,
                 vocab_size,
                 embed: Union[int, nn.Embedding],
                 hidden_size):
        super(CharRNN, self).__init__()
        self.field = field
        # the embedding layer
        if isinstance(embed, int):
            self.embed = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed)
        elif isinstance(embed, nn.Module):
            self.embed = embed
            embed = embed.embedding_dim
        else:
            raise ValueError(f'Unrecognized type for {embed}')
        # the lstm layer
        self.lstm = nn.LSTM(input_size=embed,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, batch, mask, **kwargs):
        x = batch[f'{self.field}_char_id']
        # [batch_size, seq_len, fix_len]
        mask = x.ne(0)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        char_mask = lens.gt(0)

        # [n, fix_len, n_embed]
        x = self.embed(x[char_mask])
        x = pack_padded_sequence(x, lens[char_mask], True, False)
        x, (h, _) = self.lstm(x)
        # [n, fix_len, n_out]
        h = torch.cat(torch.unbind(h), -1)
        # [batch_size, seq_len, n_out]
        embed = h.new_zeros(*lens.shape, h.size(-1))
        embed = embed.masked_scatter_(char_mask.unsqueeze(-1), h)

        return embed

    @property
    def embedding_dim(self) -> int:
        return self.lstm.hidden_size * 2


class CharRNNEmbedding(Embedding, AutoConfigurable):
    def __init__(self,
                 field,
                 embed,
                 hidden_size,
                 max_word_length=None) -> None:
        super().__init__()
        self.field = field
        self.hidden_size = hidden_size
        self.embed = embed
        self.max_word_length = max_word_length

    def transform(self, vocabs: VocabDict, **kwargs) -> Optional[Callable]:
        if isinstance(self.embed, Embedding):
            self.embed.transform(vocabs=vocabs)
        vocab_name = self.vocab_name
        if vocab_name not in vocabs:
            vocabs[vocab_name] = Vocab()
        return ToChar(self.field, vocab_name, max_word_length=self.max_word_length)

    @property
    def vocab_name(self):
        vocab_name = f'{self.field}_char'
        return vocab_name

    def module(self, vocabs: VocabDict, **kwargs) -> Optional[nn.Module]:
        embed = self.embed
        if isinstance(self.embed, Embedding):
            embed = self.embed.module(vocabs=vocabs)
        return CharRNN(self.field, len(vocabs[self.vocab_name]), embed, self.hidden_size)

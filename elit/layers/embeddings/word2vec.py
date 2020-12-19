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
from torch import nn

from elit.common.structure import AutoConfigurable
from elit.common.transform import VocabDict
from elit.common.vocab import Vocab
from elit.layers.dropout import WordDropout
from elit.layers.embeddings.embedding import Embedding, EmbeddingDim
from elit.layers.embeddings.util import build_word2vec_with_vocab


class Word2VecEmbeddingModule(nn.Module, EmbeddingDim):
    def __init__(self, field: str, embed: nn.Embedding, word_dropout: WordDropout = None, cpu=False,
                 second_channel=False, num_tokens_in_trn=None, unk_idx=1) -> None:
        super().__init__()
        self.cpu = cpu
        self.field = field
        self.embed = embed
        self.word_dropout = word_dropout
        self.num_tokens_in_trn = num_tokens_in_trn
        self.unk_idx = unk_idx
        if second_channel:
            n_words, n_embed = embed.weight.size()
            if num_tokens_in_trn:
                n_words = num_tokens_in_trn
            second_channel = nn.Embedding(num_embeddings=n_words,
                                          embedding_dim=n_embed)
            nn.init.zeros_(second_channel.weight)
        self.second_channel = second_channel

    def forward(self, batch: dict, **kwargs):
        x: torch.Tensor = batch[f'{self.field}_id']
        if self.cpu:
            device = x.device
            x = x.cpu()
        if self.word_dropout:
            x = self.word_dropout(x)
        if self.second_channel:
            ext_mask = x.ge(self.second_channel.num_embeddings)
            ext_words = x.masked_fill(ext_mask, self.unk_idx)
            x = self.embed(x) + self.second_channel(ext_words)
        else:
            x = self.embed(x)
        if self.cpu:
            # noinspection PyUnboundLocalVariable
            x = x.to(device)
        return x

    @property
    def embedding_dim(self) -> int:
        return self.embed.embedding_dim

    # noinspection PyMethodOverriding
    # def to(self, device, **kwargs):
    #     print(self.cpu)
    #     exit(1)
    #     if self.cpu:
    #         return super(Word2VecEmbeddingModule, self).to(-1, **kwargs)
    #     return super(Word2VecEmbeddingModule, self).to(device, **kwargs)

    def _apply(self, fn):

        if not self.cpu:  # This might block all fn not limiting to moving between devices.
            return super(Word2VecEmbeddingModule, self)._apply(fn)


class Word2VecEmbedding(Embedding, AutoConfigurable):
    def __init__(self,
                 field,
                 embed: Union[int, str],
                 extend_vocab=True,
                 pad=None,
                 unk=None,
                 lowercase=False,
                 trainable=False,
                 second_channel=False,
                 word_dropout: float = 0,
                 normalize=False,
                 cpu=False,
                 init='zeros') -> None:
        super().__init__()
        self.pad = pad
        self.second_channel = second_channel
        self.cpu = cpu
        self.normalize = normalize
        self.word_dropout = word_dropout
        self.init = init
        self.lowercase = lowercase
        self.unk = unk
        self.extend_vocab = extend_vocab
        self.trainable = trainable
        self.embed = embed
        self.field = field

    def module(self, vocabs: VocabDict, **kwargs) -> Optional[nn.Module]:
        vocab = vocabs[self.field]
        num_tokens_in_trn = len(vocab)
        embed = build_word2vec_with_vocab(self.embed,
                                          vocab,
                                          self.extend_vocab,
                                          self.unk,
                                          self.lowercase,
                                          self.trainable,
                                          normalize=self.normalize)
        if self.word_dropout:
            assert vocab.unk_token, f'unk_token of vocab {self.field} has to be set in order to ' \
                                    f'make use of word_dropout'
            padding = []
            if vocab.pad_token:
                padding.append(vocab.pad_idx)
            word_dropout = WordDropout(self.word_dropout, vocab.unk_idx, exclude_tokens=padding)
        else:
            word_dropout = None
        return Word2VecEmbeddingModule(self.field, embed, word_dropout=word_dropout, cpu=self.cpu,
                                       second_channel=self.second_channel, num_tokens_in_trn=num_tokens_in_trn,
                                       unk_idx=vocab.unk_idx)

    def transform(self, vocabs: VocabDict = None, **kwargs) -> Optional[Callable]:
        assert vocabs is not None
        if self.field not in vocabs:
            vocabs[self.field] = Vocab(pad_token=self.pad, unk_token=self.unk)
        return super().transform(**kwargs)


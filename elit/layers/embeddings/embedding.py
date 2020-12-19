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
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Optional, Iterable

import torch
from torch import nn
from torch.nn import Module

from elit.common.structure import AutoConfigurable
from elit.common.transform import TransformList
from elit.layers.dropout import IndependentDropout


class EmbeddingDim(ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        return -1

    def get_output_dim(self) -> int:
        return self.embedding_dim


class Embedding(AutoConfigurable, ABC):
    def transform(self, **kwargs) -> Optional[Callable]:
        return None

    def module(self, **kwargs) -> Optional[nn.Module]:
        return None


class ConcatModuleList(nn.ModuleList, EmbeddingDim):

    def __init__(self, *modules: Optional[Iterable[Module]], dropout=None) -> None:
        super().__init__(*modules)
        if dropout:
            dropout = IndependentDropout(p=dropout)
        self.dropout = dropout

    @property
    def embedding_dim(self) -> int:
        """This method causes PyTorch throw AttributeError. We are looking for a solution.
        :return:

        Args:

        Returns:

        """
        return sum(embed.embedding_dim for embed in self)

    def get_output_dim(self) -> int:
        return sum(embed.get_output_dim() for embed in self)

    # noinspection PyMethodOverriding
    def forward(self, batch: dict, **kwargs):
        embeds = [embed(batch, **kwargs) for embed in self.embeddings]
        if self.dropout:
            embeds = self.dropout(*embeds)
        return torch.cat(embeds, -1)

    @property
    def embeddings(self):
        embeddings = [x for x in self]
        if self.dropout:
            embeddings.remove(self.dropout)
        return embeddings


class EmbeddingList(Embedding):
    def __init__(self, *embeddings_, embeddings: dict = None, dropout=None) -> None:
        # noinspection PyTypeChecker
        self.dropout = dropout
        self._embeddings: List[Embedding] = list(embeddings_)
        if embeddings:
            for each in embeddings:
                if isinstance(each, dict):
                    each = AutoConfigurable.from_config(each)
                self._embeddings.append(each)
        self.embeddings = [e.config for e in self._embeddings]

    def transform(self, **kwargs):
        transforms = [e.transform(**kwargs) for e in self._embeddings]
        transforms = [t for t in transforms if t]
        return TransformList(*transforms)

    def module(self, **kwargs):
        modules = [e.module(**kwargs) for e in self._embeddings]
        modules = [m for m in modules if m]
        return ConcatModuleList(modules, dropout=self.dropout)

    def to_list(self):
        return self._embeddings


def find_embedding_by_class(embed: Embedding, cls):
    if isinstance(embed, cls):
        return embed
    if isinstance(embed, EmbeddingList):
        for child in embed.to_list():
            found = find_embedding_by_class(child, cls)
            if found:
                return found

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
import os
import sys
from typing import Optional, Callable

import fasttext
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from elit.common.structure import AutoConfigurable
from elit.common.transform import EmbeddingNamedTransform
from elit.layers.embeddings.embedding import Embedding, EmbeddingDim
from elit.utils.io_util import get_resource, stdout_redirected
from elit.utils.log_util import flash


class FastTextTransform(EmbeddingNamedTransform):
    def __init__(self, filepath: str, src, dst=None, **kwargs) -> None:
        if not dst:
            dst = src + '_fasttext'
        self.filepath = filepath
        flash(f'Loading fasttext model {filepath} [blink][yellow]...[/yellow][/blink]')
        filepath = get_resource(filepath)
        with stdout_redirected(to=os.devnull, stdout=sys.stderr):
            self._model = fasttext.load_model(filepath)
        flash('')
        output_dim = self._model['king'].size
        super().__init__(output_dim, src, dst)

    def __call__(self, sample: dict):
        word = sample[self.src]
        if isinstance(word, str):
            vector = self.embed(word)
        else:
            vector = torch.stack([self.embed(each) for each in word])
        sample[self.dst] = vector
        return sample

    def embed(self, word: str):
        return torch.tensor(self._model[word])


class PassThroughModule(torch.nn.Module):
    def __init__(self, key) -> None:
        super().__init__()
        self.key = key

    def __call__(self, batch: dict, mask=None, **kwargs):
        return batch[self.key]


class FastTextEmbeddingModule(PassThroughModule):

    def __init__(self, key, embedding_dim: int) -> None:
        super().__init__(key)
        self.embedding_dim = embedding_dim

    def __call__(self, batch: dict, mask=None, **kwargs):
        outputs = super().__call__(batch, **kwargs)
        outputs = pad_sequence(outputs, True, 0).to(mask.device)
        return outputs

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'key={self.key}, embedding_dim={self.embedding_dim}'
        s += ')'
        return s

    def get_output_dim(self):
        return self.embedding_dim


class FastTextEmbedding(Embedding, AutoConfigurable):
    def __init__(self, src: str, filepath: str) -> None:
        super().__init__()
        self.src = src
        self.filepath = filepath
        self._fasttext = FastTextTransform(self.filepath, self.src)

    def transform(self, **kwargs) -> Optional[Callable]:
        return self._fasttext

    def module(self, **kwargs) -> Optional[nn.Module]:
        return FastTextEmbeddingModule(self._fasttext.dst, self._fasttext.output_dim)

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
from collections import defaultdict
from typing import List, Union

import torch

from elit.metrics.f1 import F1


class BinaryChunkingF1(F1):
    def __call__(self, pred_tags: torch.LongTensor, gold_tags: torch.LongTensor, lens: List[int] = None):
        if lens is None:
            lens = [gold_tags.size(1)] * gold_tags.size(0)
        self.update(self.decode_spans(pred_tags, lens), self.decode_spans(gold_tags, lens))

    def update(self, pred_tags, gold_tags):
        for pred, gold in zip(pred_tags, gold_tags):
            super().__call__(set(pred), set(gold))

    @staticmethod
    def decode_spans(pred_tags: torch.LongTensor, lens: Union[List[int], torch.LongTensor]):
        if isinstance(lens, torch.Tensor):
            lens = lens.tolist()
        batch_pred = defaultdict(list)
        for batch, offset in pred_tags.nonzero(as_tuple=False).tolist():
            batch_pred[batch].append(offset)
        batch_pred_spans = [[(0, l)] for l in lens]
        for batch, offsets in batch_pred.items():
            l = lens[batch]
            batch_pred_spans[batch] = list(zip(offsets, offsets[1:] + [l]))
        return batch_pred_spans

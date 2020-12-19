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
from typing import List

from elit.metrics.chunking.sequence_labeling import get_entities
from elit.metrics.f1 import F1
from elit.metrics.metric import Metric


class ChunkingF1(F1):

    def __call__(self, pred_tags: List[List[str]], gold_tags: List[List[str]]):
        for p, g in zip(pred_tags, gold_tags):
            pred = set(get_entities(p))
            gold = set(get_entities(g))
            self.nb_pred += len(pred)
            self.nb_true += len(gold)
            self.nb_correct += len(pred & gold)

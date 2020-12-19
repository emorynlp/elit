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
from typing import Union, List

from alnlp.modules import feedforward

from elit.common.structure import ConfigTracker


class FeedForward(feedforward.FeedForward, ConfigTracker):
    def __init__(self, input_dim: int, num_layers: int, hidden_dims: Union[int, List[int]],
                 activations: Union[str, List[str]], dropout: Union[float, List[float]] = 0.0) -> None:
        super().__init__(input_dim, num_layers, hidden_dims, activations, dropout)
        ConfigTracker.__init__(self, locals())

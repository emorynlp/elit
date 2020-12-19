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
import math

import torch
from torch import nn
import functools


def embedding_uniform(tensor:torch.Tensor, seed=233):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        fan_out = tensor.size(-1)
        bound = math.sqrt(3.0 / fan_out)
        return tensor.uniform_(-bound, bound, generator=gen)

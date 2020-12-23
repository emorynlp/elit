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
# Author: Liyan Xu
from copy import deepcopy
import torch
from transformers import BertTokenizer


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_vocab_size(config):
    tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_name'])
    if config['add_speaker_token']:
        return len(tokenizer) + config['max_num_speakers'] + 1
    else:
        return len(tokenizer)


def bucket_distance(offsets):
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def batch_select(tensor, idx, device=torch.device('cpu')):
    assert tensor.shape[0] == idx.shape[0]
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]

    if tensor.shape[-1] == 1:
        selected = torch.squeeze(selected, -1)

    return selected


def random_select(tensor, num_selection):
    if tensor.shape[0] > num_selection:
        rand_idx = torch.randperm(tensor.shape[0])[:num_selection]
        return tensor[rand_idx]
    else:
        return tensor

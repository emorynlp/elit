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
# Author: hankcs, Liyan Xu
from typing import Union, List, Tuple
from pydantic import BaseModel


class OnlineCorefInput(BaseModel):
    # Required
    inputs_ids: List[int]
    sentence_map: List[int]
    subtoken_map: List[int]
    # Optional
    speaker_ids: List[int] = None
    genre: str = None
    uttr_start_idx: List[int] = None
    mentions: List[Tuple[int]] = None


class Input(BaseModel):
    text: Union[str, List[str]] = None
    tokens: List[List[str]] = None
    context_coref: OnlineCorefInput = None
    models: List[str] = ["lem", "pos", "ner", "con", "dep", "srl", "amr", "dcr", "ocr"]
    language: str = 'en'
    verbose: bool = True

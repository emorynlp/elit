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
from typing import List, Tuple
from alnlp.modules.conditional_random_field import allowed_transitions


def guess_tagging_scheme(labels: List[str]) -> str:
    tagset = set(y.split('-')[0] for y in labels)
    for scheme in "BIO", "BIOUL", "BMES", 'IOBES':
        if tagset == set(list(scheme)):
            return scheme


def guess_allowed_transitions(labels) -> List[Tuple[int, int]]:
    scheme = guess_tagging_scheme(labels)
    if not scheme:
        return None
    if scheme == 'IOBES':
        scheme = 'BIOUL'
        labels = [y.replace('E-', 'L-').replace('S-', 'U-') for y in labels]
    return allowed_transitions(scheme, dict(enumerate(labels)))

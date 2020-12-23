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
from typing import Union, List, Tuple


class CorefOutput:
    def __init__(self,
                 clusters: List[List[Tuple[int]]],
                 input_ids: List[int] = None,
                 sentence_map: List[int] = None,
                 subtoken_map: List[int] = None,
                 mentions: List[Tuple[int]] = None,
                 speaker_ids: List[int] = None,
                 uttr_start_idx: List[int] = None
                 ):
        """
        Coreference output model.

        For online coreference, this output will be used in next utterance's input.
        Args:
            clusters (): by global original token indices (cross-utterance for online)
            input_ids (): by subtoken
            sentence_map (): by subtoken; map to global sentence indices (cross-utterance for online)
            subtoken_map (): by subtoken; map to global original token indices (cross-utterance for online)
            mentions (): by subtoken
            speaker_ids (): by subtoken
            uttr_start_idx (): by subtoken
        """
        self.input_ids = input_ids
        self.sentence_map = sentence_map
        self.subtoken_map = subtoken_map
        self.mentions = mentions
        self.clusters = clusters
        self.speaker_ids = speaker_ids
        self.uttr_start_idx = uttr_start_idx


class CorefInput:
    def __init__(self,
                 doc_or_uttr: List[List[str]],
                 speaker_ids: Union[int, List[int]] = None,
                 context: CorefOutput = None):
        """
        Coreference input model.

        Args:
            doc_or_uttr (): one entire document or one utterance; already tokenized
            speaker_ids (): single speaker id for the utterance or document, or list of speakers for each sentence
            context (): the output of previous utterance for online coreference
        """
        self.doc_or_uttr = doc_or_uttr
        self.speaker_ids = speaker_ids
        self.context = context

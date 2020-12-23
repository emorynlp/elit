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
from typing import Union, List, Tuple, Dict


class CorefOutput:
    def __init__(self,
                 clusters: List[List[Tuple[int]]] = None,
                 input_ids: List[int] = None,
                 sentence_map: List[int] = None,
                 subtoken_map: List[int] = None,
                 mentions: List[Tuple[int, int]] = None,
                 uttr_start_idx: List[int] = None,
                 speaker_ids: List[int] = None,
                 linking_prob: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = None,
                 error_msg: str = None
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
            uttr_start_idx (): by subtoken
            speaker_ids (): by subtoken
            linking_prob (): by global original token indices, same as clusters,
            error_msg ():
        """
        self.clusters = clusters
        self.input_ids = input_ids
        self.sentence_map = sentence_map
        self.subtoken_map = subtoken_map
        self.mentions = mentions
        self.uttr_start_idx = uttr_start_idx
        self.speaker_ids = speaker_ids
        self.linking_prob = linking_prob
        self.error_msg = error_msg

    def __len__(self):
        return 0 if self.input_ids is None else len(self.input_ids)

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(self)


class CorefInput:
    def __init__(self,
                 doc_or_uttr: List[List[str]],
                 speaker_ids: Union[int, List[int]] = None,
                 genre: str = None,
                 context: CorefOutput = None,
                 return_prob: bool = False,
                 language: str = 'en',
                 verbose: bool = True
    ):
        """
        Coreference input model.

        Args:
            doc_or_uttr (): one entire document or one utterance; already tokenized
            speaker_ids (): speaker id for the utterance or document, or list of ids for each sentence; id starts from 1
            genre (): see :meth:`elit.components.coref.coref_resolver.CoreferenceResolver.available_genres`
            context (): see :meth:`elit.components.coref.dto.CorefOutput.prepare_as_next_online_context`
            return_prob (): whether return the mention linking probability
            language ():
            verbose ():
        """
        self.doc_or_uttr = doc_or_uttr
        self.speaker_ids = speaker_ids
        self.genre = genre
        self.context = context
        self.return_prob = return_prob
        self.language = language
        self.verbose = verbose

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(self)

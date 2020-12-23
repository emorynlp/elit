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
from transformers import BertTokenizer
import torch
import logging
from typing import List, Tuple

from elit.components.coref.io import CorefInput, CorefOutput

logger = logging.getLogger(__name__)


class CorefInstance:
    """ Tensorized coreference instance consumed by the model directly. """
    def __init__(self,
                 input_ids: torch.Tensor = None,
                 input_mask: torch.Tensor = None,
                 sentence_map: torch.Tensor = None,
                 subtoken_map: List[int] = None,
                 speaker_ids: torch.Tensor = None,
                 genre_id: torch.Tensor = None,
                 uttr_start_idx: List[int] = None,
                 mentions: List[Tuple[int]] = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.sentence_map = sentence_map
        self.subtoken_map = subtoken_map
        self.speaker_ids = speaker_ids
        self.genre_id = genre_id
        self.uttr_start_idx = uttr_start_idx

        # To be updated during prediction
        self.mentions = mentions
        self.clusters = None
        self.mention_to_cluster_id = {}

    def generate_output(self, verbose: bool = True) -> CorefOutput:
        """
        Helper method to generate corresponding coreference output.

        Args:
            verbose (): must be true for online coreference to keep context

        Returns:

        """
        if verbose:
            output = CorefOutput(
                clusters=self.clusters,
                input_ids=self.input_ids.tolist(),
                sentence_map=self.sentence_map.tolist(),
                subtoken_map=self.subtoken_map,
                speaker_ids=self.speaker_ids.tolist(),
                uttr_start_idx=self.uttr_start_idx,
                mentions=self.mentions
            )
        else:
            output = CorefOutput(clusters=self.clusters)
        return output

    def __len__(self):
        return 0 if self.input_ids is None else self.input_ids.shape[0]


class Tensorizer:
    def __init__(self, config):
        self.max_segment_len = config['max_segment_len']
        self.add_speaker_token = config['add_speaker_token']
        self.use_speaker_indicator = config['use_speaker_indicator']
        self.add_sep_token = config['add_sep_token']
        self.max_speakers = config['max_num_speakers']
        self.genres = config['genres']
        self.genre_dict = {genre: idx for idx, genre in enumerate(self.genres)}

        self.tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_name'])
        if self.add_speaker_token:
            self.tokenizer.add_tokens([self.get_speaker_token(i) for i in range(self.max_speakers + 1)],
                                      special_tokens=True)

    def get_speaker_token(self, speaker_id):
        return f'[SPK{speaker_id}]'

    def available_genres(self):
        return self.genres[:]

    def encode_doc(self):
        raise NotImplementedError()

    def encode_online(self, utterance, speaker_id=None, genre=None, context_inst=None):
        """
        utterance: tokenized string
        speaker_id: 1-max_speakers
        genre: see available_genres()
        """
        tokenizer = self.tokenizer

        def create_inst(input_ids, sentence_map, subtoken_map, speaker_ids, uttr_start_idx, genre, mentions):
            """ Input: without considering CLS, SEP """
            inst = CorefInstance(
                input_ids=torch.tensor([tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id],
                                       dtype=torch.long),
                input_mask=torch.tensor([1] * (len(input_ids) + 2), dtype=torch.long),
                sentence_map=torch.tensor([sentence_map[0]] + sentence_map + [sentence_map[-1] + 1], dtype=torch.long),
                subtoken_map=([subtoken_map[0]] + subtoken_map + [subtoken_map[-1]]),
                speaker_ids=torch.tensor(([speaker_ids[0]] + speaker_ids + [speaker_ids[-1]]) if speaker_ids else [],
                                         dtype=torch.long),
                uttr_start_idx=[idx + 1 for idx in uttr_start_idx],
                genre_id=torch.tensor(self.genre_dict[genre], dtype=torch.long),
                mentions=[(m[0] + 1, m[1] + 1) for m in mentions]
            )
            return inst

        sentence_idx, token_idx = 0, 0
        if context_inst is not None:
            sentence_idx = context_inst.sentence_map[-1].item()
            token_idx = context_inst.subtoken_map[-1] + 1
        if not speaker_id:
            speaker_id = 1
        if genre is None:
            genre = 'en' if 'en' in self.genres else self.genres[0]

        # Current utterance
        subtokens, sentence_map, subtoken_map = [], [], []
        if self.add_sep_token:
            subtokens.append(tokenizer.sep_token)
            sentence_map.append(sentence_idx)
            subtoken_map.append(token_idx)
        if self.add_speaker_token:
            subtokens.append(self.get_speaker_token(speaker_id))
            sentence_map.append(sentence_idx)
            subtoken_map.append(token_idx)
        for sent in utterance:
            for token in sent:
                subtoks = tokenizer.tokenize(token)
                subtokens += subtoks
                sentence_map += [sentence_idx] * len(subtoks)
                subtoken_map += [token_idx] * len(subtoks)
                token_idx += 1
            sentence_idx += 1
        # Truncate
        if len(subtokens) + 2 > self.max_segment_len:
            subtokens = subtokens[:self.max_segment_len - 2]
            sentence_map = sentence_map[:self.max_segment_len - 2]
            subtoken_map = subtoken_map[:self.max_segment_len - 2]

        if context_inst is None:
            inst = create_inst(
                input_ids=tokenizer.convert_tokens_to_ids(subtokens),
                sentence_map=sentence_map, subtoken_map=subtoken_map,
                speaker_ids=[speaker_id] * len(subtokens) if self.use_speaker_indicator else [],
                uttr_start_idx=[2 if self.add_sep_token else 1],
                genre=genre, mentions=[]
            )
            return inst

        # Add context
        prev_sep_idx = len(context_inst) - 1  # The middle SEP
        if self.add_sep_token:
            prev_sep_idx = context_inst.uttr_start_idx[-1] - 1
        context_start_idx = len(context_inst)
        for idx in context_inst.uttr_start_idx:
            if len(context_inst) - idx + len(subtokens) + (0 if self.add_sep_token else 1) <= self.max_segment_len:
                context_start_idx = idx
                break

        if not self.add_sep_token or context_start_idx > prev_sep_idx:
            # No SEP with any context utterances; or use SEP with at most one context utterance
            # Context token offset: no need to adjust SEP
            uttr_start_idx = [idx - context_start_idx for idx in context_inst.uttr_start_idx if
                              idx - context_start_idx >= 0]
            if len(uttr_start_idx) == 0:
                uttr_start_idx = [1 if self.add_sep_token else 0]
            else:
                if self.add_sep_token:
                    assert uttr_start_idx == [0]
                uttr_start_idx = [0, (len(context_inst) - context_start_idx - (0 if self.add_sep_token else 1))]

            mentions = []
            for m in context_inst.mentions:
                if m[0] >= context_start_idx:
                    mentions.append((m[0] - context_start_idx, m[1] - context_start_idx))

            inst = create_inst(
                input_ids=context_inst.input_ids[context_start_idx:-1].tolist() + tokenizer.convert_tokens_to_ids(
                    subtokens),
                sentence_map=context_inst.sentence_map[context_start_idx:-1].tolist() + sentence_map,
                subtoken_map=context_inst.subtoken_map[context_start_idx:-1] + subtoken_map,
                speaker_ids=context_inst.speaker_ids[context_start_idx:-1].tolist() + [speaker_id] * len(subtokens)
                if self.use_speaker_indicator else [],
                uttr_start_idx=uttr_start_idx, genre=genre, mentions=mentions
            )
        else:
            # Use SEP and have at least two context utterances
            # Context token offset: adjust SEP for last context utterance
            uttr_start_idx = [idx - context_start_idx for idx in context_inst.uttr_start_idx if
                              idx - context_start_idx >= 0]
            assert len(uttr_start_idx) > 1
            uttr_start_idx[-1] -= 1  # Adjust for SEP
            uttr_start_idx.append(len(context_inst) - context_start_idx - 1)

            mentions = []
            for m in context_inst.mentions:
                if context_start_idx <= m[0] < prev_sep_idx:
                    mentions.append((m[0] - context_start_idx, m[1] - context_start_idx))
                elif m[0] >= context_start_idx and m[0] > prev_sep_idx:
                    mentions.append((m[0] - context_start_idx - 1, m[1] - context_start_idx - 1))  # Adjust for SEP

            inst = create_inst(
                input_ids=context_inst.input_ids[context_start_idx:prev_sep_idx].tolist() +
                          context_inst.input_ids[prev_sep_idx + 1:-1].tolist() + tokenizer.convert_tokens_to_ids(
                    subtokens),
                sentence_map=context_inst.sentence_map[context_start_idx:prev_sep_idx].tolist() +
                             context_inst.sentence_map[prev_sep_idx + 1:-1].tolist() + sentence_map,
                subtoken_map=context_inst.subtoken_map[context_start_idx:prev_sep_idx] +
                             context_inst.subtoken_map[prev_sep_idx + 1:-1] + subtoken_map,
                speaker_ids=context_inst.speaker_ids[context_start_idx:prev_sep_idx].tolist() +
                            context_inst.speaker_ids[prev_sep_idx + 1:-1].tolist() + [speaker_id] * len(subtokens)
                if self.use_speaker_indicator else [],
                uttr_start_idx=uttr_start_idx, genre=genre, mentions=mentions
            )
        return inst

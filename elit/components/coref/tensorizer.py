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
from typing import List, Tuple, Union, Any

from elit.components.coref.dto import CorefInput, CorefOutput

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
                 mentions: List[Tuple[int, int]] = None):
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
        self.mention_to_cluster_id = None
        self.linking_prob = None

    def generate_output(self, tokens: List[List[str]], verbose: bool = True,
                        online: bool = False) -> Union[CorefOutput, List[List[Tuple[Any]]]]:
        """
        Helper method to generate corresponding coreference output.

        For online coreference, verbose will be set to false,
        since only client side has knowledge about full token text.
        Args:
            tokens (): input original tokens
            verbose (): true to display text in clusters
            online (): true to include context for online coreference

        Returns:

        """
        if online:
            for cluster in self.clusters:
                for i in range(len(cluster)):
                    m1, m2 = cluster[i]
                    cluster[i] = (m1, m2 + 1)
            output = CorefOutput(
                clusters=self.clusters,
                input_ids=self.input_ids.tolist(),
                sentence_map=self.sentence_map.tolist(),
                subtoken_map=self.subtoken_map,
                speaker_ids=self.speaker_ids.tolist(),
                uttr_start_idx=self.uttr_start_idx,
                mentions=self.mentions,
                linking_prob=self.linking_prob
            )
            return output

        def convert_mention_format(sent_lens, mention):
            sent_i = 1
            while sent_i < len(sent_lens) and mention[0] >= sent_lens[sent_i]:
                sent_i += 1
            offset = sent_lens[sent_i - 1]
            converted = (sent_i - 1, mention[0] - offset, mention[1] - offset + 1)
            return converted

        # Adapt ELIT format
        sent_lens = [0]
        for sent in tokens:
            sent_lens.append(len(sent) + sent_lens[-1])
        for cluster in self.clusters:
            for i in range(len(cluster)):
                cluster[i] = convert_mention_format(sent_lens, cluster[i])

        if verbose:
            for cluster in self.clusters:
                for i in range(len(cluster)):
                    sent_i, m1, m2 = cluster[i]
                    cluster[i] = (sent_i, m1, m2, ' '.join(tokens[sent_i][m1:m2]))
        return self.clusters

    def __len__(self):
        return 0 if self.input_ids is None else self.input_ids.shape[0]


class Tensorizer:
    def __init__(self, config):
        self.max_segment_len = config['max_segment_len']
        self.add_speaker_token = config['add_speaker_token']
        self.use_speaker_indicator = config['use_speaker_indicator']
        self.add_sep_token = config.get('add_sep_token', False)
        self.max_speakers = config['max_num_speakers']
        self.genres = config['genres']
        self.genre_dict = {genre: idx for idx, genre in enumerate(self.genres)}

        self.tokenizer = BertTokenizer.from_pretrained(config['bert_tokenizer_name'])
        if self.add_speaker_token:
            self.tokenizer.add_tokens([self.get_speaker_token(i) for i in range(self.max_speakers + 1)],
                                      special_tokens=True)

    def get_speaker_token(self, speaker_id):
        return f'[SPK{speaker_id}]'

    def get_default_genre(self):
        return 'en' if 'en' in self.genres else self.genres[0]

    def encode_doc(self, coref_input: CorefInput) -> CorefInstance:
        """
        Process input for document coreference resolution.

        If add speaker token, add one speaker token per sentence.
        Args:
            coref_input ():

        Returns:

        """
        tokenizer = self.tokenizer

        def create_inst(input_ids, sentence_map, subtoken_map, speaker_ids, genre):
            """ Input: without considering CLS, SEP """
            inst = CorefInstance(
                input_ids=torch.tensor([tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id],
                                       dtype=torch.long),
                input_mask=torch.tensor([1] * (len(input_ids) + 2), dtype=torch.long),
                sentence_map=torch.tensor([sentence_map[0]] + sentence_map + [sentence_map[-1] + 1], dtype=torch.long),
                subtoken_map=([subtoken_map[0]] + subtoken_map + [subtoken_map[-1]]),
                speaker_ids=torch.tensor(([speaker_ids[0]] + speaker_ids + [speaker_ids[-1]]) if speaker_ids else [],
                                         dtype=torch.long),
                uttr_start_idx=[1], genre_id=torch.tensor(self.genre_dict[genre], dtype=torch.long), mentions=[]
            )
            return inst

        doc_or_uttr, sent_speaker_ids, genre = coref_input.doc_or_uttr, coref_input.speaker_ids, coref_input.genre
        # Assign speaker id if needed
        if not sent_speaker_ids:
            sent_speaker_ids = [1] * len(doc_or_uttr)
        if isinstance(sent_speaker_ids, int):
            sent_speaker_ids = [sent_speaker_ids] * len(doc_or_uttr)
        # Assign default genre if needed
        if genre is None or genre not in self.genres:
            genre = self.get_default_genre()
        # Assign global sentence and token idx offset
        sentence_idx, token_idx = 0, 0

        # Process current doc or utterance
        subtokens, sentence_map, subtoken_map, speaker_ids = [], [], [], []
        for sent, speaker_id in zip(doc_or_uttr, sent_speaker_ids):
            if self.add_speaker_token:
                subtokens.append(self.get_speaker_token(speaker_id))
                sentence_map.append(sentence_idx)
                subtoken_map.append(token_idx)
                speaker_ids.append(speaker_id)
            for token in sent:
                subtoks = tokenizer.tokenize(token)
                subtokens += subtoks
                sentence_map += [sentence_idx] * len(subtoks)
                subtoken_map += [token_idx] * len(subtoks)
                speaker_ids += [speaker_id] * len(subtoks)
                token_idx += 1
            sentence_idx += 1
        # Truncate
        if len(subtokens) + 2 > self.max_segment_len:
            subtokens = subtokens[:self.max_segment_len - 2]
            sentence_map = sentence_map[:self.max_segment_len - 2]
            subtoken_map = subtoken_map[:self.max_segment_len - 2]

        inst = create_inst(
            input_ids=tokenizer.convert_tokens_to_ids(subtokens),
            sentence_map=sentence_map, subtoken_map=subtoken_map,
            speaker_ids=speaker_ids if self.use_speaker_indicator else [], genre=genre
        )
        return inst

    def encode_online(self, coref_input: CorefInput) -> CorefInstance:
        """
        Process input for online coreference resolution.

        If add speaker token, add one speaker token per utterance.
        Args:
            coref_input ():

        Returns:

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

        doc_or_uttr, speaker_id, genre, context = coref_input.doc_or_uttr, coref_input.speaker_ids, \
                                                  coref_input.genre, coref_input.context
        # Assign speaker id if needed; assuming same speaker for all sentences in current utterance
        if isinstance(speaker_id, list):
            speaker_id = speaker_id[0] if len(speaker_id) > 0 else 1
        if not speaker_id:
            speaker_id = 1
        # Assign default genre if needed
        if genre is None or genre not in self.genres:
            genre = self.get_default_genre()
        # Assign global sentence and token idx offset
        sentence_idx, token_idx = 0, 0
        if context is not None:
            sentence_idx = context.sentence_map[-1]
            token_idx = context.subtoken_map[-1] + 1

        # Process current doc or utterance
        subtokens, sentence_map, subtoken_map = [], [], []
        if self.add_sep_token:
            subtokens.append(tokenizer.sep_token)
            sentence_map.append(sentence_idx)
            subtoken_map.append(token_idx)
        if self.add_speaker_token:
            subtokens.append(self.get_speaker_token(speaker_id))
            sentence_map.append(sentence_idx)
            subtoken_map.append(token_idx)
        for sent in doc_or_uttr:
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

        if context is None:
            inst = create_inst(
                input_ids=tokenizer.convert_tokens_to_ids(subtokens),
                sentence_map=sentence_map, subtoken_map=subtoken_map,
                speaker_ids=[speaker_id] * len(subtokens) if self.use_speaker_indicator else [],
                uttr_start_idx=[1 if self.add_sep_token else 0],
                genre=genre, mentions=[]
            )
            return inst

        # Process context
        prev_sep_idx = len(context) - 1  # The middle SEP
        if self.add_sep_token:
            prev_sep_idx = context.uttr_start_idx[-1] - 1
        context_start_idx = len(context) - 1
        for idx in context.uttr_start_idx:
            if len(context) - idx + len(subtokens) + (0 if self.add_sep_token else 1) <= self.max_segment_len:
                context_start_idx = idx
                break

        if not self.add_sep_token or context_start_idx >= prev_sep_idx:
            # Cases: no SEP with any context utterances; or use SEP with at most one context utterance
            # No need to adjust context token offset
            uttr_start_idx = [idx - context_start_idx for idx in context.uttr_start_idx if idx - context_start_idx >= 0]
            if self.add_sep_token:
                assert uttr_start_idx == [] or uttr_start_idx == [0]
            uttr_start_idx.append(len(context) - context_start_idx - (0 if self.add_sep_token else 1))

            mentions = []
            for m in context.mentions:
                if m[0] >= context_start_idx:
                    mentions.append((m[0] - context_start_idx, m[1] - context_start_idx))

            inst = create_inst(
                input_ids=context.input_ids[context_start_idx:-1] + tokenizer.convert_tokens_to_ids(subtokens),
                sentence_map=context.sentence_map[context_start_idx:-1] + sentence_map,
                subtoken_map=context.subtoken_map[context_start_idx:-1] + subtoken_map,
                speaker_ids=context.speaker_ids[context_start_idx:-1] + [speaker_id] * len(subtokens)
                if self.use_speaker_indicator else [],
                uttr_start_idx=uttr_start_idx, genre=genre, mentions=mentions
            )
        else:
            # Case: use SEP and have at least two context utterances
            # Adjust context token offset of the last context utterance because of SEP
            uttr_start_idx = [idx - context_start_idx for idx in context.uttr_start_idx if idx - context_start_idx >= 0]
            assert len(uttr_start_idx) > 1
            uttr_start_idx[-1] -= 1  # Adjust for SEP
            uttr_start_idx.append(len(context) - context_start_idx - 1)

            mentions = []
            for m in context.mentions:
                if context_start_idx <= m[0] < prev_sep_idx:
                    mentions.append((m[0] - context_start_idx, m[1] - context_start_idx))
                elif m[0] >= context_start_idx and m[0] > prev_sep_idx:
                    mentions.append((m[0] - context_start_idx - 1, m[1] - context_start_idx - 1))  # Adjust for SEP

            inst = create_inst(
                input_ids=context.input_ids[context_start_idx:prev_sep_idx] +
                          context.input_ids[prev_sep_idx + 1:-1] + tokenizer.convert_tokens_to_ids(subtokens),
                sentence_map=context.sentence_map[context_start_idx:prev_sep_idx] +
                             context.sentence_map[prev_sep_idx + 1:-1] + sentence_map,
                subtoken_map=context.subtoken_map[context_start_idx:prev_sep_idx] +
                             context.subtoken_map[prev_sep_idx + 1:-1] + subtoken_map,
                speaker_ids=context.speaker_ids[context_start_idx:prev_sep_idx] +
                            context.speaker_ids[prev_sep_idx + 1:-1] + [speaker_id] * len(subtokens)
                if self.use_speaker_indicator else [],
                uttr_start_idx=uttr_start_idx, genre=genre, mentions=mentions
            )
        return inst

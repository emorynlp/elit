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
import torch
import torch.nn as nn
from transformers import BertModel
import logging
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
import torch.nn.init as init

import elit.components.coref.higher_order as ho
import elit.components.coref.util as util

logger = logging.getLogger(__name__)


class MlCorefModel(nn.Module):
    """ A simplified mention-linking model omitting certain components and loss. """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = torch.device('cpu') if device is None else device

        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']

        self.num_genres = len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        config['top_span_ratio'] = 0.3

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = BertModel.from_pretrained('bert-base-cased' if self.max_seg_len == 384 else 'bert-large-cased')
        if config['add_speaker_token']:
            self.bert.resize_token_embeddings(util.get_vocab_size(config))

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_span_width']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_genre']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_speaker_indicator']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_antecedent_distance']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_span_width'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10) if config['use_antecedent_distance'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_antecedent_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres) if config['use_genre'] else None
        self.emb_same_speaker = self.make_embedding(2) if config['use_speaker_indicator'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config['use_segment_distance'] else None

        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size,
                                                  [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'],
                                                    [config['ffnn_size']] * config['ffnn_depth'], output_size=1) \
            if config['use_width_prior'] else None
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) \
            if config['use_antecedent_distance_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                               output_size=1) if config['fine_grained'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) \
            if config['coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) \
            if config['higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'],
                                                 [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) \
            if config['higher_order'] == 'cluster_merging' else None

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, **inputs):
        return self.get_predictions_and_loss(**inputs)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, genre, sentence_map,
                                 uttr_start_idx=None, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                 context_mention_starts=None, context_mention_ends=None):
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True
        if do_loss:
            logger.error('Current model is not supported for training')
            return None

        if context_mention_starts is None:
            context_mention_starts = torch.tensor([], dtype=torch.long, device=device)
            context_mention_ends = torch.tensor([], dtype=torch.long, device=device)

        # Get token emb
        mention_doc, _ = self.bert(input_ids, attention_mask=input_mask, return_dict=False)
        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        if speaker_ids.shape[0] > 0:
            speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]
        if uttr_start_idx is None:
            uttr_start_idx = 1
        elif isinstance(uttr_start_idx, torch.Tensor):
            uttr_start_idx = uttr_start_idx.item()
        num_uttr_words = num_words - uttr_start_idx
        if num_uttr_words < 1:
            logger.error(f'No current utterance in the input')

        # Get candidate span
        sentence_indices = sentence_map
        candidate_starts = torch.unsqueeze(torch.arange(uttr_start_idx, num_words, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask]
        candidate_starts = torch.cat([context_mention_starts, candidate_starts])
        candidate_ends = torch.cat([context_mention_ends, candidate_ends])
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long), 0)

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if conf['use_span_width']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
                    candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score
        num_context_spans = torch.sum(candidate_ends < uttr_start_idx).item()  # Can be 0
        uttr_candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores[num_context_spans:],descending=True)
        candidate_idx_sorted_by_score = list(range(num_context_spans)) + \
                                        (num_context_spans + uttr_candidate_idx_sorted_by_score).tolist()

        # Extract top spans
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'],
                                conf['top_span_ratio'] * num_uttr_words + 1 + num_context_spans))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                   candidate_ends_cpu, num_top_spans,
                                                   allow_nested=conf['allow_nested_mentions'])
        assert len(selected_idx_cpu) == num_top_spans
        assert candidate_starts[num_context_spans] >= uttr_start_idx
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + \
                                     torch.unsqueeze(top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_antecedent_distance_prior']:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx, device)
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if not conf['fine_grained']:
            top_pairwise_scores = top_pairwise_fast_scores
        else:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_genre']:
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents, 1)
            if conf['use_speaker_indicator']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[
                    top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_antecedent_distance']:
                top_antecedent_distance = util.bucket_distance(
                    top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(
                    top_antecedent_distance)

            # Get final pairwise scores and related info
            top_pairwise_scores, cluster_merging_scores, top_span_emb, (
                target_emb, top_antecedent_emb, similarity_emb) = \
                self.get_final_pairwise_scores(top_span_emb, top_antecedent_idx, genre_emb, same_speaker_emb,
                                               seg_distance_emb, top_antecedent_distance_emb,
                                               top_pairwise_fast_scores)

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        return top_span_starts, top_span_ends, top_span_mention_scores, top_antecedent_idx, top_antecedent_scores

    def get_final_pairwise_scores(self, span_emb, antecedent_idx, genre_emb=None, same_speaker_emb=None,
                                  seg_distance_emb=None, antecedent_distance_emb=None, pairwise_fast_scores=None):
        conf = self.config
        pairwise_scores, cluster_merging_scores = None, None
        for depth in range(conf['coref_depth']):
            antecedent_emb = span_emb[antecedent_idx]
            feature_list = []
            if genre_emb is not None:
                feature_list.append(genre_emb)
            if same_speaker_emb is not None:
                feature_list.append(same_speaker_emb)
            if seg_distance_emb is not None:
                feature_list.append(seg_distance_emb)
            if antecedent_distance_emb is not None:
                feature_list.append(antecedent_distance_emb)
            feature_emb = torch.cat(feature_list, dim=-1)
            feature_emb = self.dropout(feature_emb)
            target_emb = torch.unsqueeze(span_emb, 1).repeat(1, antecedent_idx.shape[-1], 1)
            similarity_emb = target_emb * antecedent_emb
            pair_emb = torch.cat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2)
            pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
            pairwise_scores = pairwise_slow_scores
            if pairwise_fast_scores is not None:
                pairwise_scores += pairwise_fast_scores
            if conf['higher_order'] == 'cluster_merging':
                cluster_merging_scores = ho.cluster_merging(span_emb, antecedent_idx, pairwise_scores,
                                                            self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                            self.dropout,
                                                            device=self.device, reduce=conf['cluster_reduce'],
                                                            easy_cluster_first=conf['easy_cluster_first'])
                break
            elif depth != conf['coref_depth'] - 1:
                if conf['higher_order'] == 'attended_antecedent':
                    refined_span_emb = ho.attended_antecedent(span_emb, antecedent_emb, pairwise_scores, self.device)
                elif conf['higher_order'] == 'max_antecedent':
                    refined_span_emb = ho.max_antecedent(span_emb, antecedent_emb, pairwise_scores, self.device)
                elif conf['higher_order'] == 'entity_equalization':
                    refined_span_emb = ho.entity_equalization(span_emb, antecedent_emb, antecedent_idx, pairwise_scores,
                                                              self.device)
                elif conf['higher_order'] == 'span_clustering':
                    refined_span_emb = ho.span_clustering(span_emb, antecedent_idx, pairwise_scores,
                                                          self.span_attn_ffnn, self.device)

                gate = self.gate_ffnn(torch.cat([span_emb, refined_span_emb], dim=1))
                gate = torch.sigmoid(gate)
                span_emb = gate * refined_span_emb + (1 - gate) * span_emb
            return pairwise_scores, cluster_merging_scores, span_emb, (target_emb, antecedent_emb, similarity_emb)

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans,
                           allow_nested=True):
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}  # For nested
        selected_token_idx = set()  # For non-nested
        for i, candidate_idx in enumerate(candidate_idx_sorted):
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            if allow_nested:
                cross_overlap = False
                for token_idx in range(span_start_idx, span_end_idx + 1):
                    max_end = start_to_max_end.get(token_idx, -1)
                    if token_idx > span_start_idx and max_end > span_end_idx:
                        cross_overlap = True
                        break
                    min_start = end_to_min_start.get(token_idx, -1)
                    if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                        cross_overlap = True
                        break
                if not cross_overlap:
                    # Pass check; select idx and update dict stats
                    selected_candidate_idx.append(candidate_idx)
                    max_end = start_to_max_end.get(span_start_idx, -1)
                    if span_end_idx > max_end:
                        start_to_max_end[span_start_idx] = span_end_idx
                    min_start = end_to_min_start.get(span_end_idx, -1)
                    if min_start == -1 or span_start_idx < min_start:
                        end_to_min_start[span_end_idx] = span_start_idx
            else:
                span_start_idx = candidate_starts[candidate_idx]
                span_end_idx = candidate_ends[candidate_idx]
                overlap = False
                for token_idx in range(span_start_idx, span_end_idx + 1):
                    if token_idx in selected_token_idx:
                        overlap = True
                        break
                if not overlap:
                    # Pass check; select idx and update dict stats
                    selected_candidate_idx.append(candidate_idx)
                    for token_idx in range(span_start_idx, span_end_idx + 1):
                        selected_token_idx.add(token_idx)
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

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
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from elit.common.transform import VocabDict
from elit.components.amr.amr_parser.data import list_to_tensor, lists_of_string_to_tensor, DUM, NIL
from elit.components.amr.amr_parser.decoder import DecodeLayer
from elit.components.amr.amr_parser.encoder import ConceptEncoder
from elit.components.amr.amr_parser.search import Hypothesis, Beam, search_by_batch
from elit.components.amr.amr_parser.transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from elit.components.amr.amr_parser.utils import move_to_device


class GraphAbstractMeaningRepresentationDecoder(nn.Module):
    def __init__(self,
                 vocabs: VocabDict,
                 concept_char_dim,
                 concept_dim,
                 cnn_filters,
                 char2concept_dim,
                 embed_dim,
                 ff_embed_dim,
                 num_heads,
                 dropout,
                 graph_layers,
                 inference_layers,
                 rel_dim,
                 encoder_size,
                 **kwargs):
        super(GraphAbstractMeaningRepresentationDecoder, self).__init__()
        self.vocabs = vocabs
        self.rel_pad_idx = self.vocabs['rel'].pad_idx
        self.rel_nil_idx = self.vocabs['rel'].get_idx(NIL)
        self.concept_pad_idx = self.vocabs['concept'].pad_idx
        self.lem_pad_idx = self.vocabs.get('lemma', self.vocabs.get('lem', None)).pad_idx
        self.concept_encoder = ConceptEncoder(vocabs, concept_char_dim, concept_dim, embed_dim, cnn_filters,
                                              char2concept_dim, dropout)
        self.graph_encoder = Transformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True,
                                         weights_dropout=False)
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.word_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask()
        self.decoder = DecodeLayer(vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, concept_dim,
                                   rel_dim, dropout, joint_arc_concept=True, joint_rel=False, external_biaffine=False,
                                   optimize_every_layer=False, levi_graph=False)
        self.probe_generator = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.encoder_adaptor = nn.Linear(encoder_size, embed_dim)
        self.squeeze = False
        self.bart = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'probe_generator'):
            nn.init.normal_(self.probe_generator.weight, std=0.02)
            nn.init.constant_(self.probe_generator.bias, 0.)

    def encode_step_with_transformer(self, plm_embed, lem):
        # Not sure why the original paper has to set [CLS] to 0
        plm_embed = plm_embed.clone()
        plm_embed[:, 0, :] = 0
        plm_embed = plm_embed.transpose(0, 1)
        word_repr = self.encoder_adaptor(plm_embed)
        word_repr = self.embed_scale * word_repr + self.embed_positions(lem)
        word_repr = self.word_embed_layer_norm(word_repr)
        probe = torch.tanh(self.probe_generator(word_repr[:1]))
        word_repr = word_repr[1:]
        word_mask = torch.eq(lem, self.lem_pad_idx)
        word_mask = word_mask[1:]
        return word_repr, word_mask, probe

    def predict(self, batch, beam_size, max_time_step, min_time_step=1, h=None):
        with torch.no_grad():
            if callable(h):
                h = h(batch)
            word_repr, word_mask, probe = self.encode_step_with_transformer(h, batch['lem'])
            mem_dict = {'state': word_repr,
                        'mask': word_mask,
                        'probe': probe,
                        'local_idx2token': batch['local_idx2token'],
                        'copy_seq': batch['copy_seq']}
            init_state_dict = {}
            init_hyp = Hypothesis(init_state_dict, [DUM], 0.)
            batch_size = batch['copy_seq'].size(1)
            beams = [Beam(beam_size, min_time_step, max_time_step, [init_hyp], self.device) for i in
                     range(batch_size)]
            search_by_batch(self, beams, mem_dict)

        return beams

    def prepare_incremental_input(self, step_seq):
        conc = list_to_tensor(step_seq, self.vocabs['concept'])
        conc_char = lists_of_string_to_tensor(step_seq, self.vocabs['concept_char'])
        conc, conc_char = move_to_device(conc, self.device), move_to_device(conc_char, self.device)
        return conc, conc_char

    def decode_step(self, inp, state_dict, mem_dict, offset, topk):
        new_state_dict = {}
        local_vocabs = mem_dict['local_idx2token']
        levi_graph = self.decoder.levi_graph if hasattr(self, 'decoder') else False
        step_concept, step_concept_char = inp
        word_repr = snt_state = mem_dict['state']
        word_mask = snt_padding_mask = mem_dict['mask']
        probe = mem_dict['probe']
        copy_seq = mem_dict['copy_seq']
        _, bsz, _ = word_repr.size()

        concept_repr = self.embed_scale * self.concept_encoder(step_concept_char,
                                                               step_concept) + self.embed_positions(
            step_concept, offset)
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        for idx, layer in enumerate(self.graph_encoder.layers):
            name_i = 'concept_repr_%d' % idx
            if name_i in state_dict:
                prev_concept_repr = state_dict[name_i]
                new_concept_repr = torch.cat([prev_concept_repr, concept_repr], 0)
            else:
                new_concept_repr = concept_repr

            new_state_dict[name_i] = new_concept_repr
            concept_repr, _, _ = layer(concept_repr, kv=new_concept_repr, external_memories=word_repr,
                                       external_padding_mask=word_mask)
        name = 'graph_state'
        if name in state_dict:
            prev_graph_state = state_dict[name]
            new_graph_state = torch.cat([prev_graph_state, concept_repr], 0)
        else:
            new_graph_state = concept_repr
        new_state_dict[name] = new_graph_state
        conc_ll, arc_ll, rel_ll = self.decoder(probe, snt_state, new_graph_state, snt_padding_mask, None, None,
                                               copy_seq, work=True)
        for i in range(offset):
            name = 'arc_ll%d' % i
            new_state_dict[name] = state_dict[name]
            if not levi_graph:
                name = 'rel_ll%d' % i
                new_state_dict[name] = state_dict[name]
        name = 'arc_ll%d' % offset
        new_state_dict[name] = arc_ll  # tgt_len x bsz x tgt_len
        if not levi_graph:
            name = 'rel_ll%d' % offset
            new_state_dict[name] = rel_ll  # dep_num x bsz x head_num x vocab_size

        # pred_arc = torch.lt(pred_arc_prob, 0.5)
        # pred_arc[:,:,0] = 1
        # rel_confidence = rel_ll.masked_fill(pred_arc, 0.).sum(-1, keepdim=True)
        pred_arc_prob = torch.exp(arc_ll)
        arc_confidence = torch.log(torch.max(pred_arc_prob, 1 - pred_arc_prob))
        arc_confidence[:, :, 0] = 0.
        LL = conc_ll + arc_confidence.sum(-1, keepdim=True)  # + rel_confidence

        def idx2token(idx, local_vocab):
            if idx in local_vocab:
                return local_vocab[idx]
            return self.vocabs['predictable_concept'].idx_to_token[idx]

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1)  # bsz x k

        results = []
        for s, t, local_vocab in zip(topk_scores.tolist(), topk_token.tolist(), local_vocabs):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, local_vocab), score))
            results.append(res)

        return new_state_dict, results

    def forward(self, plm_embed, mask=None, batch=None, **kwargs: Any):
        word_repr, word_mask, probe = self.encode_step_with_transformer(plm_embed, batch['lem'])
        concept_repr = self.embed_scale * self.concept_encoder(batch['concept_char_in'],
                                                               batch['concept_in']) + self.embed_positions(
            batch['concept_in'])
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_repr = F.dropout(concept_repr, p=self.dropout, training=self.training)
        concept_mask = torch.eq(batch['concept_in'], self.concept_pad_idx)
        attn_mask = self.self_attn_mask(batch['concept_in'].size(0))
        for idx, layer in enumerate(self.graph_encoder.layers):
            concept_repr, arc_weight, _ = layer(concept_repr,
                                                self_padding_mask=concept_mask, self_attn_mask=attn_mask,
                                                external_memories=word_repr, external_padding_mask=word_mask,
                                                need_weights='max')

        graph_target_rel = batch['rel'][:-1]
        graph_target_arc = torch.ne(graph_target_rel, self.rel_nil_idx)  # 0 or 1
        graph_arc_mask = torch.eq(graph_target_rel, self.rel_pad_idx)
        graph_arc_loss = F.binary_cross_entropy(arc_weight, graph_target_arc.float(), reduction='none')
        graph_arc_loss = graph_arc_loss.masked_fill_(graph_arc_mask, 0.).sum((0, 2))

        if self.decoder.joint_arc_concept:
            probe: torch.Tensor = probe.expand(word_repr.size(0) + concept_repr.size(0), -1, -1)
        else:
            probe = probe.expand_as(concept_repr)  # tgt_len x bsz x embed_dim
        concept_loss, arc_loss, rel_loss = self.decoder(probe, word_repr, concept_repr, word_mask, concept_mask,
                                                        attn_mask, batch['copy_seq'], target=batch['concept_out'],
                                                        target_rel=batch['rel'][1:])

        concept_tot = concept_mask.size(0) - concept_mask.float().sum(0)
        if self.decoder.joint_arc_concept:
            concept_loss, concept_correct, concept_total = concept_loss
        if rel_loss is not None:
            rel_loss, rel_correct, rel_total = rel_loss
            rel_loss = rel_loss / concept_tot
        concept_loss = concept_loss / concept_tot
        arc_loss = arc_loss / concept_tot
        graph_arc_loss = graph_arc_loss / concept_tot
        if self.decoder.joint_arc_concept:
            # noinspection PyUnboundLocalVariable
            if rel_loss is not None:
                rel_out = (rel_loss.mean(), rel_correct, rel_total)
            else:
                rel_out = None
            return (concept_loss.mean(), concept_correct, concept_total), \
                   arc_loss.mean(), rel_out, graph_arc_loss.mean()
        return concept_loss.mean(), arc_loss.mean(), rel_loss.mean(), graph_arc_loss.mean()

    @property
    def device(self):
        return next(self.parameters()).device

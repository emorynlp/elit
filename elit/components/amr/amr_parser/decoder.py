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
from typing import Any, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from elit.common.transform import VocabDict
from elit.components.amr.amr_parser.data import NIL
from elit.components.amr.amr_parser.utils import compute_f_by_tensor
from elit.components.amr.amr_parser.transformer import MultiheadAttention
from elit.components.amr.amr_parser.utils import label_smoothed_nll_loss
from elit.components.parsers.biaffine.biaffine_model import BiaffineDecoder


class ArcGenerator(nn.Module):
    def __init__(self, vocabs: VocabDict, embed_dim, ff_embed_dim, num_heads, dropout):
        super(ArcGenerator, self).__init__()
        self.vocabs = vocabs
        self.rel_pad_idx = self.vocabs['rel'].pad_idx
        self.rel_nil_idx = self.vocabs['rel'].get_idx(NIL)
        self.arc_layer = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout=False)
        self.arc_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, outs, graph_state, graph_padding_mask, attn_mask, target_rel=None, work=False):
        """

        Args:
            outs: [CLS] representation from sequence encoder or the attended value from concept generator
            graph_state: concept representation from concept Transformer
            graph_padding_mask:
            attn_mask:
            target_rel:
            work:

        Returns:
            (tuple): tuple containing:

                - (torch.Tensor): arc_loss
                - (torch.Tensor): Attended value

        """
        x, arc_weight = self.arc_layer(outs, graph_state, graph_state,
                                       key_padding_mask=graph_padding_mask,
                                       attn_mask=attn_mask,
                                       need_weights='max')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.arc_layer_norm(outs + x)
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.ff_layer_norm(residual + x)

        if work:
            arc_ll = torch.log(arc_weight + 1e-12)
            return arc_ll, outs
        target_arc = torch.ne(target_rel, self.rel_nil_idx)  # 0 or 1
        arc_mask = torch.eq(target_rel, self.rel_pad_idx)
        if not self.training:
            pred = torch.ge(arc_weight, 0.5)
            print('arc p %.3f r %.3f f %.3f' % compute_f_by_tensor(pred, target_arc, arc_mask))
        arc_loss = F.binary_cross_entropy(arc_weight, target_arc.float(), reduction='none')
        arc_loss = arc_loss.masked_fill_(arc_mask, 0.).sum((0, 2))
        return arc_loss, outs


class ConceptGenerator(nn.Module):
    def __init__(self, vocabs, embed_dim, ff_embed_dim, conc_size, dropout):
        super(ConceptGenerator, self).__init__()
        self.concept_padding_idx = vocabs['predictable_concept'].pad_idx
        self.alignment_layer = MultiheadAttention(embed_dim, 1, dropout, weights_dropout=False)
        self.alignment_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.transfer = nn.Linear(embed_dim, conc_size)
        self.generator = nn.Linear(conc_size, len(vocabs['predictable_concept']))
        self.diverter = nn.Linear(conc_size, 3)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer.weight, std=0.02)
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.normal_(self.generator.weight, std=0.02)
        nn.init.constant_(self.diverter.bias, 0.)
        nn.init.constant_(self.transfer.bias, 0.)
        nn.init.constant_(self.generator.bias, 0.)

    def forward(self, outs, snt_state, snt_padding_mask, copy_seq,
                target=None, work=False):
        """

        Args:
          outs: Attended value from arc_generator
          snt_state: The Transformer representations for each word.
          snt_padding_mask: 
          copy_seq: Stack of lemma and concept (which is simply lemma + _)
          target:  (Default value = None)
          work:  (Default value = False)

        Returns:

        
        """
        # Q: last concept embedding, K and V: token embeddings
        x, alignment_weight = self.alignment_layer(outs, snt_state, snt_state,
                                                   key_padding_mask=snt_padding_mask,
                                                   need_weights='one')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.alignment_layer_norm(outs + x)
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.ff_layer_norm(residual + x)

        seq_len, bsz, _ = outs.size()
        outs_concept = torch.tanh(self.transfer(outs))
        outs_concept = F.dropout(outs_concept, p=self.dropout, training=self.training)

        gen_gate, map_gate, copy_gate = F.softmax(self.diverter(outs_concept), -1).chunk(3, dim=-1)
        copy_gate = torch.cat([copy_gate, map_gate], -1)

        probs = gen_gate * F.softmax(self.generator(outs_concept), -1)

        tot_ext = 1 + copy_seq.max().item()
        vocab_size = probs.size(-1)

        if tot_ext - vocab_size > 0:
            ext_probs = probs.new_zeros((1, 1, tot_ext - vocab_size)).expand(seq_len, bsz, -1)
            probs = torch.cat([probs, ext_probs], -1)
        # copy_seq: src_len x bsz x 2
        # copy_gate: tgt_len x bsz x 2
        # alignment_weight: tgt_len x bsz x src_len
        # index: tgt_len x bsz x (src_len x 2)
        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(seq_len, -1, -1)
        copy_probs = (copy_gate.unsqueeze(2) * alignment_weight.unsqueeze(-1)).view(seq_len, bsz, -1)
        probs = probs.scatter_add_(-1, index, copy_probs)
        ll = torch.log(probs + 1e-12)

        if work:
            return ll, outs

        if not self.training:
            _, pred = torch.max(ll, -1)
            total_concepts = torch.ne(target, self.concept_padding_idx)
            acc = torch.eq(pred, target).masked_select(total_concepts).float().sum().item()
            tot = total_concepts.sum().item()
            print('conc acc', acc / tot)

        concept_loss = -ll.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        concept_mask = torch.eq(target, self.concept_padding_idx)
        concept_loss = concept_loss.masked_fill_(concept_mask, 0.).sum(0)
        return concept_loss, outs


class ArcConceptDecoder(nn.Module):
    def __init__(self, vocabs, embed_dim, conc_size, dropout, joint_rel=False):
        super(ArcConceptDecoder, self).__init__()
        self.joint_rel = joint_rel
        self.concept_padding_idx = vocabs['predictable_concept'].pad_idx
        self.transfer = nn.Linear(embed_dim, conc_size)
        self.generator = nn.Linear(conc_size, len(vocabs['predictable_concept']))
        self.separate_rel = 'concept_and_rel' in vocabs
        if self.separate_rel:
            self.concept_or_rel = nn.Linear(conc_size, 2)
            self.rel_generator = nn.Linear(conc_size, len(vocabs['rel']))
        self.diverter = nn.Linear(conc_size, 3)
        self.dropout = dropout
        self.rel_nil_idx = vocabs['rel'].get_idx(NIL)
        self.rel_pad_idx = vocabs['rel'].pad_idx
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer.weight, std=0.02)
        nn.init.normal_(self.diverter.weight, std=0.02)
        nn.init.normal_(self.generator.weight, std=0.02)
        nn.init.constant_(self.diverter.bias, 0.)
        nn.init.constant_(self.transfer.bias, 0.)
        nn.init.constant_(self.generator.bias, 0.)
        if self.separate_rel:
            nn.init.normal_(self.rel_generator.weight, std=0.02)
            nn.init.constant_(self.rel_generator.bias, 0.)

    def forward(self, alignment_weight, arc_weight, rel_weight, concept_outs, copy_seq, target, target_rel, bsz,
                src_len, outs=None, work=False):
        snt_len = alignment_weight.size(-1)
        if snt_len < copy_seq.size(0):
            copy_seq = copy_seq[:snt_len, :, :]
        outs_concept = torch.tanh(self.transfer(concept_outs))
        outs_concept = F.dropout(outs_concept, p=self.dropout, training=self.training)
        gen_gate, map_gate, copy_gate = F.softmax(self.diverter(outs_concept), -1).chunk(3, dim=-1)
        copy_gate = torch.cat([copy_gate, map_gate], -1)
        probs = gen_gate * F.softmax(self.generator(outs_concept), -1)
        tot_ext = 1 + copy_seq.max().item()
        vocab_size = probs.size(-1)
        if tot_ext - vocab_size > 0:
            ext_probs = probs.new_zeros((1, 1, tot_ext - vocab_size)).expand(src_len, bsz, -1)
            probs = torch.cat([probs, ext_probs], -1)
        # copy_seq: src_len x bsz x 2
        # copy_gate: tgt_len x bsz x 2
        # alignment_weight: tgt_len x bsz x src_len
        # index: tgt_len x bsz x (src_len x 2)
        index = copy_seq.transpose(0, 1).contiguous().view(1, bsz, -1).expand(src_len, -1, -1)
        # attn: alignment x 1, arc, rel
        if arc_weight.ndim == 4:
            arc_weight, _ = torch.max(arc_weight, dim=0)
        # rel attention
        if self.joint_rel:
            # noinspection PyCallByClass
            rel_rets = RelationGenerator.forward_scores(self, rel_weight, outs, target_rel, work)
        copy_probs = (copy_gate.unsqueeze(2) * alignment_weight.unsqueeze(-1)).reshape(src_len, bsz, -1)
        probs = probs.scatter_add_(-1, index, copy_probs)
        # Switch: concept or rel?
        if self.separate_rel:
            concept_gate, rel_gate = F.softmax(self.concept_or_rel(outs_concept), -1).chunk(2, dim=-1)
            probs *= concept_gate
            rel_probs = rel_gate * F.softmax(self.rel_generator(outs_concept), -1)
            probs[:, :, vocab_size:vocab_size + rel_probs.size(-1)] = rel_probs

        ll = torch.log(probs + 1e-12)
        if work:
            arc_ll = torch.log(arc_weight + 1e-12)
            if self.joint_rel:
                return ll, arc_ll, rel_rets, outs
            return ll, arc_ll, outs
        concept_loss = -ll.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        concept_mask = torch.eq(target, self.concept_padding_idx)
        concept_loss = concept_loss.masked_fill_(concept_mask, 0.).sum(0)
        assert self.training
        _, pred = torch.max(ll, -1)
        total_concepts = torch.ne(target, self.concept_padding_idx)
        acc = torch.eq(pred, target).masked_select(total_concepts).float().sum().item()
        tot = total_concepts.sum().item()
        concept_loss = (concept_loss, acc, tot)
        target_arc = torch.ne(target_rel, self.rel_nil_idx)  # 0 or 1
        arc_mask = torch.eq(target_rel, self.rel_pad_idx)
        arc_loss = F.binary_cross_entropy(arc_weight, target_arc.float(), reduction='none')
        arc_loss = arc_loss.masked_fill_(arc_mask, 0.).sum((0, 2))
        if self.joint_rel:
            return concept_loss, arc_loss, rel_rets, outs
        return concept_loss, arc_loss, outs


class JointArcConceptGenerator(ArcConceptDecoder):
    def __init__(self, vocabs, embed_dim, ff_embed_dim, conc_size, dropout, num_heads, joint_rel=False):
        super(JointArcConceptGenerator, self).__init__(vocabs, embed_dim, conc_size, dropout, joint_rel)
        if joint_rel:
            # Make num_heads divider of embed_dim
            self.rel_vocab_size = len(vocabs['rel'])
            num_heads = self.rel_vocab_size + 8  # We need at least this amount of heads
            # Find the divisor >= num_heads
            possible_heads = [x for x in range(num_heads, embed_dim) if embed_dim % x == 0]
            assert possible_heads, 'Cannot find a proper num_heads, try to increase embed_dim'
            num_heads = possible_heads[0]

        self.transformer_layer = MultiheadAttention(embed_dim, num_heads, dropout, weights_dropout=False)
        self.transformer_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)

    # noinspection PyMethodOverriding
    def forward(self, outs, state, mask, attn_mask, copy_seq, target=None, target_rel=None,
                work=False):
        """

        Args:
          outs: Attended value from arc_generator
          state: The Transformer representations for each word.
          mask:
          attn_mask:
          copy_seq: Stack of lemma and concept (which is simply lemma + _)
          target:  Target concept
          work:  (Default value = False)

        Returns:


        """
        x, attention = self.transformer_layer(outs, state, state,
                                              key_padding_mask=mask,
                                              attn_mask=attn_mask,
                                              need_weights='all')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.transformer_layer_norm(outs + x)
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.ff_layer_norm(residual + x)

        src_len, bsz, _ = outs.size()
        snt_len = copy_seq.size(0)
        if src_len > snt_len:
            src_len -= snt_len
            tgt_len = src_len
            snt_outs, concept_outs = outs[:-src_len, :, :], outs[-src_len:, :, :]
        else:
            tgt_len = state.size(0) - snt_len
            concept_outs = outs
        rel_vocab_size = self.rel_vocab_size if self.joint_rel else None
        arc_weight = attention[1:-rel_vocab_size if rel_vocab_size else None, -src_len:, :, -tgt_len:]
        alignment_weight = attention[0, -src_len:, :, :snt_len]

        if self.joint_rel:
            rel_weight = attention[-rel_vocab_size:, -src_len:, :, -tgt_len:]
            rel_weight = rel_weight.permute([1, 2, 3, 0])
        else:
            rel_weight = None

        return super(JointArcConceptGenerator, self).forward(alignment_weight, arc_weight, rel_weight, concept_outs,
                                                             copy_seq, target, target_rel, bsz, src_len, outs, work)


class RelationGenerator(nn.Module):

    def __init__(self, vocabs, embed_dim, rel_size, dropout):
        super(RelationGenerator, self).__init__()
        self.rel_nil_idx = vocabs['rel'].get_idx(NIL)
        self.rel_pad_idx = vocabs['rel'].pad_idx
        self.transfer_head = nn.Linear(embed_dim, rel_size)
        self.transfer_dep = nn.Linear(embed_dim, rel_size)

        self.rel_vocab_size = len(vocabs['rel'])
        self.proj = nn.Linear(rel_size + 1, self.rel_vocab_size * (rel_size + 1))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.transfer_head.weight, std=0.02)
        nn.init.normal_(self.transfer_dep.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

        nn.init.constant_(self.proj.bias, 0.)
        nn.init.constant_(self.transfer_head.bias, 0.)
        nn.init.constant_(self.transfer_dep.bias, 0.)

    def forward(self, outs, graph_state, target_rel=None, work=False):
        def get_scores(dep, head):
            head = torch.tanh(self.transfer_head(head))
            dep = torch.tanh(self.transfer_dep(dep))

            head = F.dropout(head, p=self.dropout, training=self.training)
            dep = F.dropout(dep, p=self.dropout, training=self.training)

            dep_num, bsz, _ = dep.size()
            head_num = head.size(0)

            bias_dep = dep.new_ones((dep_num, bsz, 1))
            bias_head = head.new_ones((head_num, bsz, 1))

            # seq_len x bsz x dim
            dep = torch.cat([dep, bias_dep], 2)
            head = torch.cat([head, bias_head], 2)

            # bsz x dep_num x vocab_size x dim
            dep = self.proj(dep).view(dep_num, bsz, self.rel_vocab_size, -1).transpose(0, 1).contiguous()
            # bsz x dim x head_num
            head = head.permute(1, 2, 0)

            # bsz x dep_num x vocab_size x head_num
            scores = torch.bmm(dep.view(bsz, dep_num * self.rel_vocab_size, -1), head).view(bsz, dep_num,
                                                                                            self.rel_vocab_size,
                                                                                            head_num)
            return scores

        # # dep_num x bsz x head_num x vocab_size
        scores = get_scores(outs, graph_state).permute(1, 0, 3, 2).contiguous()

        return self.forward_scores(scores, outs, target_rel, work)

    def forward_scores(self, scores, outs, target_rel, work):
        dep_num, bsz, head_num, rel_vocab_size = scores.size()
        log_probs = F.log_softmax(scores, dim=-1)
        _, rel = torch.max(log_probs, -1)
        if work:
            # dep_num x bsz x head x vocab
            return log_probs
        pad_mask = torch.ne(target_rel, self.rel_pad_idx)
        nil_mask = torch.ne(target_rel, self.rel_nil_idx)
        rel_mask = pad_mask & nil_mask
        if torch.any(rel_mask):
            rel_acc = torch.eq(rel, target_rel)[rel_mask].sum().item()
        else:
            rel_acc = 0
        rel_tot = rel_mask.sum().item()
        # if not self.training:
        # print('rel acc %.3f' % (rel_acc / rel_tot))
        rel_loss = label_smoothed_nll_loss(log_probs.view(-1, self.rel_vocab_size), target_rel.view(-1), 0.).view(
            dep_num, bsz, -1)
        rel_loss = rel_loss.masked_fill_(~rel_mask, 0.).sum((0, 2))
        # rel_loss = label_smoothed_nll_loss(log_probs[rel_mask].view(-1, self.rel_vocab_size),
        #                                    target_rel[rel_mask].view(-1), 0.)
        # rel_loss = rel_loss.sum() / bsz
        return (rel_loss, rel_acc, rel_tot)


class BiaffineArcRelDecoder(BiaffineDecoder):

    def __init__(self, hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, vocabs) -> None:
        self.rel_vocab_size = len(vocabs['rel'])
        super().__init__(hidden_size, n_mlp_arc, n_mlp_rel, mlp_dropout, self.rel_vocab_size)
        self.rel_nil_idx = vocabs['rel'].get_idx(NIL)
        self.rel_pad_idx = vocabs['rel'].pad_idx

    # noinspection PyMethodOverriding
    def forward(self, outs, graph_state, target_rel=None, work=False, **kwargs: Any) -> Tuple[
        torch.Tensor, torch.Tensor]:
        arc_h = self.mlp_arc_h(graph_state).transpose(0, 1)
        arc_d = self.mlp_arc_d(outs).transpose(0, 1)
        rel_h = self.mlp_rel_h(graph_state).transpose(0, 1)
        rel_d = self.mlp_rel_d(outs).transpose(0, 1)
        s_arc, s_rel = self.decode(arc_d, arc_h, rel_d, rel_h, None, self.arc_attn, self.rel_attn)
        s_arc = s_arc.transpose(0, 1)
        s_rel = s_rel.transpose(0, 1)
        if work:
            arc_rets = torch.log(torch.sigmoid(s_arc))
        else:
            target_arc = torch.ne(target_rel, self.rel_nil_idx)  # 0 or 1
            arc_mask = torch.ne(target_rel, self.rel_pad_idx)
            arc_loss = F.binary_cross_entropy_with_logits(s_arc[arc_mask], target_arc[arc_mask].float(),
                                                          reduction='none')
            dep_num, bsz, _ = outs.size()
            arc_loss = arc_loss.sum() / bsz
            arc_rets = arc_loss
        rel_rets = RelationGenerator.forward_scores(self, s_rel, outs, target_rel, work)
        return arc_rets, rel_rets


class DecodeLayer(nn.Module):

    def __init__(self, vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, conc_size, rel_size, dropout,
                 joint_arc_concept=False, joint_rel=False, external_biaffine=False, n_mlp_arc=500,
                 optimize_every_layer=False, levi_graph=False):
        super(DecodeLayer, self).__init__()
        self.optimize_every_layer = optimize_every_layer
        self.inference_iterations = inference_layers
        self.joint_arc_concept = joint_arc_concept
        self.joint_rel = joint_rel
        if joint_arc_concept:
            self.arc_concept_generator = JointArcConceptGenerator(vocabs, embed_dim, ff_embed_dim, conc_size, dropout,
                                                                  num_heads, joint_rel)
        else:
            self.arc_generator = ArcGenerator(vocabs, embed_dim, ff_embed_dim, num_heads, dropout)
            self.concept_generator = ConceptGenerator(vocabs, embed_dim, ff_embed_dim, conc_size, dropout)
        self.levi_graph = levi_graph
        self.external_biaffine = external_biaffine
        if external_biaffine:
            self.biaffine = BiaffineArcRelDecoder(embed_dim, n_mlp_arc, rel_size, dropout, vocabs)
        else:
            if not levi_graph:
                self.relation_generator = RelationGenerator(vocabs, embed_dim, rel_size, dropout)
        self.dropout = dropout
        self.vocabs = vocabs

    def forward(self, probe, snt_state, graph_state,
                snt_padding_mask, graph_padding_mask, attn_mask,
                copy_seq, target=None, target_rel=None,
                work=None):
        """
        Run arc_generator and concept_generator iteratively.

        Args:
            probe: The [CLS] representation, tgt_len x bsz x embed_dim
            snt_state: The Transformer representations for each word.
            graph_state: concept representation from concept Transformer (won't be updated)
            snt_padding_mask:
            graph_padding_mask: [DUM] + x + [END] != concept_pad_idx, where x is the BFS order of AMR concepts
            attn_mask: Only allows each position in the node sequence to attend to all positions up to and including
                        that position
            copy_seq: Stack of lemma and concept (which is simply lemma + _)
            target: Target concept
            target_rel: Relation between concepts
            work: Decoding flag

        Returns:

        """
        # probe: tgt_len x bsz x embed_dim
        # state, graph_state: seq_len x bsz x embed_dim
        outs = F.dropout(probe, p=self.dropout, training=self.training)
        if work is None:
            work = not self.training

        # if work:
        #     for i in range(self.inference_iterations):
        #         arc_ll, outs = self.arc_generator(outs, graph_state, graph_padding_mask, attn_mask, work=True)
        #         concept_ll, outs = self.concept_generator(outs, snt_state, snt_padding_mask, copy_seq, work=True)
        #     rel_ll = self.relation_generator(outs, graph_state, work=True)
        #     return concept_ll, arc_ll, rel_ll

        arc_losses, concept_losses, rel_losses = [], [], []
        device = snt_state.device
        if self.joint_arc_concept:
            snt_len, concept_len, bsz = snt_state.size(0), graph_state.size(0), graph_state.size(1)
            state = torch.cat([snt_state, graph_state], dim=0)
            if graph_padding_mask is None:
                graph_padding_mask = torch.zeros((concept_len, bsz), device=device, dtype=torch.bool)
            mask = torch.cat([snt_padding_mask, graph_padding_mask], dim=0)
            if attn_mask is not None:
                left = torch.zeros((snt_len + concept_len, snt_len), dtype=torch.bool, device=device)
                upper_right = torch.ones((snt_len, concept_len), dtype=torch.bool, device=device)
                right = torch.cat([upper_right, attn_mask], dim=0)
                attn_mask = torch.cat([left, right], dim=1)
            for i in range(self.inference_iterations):
                rets = self.arc_concept_generator(outs, state, mask, attn_mask, copy_seq,
                                                  target=target, target_rel=target_rel,
                                                  work=work)
                if self.joint_rel:
                    concept_loss, arc_loss, rel_loss, outs = rets
                    rel_losses.append(rel_loss)
                else:
                    concept_loss, arc_loss, outs = rets
                concept_losses.append(concept_loss)
                arc_losses.append(arc_loss)
            outs = outs[-concept_len:, :, :]
        else:
            for i in range(self.inference_iterations):
                arc_loss, outs = self.arc_generator(outs, graph_state, graph_padding_mask, attn_mask,
                                                    target_rel=target_rel,
                                                    work=work)
                concept_loss, outs = self.concept_generator(outs, snt_state, snt_padding_mask, copy_seq, target=target,
                                                            work=work)
                arc_losses.append(arc_loss)
                concept_losses.append(concept_loss)
        alphas = None
        if self.optimize_every_layer:
            alphas = torch.logspace(start=1, end=self.inference_iterations, steps=self.inference_iterations,
                                    base=2, device=device).unsqueeze(1) / (2 ** (self.inference_iterations + 1))
        if self.external_biaffine:
            arc_loss, rel_loss = self.biaffine(outs, graph_state, target_rel=target_rel, work=work)
        else:
            if self.joint_rel:
                rel_loss = self.pool_losses(rel_losses, alphas, work)
            else:
                if self.levi_graph:
                    rel_loss = None
                else:
                    rel_loss = self.relation_generator(outs, graph_state, target_rel=target_rel, work=work)
            arc_loss = self.pool_losses(arc_losses, alphas, work)
        concept_loss = self.pool_losses(concept_losses, alphas, work)
        return concept_loss, arc_loss, rel_loss

    def pool_losses(self, losses, alphas, work):
        if self.optimize_every_layer and not work:
            loss_is_tuple = isinstance(losses[-1], tuple)
            loss = torch.stack([x[0] for x in losses] if loss_is_tuple else losses) * alphas
            loss = loss.sum(0)
            if loss_is_tuple:
                loss = (loss,) + losses[-1][1:]
        else:
            loss = losses[-1]
        return loss

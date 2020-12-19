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

import torch
import torch.nn.functional as F
from torch import nn
from elit.common.transform import VocabDict
from elit.components.amr.amr_parser.customized_bart import CustomizedDecoderLayer
from elit.components.amr.amr_parser.data import list_to_tensor, lists_of_string_to_tensor, DUM, NIL
from elit.components.amr.amr_parser.decoder import DecodeLayer, ArcConceptDecoder, RelationGenerator
from elit.components.amr.amr_parser.encoder import WordEncoder, ConceptEncoder
from elit.components.amr.amr_parser.search import Hypothesis, Beam, search_by_batch
from elit.components.amr.amr_parser.transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from elit.components.amr.amr_parser.utils import move_to_device
from elit.layers.embeddings.contextual_word_embedding import ContextualWordEmbeddingModule
from elit.layers.transformers.pt_imports import BertModel, BartModel
from elit.layers.transformers.utils import pick_tensor_for_each_token
from alnlp.modules.util import lengths_to_mask
from transformers.file_utils import ModelOutput


class GraphSequenceAbstractMeaningRepresentationModel(nn.Module):
    def __init__(self, vocabs: VocabDict, word_char_dim, word_dim, pos_dim, ner_dim,
                 concept_char_dim, concept_dim,
                 cnn_filters, char2word_dim, char2concept_dim,
                 embed_dim, ff_embed_dim, num_heads, dropout,
                 snt_layers, graph_layers, inference_layers, rel_dim,
                 pretrained_file=None, encoder=None, joint_arc_concept=False,
                 joint_rel=False,
                 external_biaffine=False,
                 optimize_every_layer=False,
                 squeeze=False,
                 levi_graph=False,
                 bart=False,
                 tokenizer=None,
                 **kwargs):
        """
        Implementation of Cai and Lam (2020) and my unpublished models.

        Args:
            vocabs:
            word_char_dim:
            word_dim:
            pos_dim:
            ner_dim:
            concept_char_dim:
            concept_dim:
            cnn_filters:
            char2word_dim:
            char2concept_dim:
            embed_dim:
            ff_embed_dim:
            num_heads:
            dropout:
            snt_layers:
            graph_layers:
            inference_layers:
            rel_dim:
            pretrained_file:
            encoder:
            joint_arc_concept:
            joint_rel:
            external_biaffine:
            optimize_every_layer:
            squeeze:
            levi_graph:
            bart:
            tokenizer:
            **kwargs:
        """
        super(GraphSequenceAbstractMeaningRepresentationModel, self).__init__()
        self.bart = bart
        self.squeeze = squeeze
        self.vocabs = vocabs
        self.rel_pad_idx = self.vocabs['rel'].pad_idx
        self.rel_nil_idx = self.vocabs['rel'].get_idx(NIL)
        self.concept_pad_idx = self.vocabs['concept'].pad_idx
        self.lem_pad_idx = self.vocabs.get('lemma', self.vocabs.get('lem', None)).pad_idx

        if squeeze or bart:
            bert_dim = encoder.get_output_dim() if hasattr(encoder, 'get_output_dim') else encoder.config.hidden_size
            self.arc_concept_decoder = ArcConceptDecoder(vocabs, bert_dim, concept_dim, dropout, joint_rel)
            self.relation_generator = RelationGenerator(vocabs, bert_dim, rel_dim, dropout)
            self.tokenizer = tokenizer
        else:
            self.word_encoder = WordEncoder(vocabs,
                                            word_char_dim, word_dim, pos_dim, ner_dim,
                                            embed_dim, cnn_filters, char2word_dim, dropout, pretrained_file)

            self.concept_encoder = ConceptEncoder(vocabs,
                                                  concept_char_dim, concept_dim, embed_dim,
                                                  cnn_filters, char2concept_dim, dropout, pretrained_file)
            self.snt_encoder = Transformer(snt_layers, embed_dim, ff_embed_dim, num_heads, dropout)
            self.graph_encoder = Transformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout,
                                             with_external=True,
                                             weights_dropout=False)
            self.embed_dim = embed_dim
            self.embed_scale = math.sqrt(embed_dim)
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
            self.word_embed_layer_norm = nn.LayerNorm(embed_dim)
            self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
            self.self_attn_mask = SelfAttentionMask()
            self.decoder = DecodeLayer(vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, concept_dim,
                                       rel_dim,
                                       dropout, joint_arc_concept=joint_arc_concept, joint_rel=joint_rel,
                                       external_biaffine=external_biaffine, optimize_every_layer=optimize_every_layer,
                                       levi_graph=levi_graph)
            self.probe_generator = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        encoder = encoder or kwargs.get('bert_encoder', None)
        self.bert_encoder: ContextualWordEmbeddingModule = encoder
        if encoder is not None:
            bert_dim = encoder.get_output_dim() if hasattr(encoder, 'get_output_dim') else encoder.config.hidden_size
            self.bert_adaptor = nn.Linear(bert_dim, embed_dim)
        if bart:
            bart: BartModel = self.bert_encoder.transformer
            # noinspection PyTypeChecker
            layers = nn.ModuleList(
                [CustomizedDecoderLayer(bart.config) for _ in range(bart.config.decoder_layers)]
            )  # type: List[CustomizedDecoderLayer]
            layers.load_state_dict(bart.decoder.layers.state_dict())
            last_layer: CustomizedDecoderLayer = layers[-1]
            last_layer.self_attn.dropout = 0
            last_layer.encoder_attn.dropout = 0
            bart.decoder.layers = layers
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'probe_generator'):
            nn.init.normal_(self.probe_generator.weight, std=0.02)
            nn.init.constant_(self.probe_generator.bias, 0.)

    def encode_step(self, tok, lem, pos, ner, word_char):
        word_repr = self.embed_scale * self.word_encoder(word_char, tok, lem, pos, ner) + self.embed_positions(tok)
        word_repr = self.word_embed_layer_norm(word_repr)
        word_mask = torch.eq(lem, self.vocabs['lem'].padding_idx)

        word_repr = self.snt_encoder(word_repr, self_padding_mask=word_mask)

        probe = torch.tanh(self.probe_generator(word_repr[:1]))
        word_repr = word_repr[1:]
        word_mask = word_mask[1:]
        return word_repr, word_mask, probe

    def encode_step_with_bert(self, tok, lem, pos, ner, word_char, batch):
        if isinstance(self.bert_encoder, ContextualWordEmbeddingModule):
            bert_embed = self.bert_encoder(batch)
            # Not sure why the original paper has to set [CLS] to 0
            bert_embed[:, 0, :] = 0
        else:
            bert_embed, _ = self.bert_encoder(batch['bert_token'], batch['token_subword_index'])
        word_repr = self.word_encoder(word_char, tok, lem, pos, ner)
        bert_embed = bert_embed.transpose(0, 1)
        word_repr = word_repr + self.bert_adaptor(bert_embed)
        word_repr = self.embed_scale * word_repr + self.embed_positions(tok)
        word_repr = self.word_embed_layer_norm(word_repr)
        word_mask = torch.eq(lem, self.lem_pad_idx)

        word_repr = self.snt_encoder(word_repr, self_padding_mask=word_mask)

        probe = torch.tanh(self.probe_generator(word_repr[:1]))
        word_repr = word_repr[1:]
        word_mask = word_mask[1:]
        return word_repr, word_mask, probe

    def predict(self, batch, beam_size, max_time_step, min_time_step=1):
        with torch.no_grad():
            if self.squeeze or self.bart:
                mem_dict = {
                    'local_idx2token': batch['local_idx2token'],
                    'copy_seq': batch['copy_seq'],
                    'token': batch['token'],
                    'tokenizer': self.tokenizer,
                }
                if self.bart:
                    extra = {
                        'token_token_span': batch['token_token_span']
                    }
                    bart: BartModel = self.bert_encoder.transformer
                    encoder_attention_mask, encoder_hidden_states = self.run_bart_encoder(bart, batch)
                    extra['encoder_attention_mask'] = encoder_attention_mask
                    extra['encoder_hidden_states'] = encoder_hidden_states
                    extra = dict((k, v.transpose_(0, 1)) for k, v in extra.items())
                    mem_dict.update(extra)

                init_state_dict = {}
                init_hyp = Hypothesis(init_state_dict, [DUM], 0.)
                batch_size = batch['copy_seq'].size(1)
                beams = [Beam(beam_size, min_time_step, max_time_step, [init_hyp], self.device) for i in
                         range(batch_size)]
                search_by_batch(self, beams, mem_dict)
            else:
                if self.bert_encoder is not None:
                    word_repr, word_mask, probe = self.encode_step_with_bert(batch['tok'], batch['lem'], batch['pos'],
                                                                             batch['ner'], batch['word_char'],
                                                                             batch=batch)
                else:
                    word_repr, word_mask, probe = self.encode_step(batch['tok'], batch['lem'], batch['pos'],
                                                                   batch['ner'],
                                                                   batch['word_char'])
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
        if self.squeeze or self.bart:
            inp['copy_seq'] = mem_dict['copy_seq']
            if self.bart:
                inp['encoder_attention_mask'] = mem_dict['encoder_attention_mask'].transpose(0, 1)
                inp['encoder_hidden_states'] = mem_dict['encoder_hidden_states'].transpose(0, 1)
                inp['token_token_span'] = mem_dict['token_token_span'].transpose(0, 1).contiguous()
            conc_ll, arc_ll, rel_ll = self.forward(inp)
            if rel_ll is None:
                rel_ll = torch.ones(arc_ll.shape + (len(self.vocabs['rel']),), device=arc_ll.device)
        else:
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

    @staticmethod
    def pool_subtoken(last_hidden_state, token_span, concept_mask, concept_lens, max_concept_len):
        last_hidden_state = pick_tensor_for_each_token(last_hidden_state, token_span, True)
        concept_hidden = torch.zeros((concept_mask.size(0), max_concept_len, last_hidden_state.size(-1),),
                                     device=concept_mask.device)
        concept_hidden[lengths_to_mask(concept_lens, max_concept_len)] = last_hidden_state[concept_mask]
        return concept_hidden

    def run_single_transformer(self, transformer: BertModel, batch: dict):

        input_ids = batch['token_and_concept_input_ids']
        token_span = batch['token_and_concept_token_span']
        attention_mask = batch['attention_mask']
        concept_mask = batch['concept_mask']
        token_type_ids = batch['token_type_ids']
        # kv_embedding = transformer.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        outputs = transformer(input_ids,
                              attention_mask=attention_mask,
                              return_dict=True,
                              output_attentions=True,
                              token_type_ids=token_type_ids,
                              output_hidden_states=True,
                              # encoder_hidden_states=kv_embedding
                              )
        second_to_last_hidden_state = outputs.hidden_states[-2]
        last_hidden_state = outputs.last_hidden_state

        concept_lens = concept_mask.sum(1)
        max_concept_len = concept_lens.max()
        curr_concept = self.pool_subtoken(second_to_last_hidden_state, token_span, concept_mask, concept_lens,
                                          max_concept_len)
        next_concept = self.pool_subtoken(last_hidden_state, token_span, concept_mask, concept_lens, max_concept_len)
        return curr_concept, next_concept, outputs.attentions[-1]

    def forward(self, batch):
        def square_average(h, span):
            y = pick_tensor_for_each_token(h, span, True)
            y = y.transpose_(1, 2)
            y = pick_tensor_for_each_token(y, span, True)
            y = y.transpose_(1, 2)
            return y

        def square_mask(mask: torch.Tensor):
            square = mask.unsqueeze(-1).expand(-1, -1, mask.size(-1))
            return square & square.transpose(1, 2)

        def pick_last_concept(src_, last_concept_3d_):
            return src_.gather(0, last_concept_3d_.expand([-1, -1, src_.size(-1)]))

        if self.squeeze:
            training = self.training
            if isinstance(self.bert_encoder, ContextualWordEmbeddingModule):
                curr_hidden, next_hidden, attention = self.run_single_transformer(self.bert_encoder.transformer, batch)
                alignment, arc_attention = attention[:, 0, :, :], attention[:, 1:, :, :].max(dim=1)[0]
                token_span = batch['token_and_concept_token_span']
                alignment = square_average(alignment, token_span)
                arc_attention = square_average(arc_attention, token_span)
                batch_size, src_len, _ = next_hidden.size()
                input_len = token_span.size(1)
                snt_len = batch['snt_len'].max()
                # [CLS] snt_len [SEP] [DUMMY] concep_len (src_len)
                concept_mask = batch['concept_mask']
                token_mask = batch['token_mask']

                masked_concept_alignment = torch.zeros((batch_size, src_len, input_len), device=next_hidden.device)
                concept_len = concept_mask.sum(1)
                concept_only_mask = lengths_to_mask(concept_len)
                masked_concept_alignment[concept_only_mask] = alignment[concept_mask]
                masked_concept_token_alignment = torch.zeros((batch_size, snt_len, src_len), device=next_hidden.device)
                masked_concept_alignment.transpose_(1, 2)
                masked_concept_token_alignment[lengths_to_mask(token_mask.sum(1))] = masked_concept_alignment[
                    token_mask]
                masked_concept_token_alignment = masked_concept_token_alignment.permute([2, 0, 1])

                square_concept_mask = square_mask(concept_mask)
                masked_arc_attention = torch.zeros((batch_size, src_len, src_len), device=next_hidden.device)
                square_concept_only_mask = square_mask(concept_only_mask)
                masked_arc_attention[square_concept_only_mask] = arc_attention[square_concept_mask]
                masked_arc_attention.transpose_(0, 1)

                next_hidden.transpose_(0, 1)
                curr_hidden.transpose_(0, 1)
                if not training:
                    last_concept_offset = batch['last_concept_offset']
                    last_concept_3d = last_concept_offset.unsqueeze(0).unsqueeze(-1)

                    masked_concept_token_alignment = pick_last_concept(masked_concept_token_alignment, last_concept_3d)
                    masked_arc_attention = pick_last_concept(masked_arc_attention, last_concept_3d)
                    next_hidden = pick_last_concept(next_hidden, last_concept_3d)
                    src_len = 1
                outputs = self.arc_concept_decoder(masked_concept_token_alignment, masked_arc_attention, None,
                                                   next_hidden,
                                                   batch['copy_seq'], batch.get('concept_out', None),
                                                   batch['rel'][1:] if training else None, batch_size, src_len,
                                                   work=not training)
                rel_loss = self.relation_generator(next_hidden, curr_hidden,
                                                   target_rel=batch['rel'][1:] if training else None, work=not training)
                return self.ret_squeeze_or_bart(batch, outputs, rel_loss, training)
            else:
                raise NotImplementedError()
                bert_embed, _ = self.bert_encoder(batch['bert_token'], batch['token_subword_index'])
        elif self.bart:
            training = self.training
            bart: BartModel = self.bert_encoder.transformer

            # Run encoder if not cached
            if training:
                encoder_attention_mask, encoder_hidden_states = self.run_bart_encoder(bart, batch)
            else:
                encoder_attention_mask = batch['encoder_attention_mask']
                encoder_hidden_states = batch['encoder_hidden_states']

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            decoder = bart.decoder
            concept_input_ids = batch['concept_input_ids']
            decoder_padding_mask = concept_input_ids == self.tokenizer.tokenizer.pad_token_id
            decoder_mask = batch['decoder_mask']
            decoder_causal_mask = torch.zeros_like(decoder_mask, device=decoder_mask.device, dtype=torch.float)
            decoder_causal_mask.masked_fill_(~decoder_mask, float('-inf'))
            decoder_causal_mask.unsqueeze_(1)
            decoder_outputs = decoder(
                concept_input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                decoder_padding_mask,
                decoder_causal_mask=decoder_causal_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            next_hidden = decoder_outputs.last_hidden_state
            token_span = batch['token_token_span']
            concept_span = batch['concept_token_span']
            next_hidden = pick_tensor_for_each_token(next_hidden, concept_span, True)
            arc_attention, alignment = decoder_outputs.attentions[-1]
            arc_attention = arc_attention.max(dim=1)[0]
            alignment = alignment.max(dim=1)[0]
            alignment = pick_tensor_for_each_token(alignment, concept_span, True)
            alignment.transpose_(1, 2)
            alignment = pick_tensor_for_each_token(alignment, token_span, True)
            # Strip [CLS] as it won't be copied anyway
            alignment = alignment[:, 1:, :]
            arc_attention = square_average(arc_attention, concept_span)
            batch_size, src_len, _ = next_hidden.size()
            next_hidden.transpose_(0, 1)
            alignment = alignment.permute(2, 0, 1)
            arc_attention.transpose_(0, 1)

            curr_hidden = decoder_outputs.hidden_states[-2]
            curr_hidden = pick_tensor_for_each_token(curr_hidden, concept_span, True)
            curr_hidden.transpose_(0, 1)

            if not training:
                last_concept_offset = batch['last_concept_offset']
                last_concept_3d = last_concept_offset.unsqueeze(0).unsqueeze(-1)

                alignment = pick_last_concept(alignment, last_concept_3d)
                arc_attention = pick_last_concept(arc_attention, last_concept_3d)
                next_hidden = pick_last_concept(next_hidden, last_concept_3d)
                src_len = 1
            outputs = self.arc_concept_decoder(alignment, arc_attention, None,
                                               next_hidden,
                                               batch['copy_seq'], batch.get('concept_out', None),
                                               batch['rel'][1:] if training else None, batch_size, src_len,
                                               work=not training)
            rel_loss = self.relation_generator(next_hidden, curr_hidden,
                                               target_rel=batch['rel'][1:] if training else None, work=not training)
            return self.ret_squeeze_or_bart(batch, outputs, rel_loss, training)
        if self.bert_encoder is not None:
            word_repr, word_mask, probe = self.encode_step_with_bert(batch['tok'], batch['lem'], batch['pos'],
                                                                     batch['ner'],
                                                                     batch['word_char'], batch=batch)
        else:
            word_repr, word_mask, probe = self.encode_step(batch['tok'], batch['lem'], batch['pos'], batch['ner'],
                                                           batch['word_char'])
        concept_repr = self.embed_scale * self.concept_encoder(batch['concept_char_in'],
                                                               batch['concept_in']) + self.embed_positions(
            batch['concept_in'])
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_repr = F.dropout(concept_repr, p=self.dropout, training=self.training)
        concept_mask = torch.eq(batch['concept_in'], self.concept_pad_idx)
        attn_mask = self.self_attn_mask(batch['concept_in'].size(0))
        # concept_repr = self.graph_encoder(concept_repr,
        #                          self_padding_mask=concept_mask, self_attn_mask=attn_mask,
        #                          external_memories=word_repr, external_padding_mask=word_mask)
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
                                                        attn_mask, \
                                                        batch['copy_seq'], target=batch['concept_out'],
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
            return (
                       concept_loss.mean(), concept_correct,
                       concept_total), arc_loss.mean(), rel_out, graph_arc_loss.mean()
        return concept_loss.mean(), arc_loss.mean(), rel_loss.mean(), graph_arc_loss.mean()

    def run_bart_encoder(self, bart, batch):
        # get encoder and store encoder outputs
        encoder = bart.encoder
        input_ids = batch['token_input_ids']
        attention_mask = input_ids != self.tokenizer.tokenizer.pad_token_id
        encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)
        encoder_hidden_states = encoder_outputs[0]
        return attention_mask, encoder_hidden_states

    def ret_squeeze_or_bart(self, data, outputs, rel_loss, training):
        if not training:
            return outputs[:-1] + (rel_loss,)
        (concept_loss, concept_correct, concept_total), arc_loss, _ = outputs
        graph_arc_loss = None
        rel_loss, rel_correct, rel_total = rel_loss
        concept_mask = torch.eq(data['concept_in'], self.concept_pad_idx)
        concept_tot = concept_mask.size(0) - concept_mask.float().sum(0)
        concept_loss = concept_loss / concept_tot
        arc_loss = arc_loss / concept_tot
        rel_loss = rel_loss / concept_tot
        return (concept_loss.mean(), concept_correct, concept_total), arc_loss.mean(), (
            rel_loss.mean(), rel_correct, rel_total), graph_arc_loss

    @property
    def device(self):
        return next(self.parameters()).device

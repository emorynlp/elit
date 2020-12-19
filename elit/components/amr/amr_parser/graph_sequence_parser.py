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
import functools
import logging
import os
import traceback
from typing import Union, List, Dict

import torch
from torch.utils.data import DataLoader

from elit.common.constant import CLS
from elit.common.dataset import SortingSampler, PrefetchDataLoader
from elit.common.structure import History
from elit.common.torch_component import TorchComponent
from elit.common.transform import VocabDict
from elit.common.vocab import VocabWithFrequency
from elit.components.amr.amr_parser.adam import AdamWeightDecayOptimizer
from elit.components.amr.amr_parser.amrio import make_vocab
from elit.components.amr.amr_parser.data import END, DUM, NIL, REL
from elit.components.amr.amr_parser.parser_model import GraphSequenceAbstractMeaningRepresentationModel
from elit.components.amr.amr_parser.postprocess import PostProcessor
from elit.components.amr.amr_parser.work import parse_data
from elit.datasets.parsing.amr import AbstractMeaningRepresentationDataset, append_bos, batchify, get_concepts, \
    linearize
from elit.layers.transformers.utils import build_optimizer_scheduler_with_transformer
from elit.metrics.amr.smatch_eval import smatch_eval, post_process
from elit.metrics.f1 import F1_
from elit.utils.io_util import eprint
from elit.utils.time_util import CountdownTimer
from elit.utils.util import merge_locals_kwargs, merge_list_of_dict, merge_dict


class GraphSequenceAbstractMeaningRepresentationParser(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """
        An AMR parser implementing Cai and Lam (2020) and my unpublished models.

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)
        self.model: GraphSequenceAbstractMeaningRepresentationModel = self.model

    def build_optimizer(self,
                        trn,
                        epochs,
                        lr,
                        adam_epsilon,
                        weight_decay,
                        warmup_steps,
                        transformer_lr,
                        gradient_accumulation,
                        **kwargs):
        model = self.model
        if self.config.squeeze and False:
            num_training_steps = len(trn) * epochs // gradient_accumulation
            optimizer, scheduler = build_optimizer_scheduler_with_transformer(model,
                                                                              model.bert_encoder,
                                                                              lr,
                                                                              transformer_lr,
                                                                              num_training_steps,
                                                                              warmup_steps,
                                                                              weight_decay,
                                                                              adam_epsilon)
        else:
            weight_decay_params = []
            no_weight_decay_params = []
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            for name, param in model.named_parameters():
                if name.endswith('bias') or 'layer_norm' in name or any(nd in name for nd in no_decay):
                    no_weight_decay_params.append(param)
                else:
                    weight_decay_params.append(param)
            grouped_params = [{'params': weight_decay_params, 'weight_decay': weight_decay},
                              {'params': no_weight_decay_params, 'weight_decay': 0.}]
            optimizer = AdamWeightDecayOptimizer(grouped_params, lr, betas=(0.9, 0.999), eps=adam_epsilon)
            lr_scale = self.config.lr_scale
            embed_dim = self.config.embed_dim
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda steps: lr_scale * embed_dim ** -0.5 * min((steps + 1) ** -0.5,
                                                                 (steps + 1) * (warmup_steps ** -1.5)))
        return optimizer, scheduler

    def build_criterion(self, **kwargs):
        pass

    def build_metric(self, **kwargs):
        pass

    def execute_training_loop(self, trn: PrefetchDataLoader, dev: PrefetchDataLoader, epochs, criterion, optimizer,
                              metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, dev_data=None, gradient_accumulation=1,
                              **kwargs):
        best_epoch, best_metric = 0, -1
        timer = CountdownTimer(epochs)
        history = History()
        try:
            for epoch in range(1, epochs + 1):
                logger.info(f"[yellow]Epoch {epoch} / {epochs}:[/yellow]")
                trn = self.fit_dataloader(trn, criterion, optimizer, metric, logger, ratio_width=ratio_width,
                                          gradient_accumulation=gradient_accumulation, history=history,
                                          save_dir=save_dir)
                report = f'{timer.elapsed_human}/{timer.total_time_human}'
                if epoch % self.config.eval_every == 0 or epoch == epochs:
                    metric = self.evaluate_dataloader(dev, logger, dev_data, ratio_width=ratio_width, save_dir=save_dir,
                                                      use_fast=True)
                    if metric > best_metric:
                        self.save_weights(save_dir)
                        best_metric = metric
                        best_epoch = epoch
                        report += ' [red]saved[/red]'
                timer.log(report, ratio_percentage=False, newline=True, ratio=False)
            if best_epoch and best_epoch != epochs:
                logger.info(f'Restored the best model with {best_metric} saved {epochs - best_epoch} epochs ago')
                self.load_weights(save_dir)
        finally:
            trn.close()
            dev.close()

    def fit_dataloader(self,
                       trn: PrefetchDataLoader,
                       criterion,
                       optimizer,
                       metric,
                       logger: logging.Logger,
                       gradient_accumulation=1,
                       ratio_width=None,
                       history=None,
                       save_dir=None,
                       **kwargs):
        self.model.train()
        num_training_steps = len(trn) * self.config.epochs // gradient_accumulation
        shuffle_sibling_steps = self.config.shuffle_sibling_steps
        if isinstance(shuffle_sibling_steps, float):
            shuffle_sibling_steps = int(shuffle_sibling_steps * num_training_steps)
        timer = CountdownTimer(len(
            [i for i in range(history.num_mini_batches + 1, history.num_mini_batches + len(trn) + 1) if
             i % gradient_accumulation == 0]))
        total_loss = 0
        optimizer, scheduler = optimizer
        correct_conc, total_conc, correct_rel, total_rel = 0, 0, 0, 0
        for idx, batch in enumerate(trn):
            loss = self.compute_loss(batch)
            if self.config.joint_arc_concept or self.model.squeeze or self.config.bart:
                loss, (concept_correct, concept_total), rel_out = loss
                correct_conc += concept_correct
                total_conc += concept_total
                if rel_out is not None:
                    rel_correct, rel_total = rel_out
                    correct_rel += rel_correct
                    total_rel += rel_total
            loss /= gradient_accumulation
            # loss = loss.sum()  # For data parallel
            loss.backward()
            total_loss += loss.item()
            history.num_mini_batches += 1
            if history.num_mini_batches % gradient_accumulation == 0:
                self._step(optimizer, scheduler)
                metric = ''
                if self.config.joint_arc_concept or self.model.squeeze or self.model.bart:
                    metric = f' Concept acc: {correct_conc / total_conc:.2%}'
                    if not self.config.levi_graph:
                        metric += f' Relation acc: {correct_rel / total_rel:.2%}'
                timer.log(
                    f'loss: {total_loss / (timer.current + 1):.4f} lr: {optimizer.param_groups[0]["lr"]:.2e}' + metric,
                    ratio_percentage=None, ratio_width=ratio_width, logger=logger)

                if history.num_mini_batches // gradient_accumulation == shuffle_sibling_steps:
                    trn.batchify = self.build_batchify(self.device, shuffle=True, shuffle_sibling=False)
                    timer.print(
                        f'Switched to [bold]deterministic order[/bold] after {shuffle_sibling_steps} steps',
                        newline=True)
            del loss
        return trn

    def _step(self, optimizer, scheduler):
        if self.config.grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
        optimizer.step()
        # model = self.model
        # print(mean_model(model))
        optimizer.zero_grad()
        scheduler.step()

    def update_metrics(self, batch: dict, prediction: Union[Dict, List], metrics):
        if isinstance(prediction, dict):
            prediction = prediction['prediction']
        assert len(prediction) == len(batch['ner'])
        for pred, gold in zip(prediction, batch['ner']):
            metrics(set(pred), set(gold))

    def compute_loss(self, batch):
        # debug
        # gold = torch.load('/home/hhe43/amr_gs/batch.pt', map_location=self.device)
        # self.debug_assert_batch_equal(batch, gold)
        # set_seed()
        # end debug
        concept_loss, arc_loss, rel_loss, graph_arc_loss = self.model(batch)
        if self.config.joint_arc_concept or self.config.squeeze or self.config.bart:
            concept_loss, concept_correct, concept_total = concept_loss
            if rel_loss is not None:
                rel_loss, rel_correct, rel_total = rel_loss
                loss = concept_loss + arc_loss + rel_loss
                rel_acc = (rel_correct, rel_total)
            else:
                loss = concept_loss + arc_loss
                rel_acc = None
            return loss, (concept_correct, concept_total), rel_acc
        loss = concept_loss + arc_loss + rel_loss
        return loss

    def debug_assert_batch_equal(self, batch, gold):
        # assert torch.equal(batch['token_input_ids'], gold['bert_token'])
        for k, v in gold.items():
            pred = batch.get(k, None)
            if pred is not None:
                if isinstance(v, torch.Tensor) and not torch.equal(pred, v):
                    assert torch.equal(pred, v), f'{k} not equal'

    @torch.no_grad()
    def evaluate_dataloader(self, data: PrefetchDataLoader, logger, input, output=False, ratio_width=None,
                            save_dir=None,
                            use_fast=False,
                            test=False,
                            **kwargs):
        self.model.eval()
        pp = PostProcessor(self.vocabs['rel'])
        if not output:
            output = os.path.join(save_dir, os.path.basename(input) + '.pred')
        # Squeeze tokens and concepts into one transformer basically reduces the max num of inputs it can handle
        parse_data(self.model, pp, data, input, output, max_time_step=80 if self.model.squeeze else 100)
        # noinspection PyBroadException
        try:
            output = post_process(output, amr_version=self.config.get('amr_version', '2.0'))
            scores = smatch_eval(output, input.replace('.features.preproc', ''), use_fast=use_fast)
        except Exception:
            eprint(f'Evaluation failed due to the following error:')
            traceback.print_exc()
            eprint('As smatch usually fails on erroneous outputs produced at early epochs, '
                   'it might be OK to ignore it. Now `nan` will be returned as the score.')
            scores = F1_(float("nan"), float("nan"), float("nan"))
        if logger:
            header = f'{len(data)}/{len(data)}'
            if not ratio_width:
                ratio_width = len(header)
            logger.info(header.rjust(ratio_width) + f' {scores}')
        if test:
            data.close()
        return scores

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        transformer = self.config.encoder.module()
        model = GraphSequenceAbstractMeaningRepresentationModel(self.vocabs,
                                                                **merge_dict(self.config, overwrite=True,
                                                                             encoder=transformer),
                                                                tokenizer=self.config.encoder.transform())
        # self.model = model
        # self.debug_load()
        return model

    def debug_load(self):
        model = self.model
        states = torch.load('/home/hhe43/amr_gs/model.pt', map_location=self.device)
        model.load_state_dict(states, strict=False)

    def build_dataloader(self, data, batch_size, shuffle=False, device=None, logger: logging.Logger = None,
                         gradient_accumulation=1,
                         batch_max_tokens=None,
                         **kwargs) -> DataLoader:
        dataset, lens = self.build_dataset(data, logger, training=shuffle)
        if batch_max_tokens:
            batch_max_tokens //= gradient_accumulation
        if not shuffle:
            batch_max_tokens //= 2
        sampler = SortingSampler(lens, batch_size=None, batch_max_tokens=batch_max_tokens, shuffle=shuffle)
        dataloader = PrefetchDataLoader(DataLoader(batch_sampler=sampler, dataset=dataset,
                                                   collate_fn=merge_list_of_dict,
                                                   num_workers=0), batchify=self.build_batchify(device, shuffle))
        return dataloader

    def build_batchify(self, device, shuffle, shuffle_sibling=None):
        if shuffle_sibling is None:
            shuffle_sibling = shuffle
        return functools.partial(batchify, vocabs=self.vocabs, squeeze=self.config.get('squeeze', None),
                                 tokenizer=self.config.encoder.transform(),
                                 levi_graph=self.config.get('levi_graph', False),
                                 bart=self.config.get('bart', False),
                                 extra_arc=self.config.get('extra_arc', False),
                                 unk_rate=self.config.unk_rate if shuffle else 0,
                                 shuffle_sibling=shuffle_sibling, device=device)

    def build_dataset(self, data, logger: logging.Logger = None, training=True):
        dataset = AbstractMeaningRepresentationDataset(data, generate_idx=not training)
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
            self.vocabs.lock()
            self.vocabs.summary(logger)
        lens = [len(x['token']) + len(x['amr']) for x in dataset]
        dataset.append_transform(functools.partial(get_concepts, vocab=self.vocabs.predictable_concept,
                                                   rel_vocab=self.vocabs.rel if self.config.get('separate_rel',
                                                                                                False) else None))
        dataset.append_transform(append_bos)
        # Tokenization will happen in batchify
        if not self.config.get('squeeze', None):
            dataset.append_transform(self.config.encoder.transform())
        if isinstance(data, str):
            dataset.purge_cache()
            timer = CountdownTimer(len(dataset))
            for each in dataset:
                timer.log('Caching samples [blink][yellow]...[/yellow][/blink]')
        return dataset, lens

    def build_vocabs(self, dataset, logger: logging.Logger = None, **kwargs):
        # debug
        # self.load_vocabs('/home/hhe43/elit/data/model/amr2.0/convert/')
        # return
        # collect concepts and relations
        conc = []
        rel = []
        predictable_conc = []  # concepts that are not able to generate by copying lemmas ('multi-sentence', 'sense-01')
        tokens = []
        lemmas = []
        poses = []
        ners = []
        repeat = 10
        levi_graph = self.config.get('levi_graph', False)
        separate_rel = self.config.separate_rel
        timer = CountdownTimer(repeat * len(dataset))
        for i in range(repeat):
            # run 10 times random sort to get the priorities of different types of edges
            for sample in dataset:
                amr, lem, tok, pos, ner = sample['amr'], sample['lemma'], sample['token'], sample['pos'], sample['ner']
                if levi_graph == 'kahn':
                    concept, edge = amr.to_levi()
                else:
                    concept, edge, not_ok = amr.root_centered_sort()
                if levi_graph is True:
                    concept, edge = linearize(concept, edge, NIL, prefix=REL)
                lexical_concepts = set()
                for lemma in lem:
                    lexical_concepts.add(lemma + '_')
                    lexical_concepts.add(lemma)

                if i == 0:
                    if separate_rel:
                        edge = [(c,) for c in concept if c.startswith(REL)]
                        concept = [c for c in concept if not c.startswith(REL)]
                    predictable_conc.append([c for c in concept if c not in lexical_concepts])
                    conc.append(concept)
                    tokens.append(tok)
                    lemmas.append(lem)
                    poses.append(pos)
                    ners.append(ner)
                    rel.append([e[-1] for e in edge])
                timer.log('Building vocabs [blink][yellow]...[/yellow][/blink]')

        # make vocabularies
        token_vocab, token_char_vocab = make_vocab(tokens, char_level=True)
        lemma_vocab, lemma_char_vocab = make_vocab(lemmas, char_level=True)
        pos_vocab = make_vocab(poses)
        ner_vocab = make_vocab(ners)
        conc_vocab, conc_char_vocab = make_vocab(conc, char_level=True)

        predictable_conc_vocab = make_vocab(predictable_conc)
        num_predictable_conc = sum(len(x) for x in predictable_conc)
        num_conc = sum(len(x) for x in conc)
        rel_vocab = make_vocab(rel)
        logger.info(
            f'Predictable concept coverage {num_predictable_conc} / {num_conc} = {num_predictable_conc / num_conc:.2%}')
        vocabs = self.vocabs
        vocab_min_freq = self.config.get('vocab_min_freq', 5)
        vocabs.token = VocabWithFrequency(token_vocab, vocab_min_freq, specials=[CLS])
        vocabs.lemma = VocabWithFrequency(lemma_vocab, vocab_min_freq, specials=[CLS])
        vocabs.pos = VocabWithFrequency(pos_vocab, vocab_min_freq, specials=[CLS])
        vocabs.ner = VocabWithFrequency(ner_vocab, vocab_min_freq, specials=[CLS])
        vocabs.predictable_concept = VocabWithFrequency(predictable_conc_vocab, vocab_min_freq, specials=[DUM, END])
        vocabs.concept = VocabWithFrequency(conc_vocab, vocab_min_freq, specials=[DUM, END])
        vocabs.rel = VocabWithFrequency(rel_vocab, vocab_min_freq * 10, specials=[NIL])
        vocabs.word_char = VocabWithFrequency(token_char_vocab, vocab_min_freq * 20, specials=[CLS, END])
        vocabs.concept_char = VocabWithFrequency(conc_char_vocab, vocab_min_freq * 20, specials=[CLS, END])
        if separate_rel:
            vocabs.concept_and_rel = VocabWithFrequency(conc_vocab + rel_vocab, vocab_min_freq,
                                                        specials=[DUM, END, NIL])
        # if levi_graph:
        #     # max = 993
        #     tokenizer = self.config.encoder.transform()
        #     rel_to_unused = dict()
        #     for i, rel in enumerate(vocabs.rel.idx_to_token):
        #         rel_to_unused[rel] = f'[unused{i + 100}]'
        #     tokenizer.rel_to_unused = rel_to_unused

    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        pass

    def fit(self, trn_data, dev_data, save_dir,
            encoder,
            batch_size=None,
            batch_max_tokens=17776,
            epochs=1000,
            gradient_accumulation=4,
            char2concept_dim=128,
            char2word_dim=128,
            cnn_filters=((3, 256),),
            concept_char_dim=32,
            concept_dim=300,
            dropout=0.2,
            embed_dim=512,
            eval_every=20,
            ff_embed_dim=1024,
            graph_layers=2,
            inference_layers=4,
            lr_scale=1.0,
            ner_dim=16,
            num_heads=8,
            pos_dim=32,
            pretrained_file=None,
            rel_dim=100,
            snt_layers=4,
            start_rank=0,
            unk_rate=0.33,
            warmup_steps=2000,
            with_bert=True,
            word_char_dim=32,
            word_dim=300,
            lr=1.,
            transformer_lr=None,
            adam_epsilon=1e-6,
            weight_decay=1e-4,
            grad_norm=1.0,
            joint_arc_concept=False,
            joint_rel=False,
            external_biaffine=False,
            optimize_every_layer=False,
            squeeze=False,
            levi_graph=False,
            separate_rel=False,
            extra_arc=False,
            bart=False,
            shuffle_sibling_steps=50000,
            vocab_min_freq=5,
            amr_version='2.0',
            devices=None, logger=None, seed=None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        if hasattr(self, 'vocabs'):
            self.vocabs = VocabDict()
            self.vocabs.load_vocabs(save_dir, filename, VocabWithFrequency)

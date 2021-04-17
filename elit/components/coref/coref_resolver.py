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
import logging
from typing import Union, List, Callable, Dict, Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from elit.utils.torch_util import cuda_devices
from elit.common.torch_component import TorchComponent
from elit.components.coref.coref_model import MlCorefModel
from elit.components.coref.tensorizer import Tensorizer, CorefInstance
from elit.components.coref.dto import CorefInput, CorefOutput


class CoreferenceResolver(TorchComponent):
    """ Coreference resolution component.

    This component handles either document coreference or online coreference, based on configuration.
    Currently only inference is supported; training-related is not available.
    Supported operations: build_model(), to(), load(), predict(), validate_input(), available_genres, speaker_id_range
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: MlCorefModel = None
        self.tensorizer: Tensorizer = None
        self.torch_device: torch.device = torch.device('cpu')

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        """ Build from config; ignore overridden values. """
        model = MlCorefModel(self.config, self.torch_device)
        return model

    def to(self, devices=Union[int, float, List[int], Dict[str, Union[int, torch.device]]],
           logger: logging.Logger = None):
        """ For inference, can only move to CPU or one GPU. """
        if devices == -1 or devices == [-1]:
            devices = []
        elif isinstance(devices, (int, float)) or devices is None:
            devices = cuda_devices(devices)
        if isinstance(devices, list) and len(devices) > 1:
            raise ValueError(f'Invalid devices{devices}; at most one GPU can be accepted for inference')
        
        super(CoreferenceResolver, self).to(devices=devices, logger=logger)

        self.torch_device = torch.device('cpu' if len(devices) == 0 else f'cuda:{devices[0]}')
        self.model.device = self.torch_device
        self.model.to(self.torch_device)  # Double check; if already on device (should be), no action

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        pass  # Vocab is handled inside self.tensorizer

    def load(self, save_dir: str, devices=None, **kwargs):
        super(CoreferenceResolver, self).load(save_dir, devices, **kwargs)
        self.tensorizer = Tensorizer(self.config)

    @property
    def is_online(self) -> bool:
        return self.config['online']

    @property
    def available_genres(self) -> List[str]:
        return self.tensorizer.genres

    @property
    def default_genre(self) -> str:
        if self.is_online:
            return 'en' if 'en' in self.available_genres else self.available_genres[0]
        else:
            return 'nw' if 'nw' in self.available_genres else self.available_genres[0]

    @property
    def speaker_id_range(self) -> Tuple[int, int]:
        return 1, self.tensorizer.max_speakers

    def validate_input(self, coref_input: CorefInput) -> Optional[str]:
        if coref_input.speaker_ids:
            speaker_ids = coref_input.speaker_ids
            if isinstance(speaker_ids, list) and len(speaker_ids) != len(coref_input.doc_or_uttr):
                return 'speaker_ids has invalid length'
            if isinstance(speaker_ids, int):
                speaker_ids = [speaker_ids]
            if not all(1 <= spk <= self.tensorizer.max_speakers for spk in speaker_ids):
                return f'speaker_id exceeds range {self.speaker_id_range}'

        if coref_input.genre:
            if coref_input.genre not in self.available_genres:
                return f'genre should be in {self.available_genres}'
        else:
            coref_input.genre = self.default_genre

        if coref_input.context:
            context = coref_input.context
            if not context.input_ids or not context.sentence_map \
                    or not context.subtoken_map or not context.uttr_start_idx:
                return 'context should have input_ids, sentence_map, subtoken_map, uttr_start_idx'
            if not all(length == len(context.input_ids) for length in
                       [len(context.input_ids), len(context.sentence_map), len(context.subtoken_map)]):
                return 'context should have equal length for input_ids, sentence_map, subtoken_map'
            if context.mentions is None or context.speaker_ids is None:
                return 'context should have mentions and speaker_ids (can be empty list)'
        return None

    def predict(self, data: CorefInput, **kwargs) -> CorefOutput:
        """
        Prediction for coreference resolution.

        Resolve either document coreference or online coreference based on configuration.
        Args:
            data ():
            **kwargs ():

        Returns:

        """
        error_msg = self.validate_input(data)
        if error_msg:
            return CorefOutput(error_msg=error_msg)

        if self.config['online']:
            output = self._predict_online(data, allow_singleton=kwargs.get('allow_singleton', True),
                                          check_sanitization=kwargs.get('check_sanitization', False))
        else:
            output = self._predict_doc(data)
        return output

    def _predict_doc(self, data: CorefInput) -> Union[CorefOutput, List[List[Tuple[Any]]]]:
        inst: CorefInstance = self.tensorizer.encode_doc(data)
        inputs = {
            'input_ids': inst.input_ids,
            'input_mask': inst.input_mask,
            'speaker_ids': inst.speaker_ids,
            'genre': inst.genre_id,
            'sentence_map': inst.sentence_map
        }
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

        # self.model.eval()
        with torch.no_grad():
            span_starts, span_ends, span_mention_scores, antecedent_idx, antecedent_scores = self.model(**inputs)
        inst = self._get_predicted_clusters(inst, span_starts, span_ends, span_mention_scores, antecedent_idx,
                                            antecedent_scores, inst.uttr_start_idx[-1],
                                            allow_singleton=False, return_prob=data.return_prob)  # No singletons
        output = inst.generate_output(data.doc_or_uttr, verbose=data.verbose, online=False)
        return output

    def _predict_online(self, data: CorefInput, allow_singleton: bool = True,
                        check_sanitization: bool = False) -> Union[CorefOutput, List[List[Tuple[Any]]]]:
        if self.config['mention_loss_coef'] < 1e-6:
            allow_singleton = False

        inst: CorefInstance = self.tensorizer.encode_online(data)
        # Check sanitization
        if data.context is not None and check_sanitization:
            context = data.context
            previous_mentions = [(context.input_ids[m[0]], context.input_ids[m[1]]) for m in context.mentions]
            current_mentions = [(inst.input_ids[m[0]].item(), inst.input_ids[m[1]].item()) for m in inst.mentions]
            if previous_mentions != current_mentions:
                raise RuntimeWarning(f'Mentions not consistent: {previous_mentions} vs. {current_mentions}')

        inputs = {
            'input_ids': inst.input_ids.unsqueeze(0),
            'input_mask': inst.input_mask.unsqueeze(0),
            'speaker_ids': inst.speaker_ids,
            'genre': inst.genre_id,
            'sentence_map': inst.sentence_map,
            'uttr_start_idx': torch.tensor(inst.uttr_start_idx[-1], dtype=torch.long),
            'context_mention_starts': torch.tensor([m[0] for m in inst.mentions], dtype=torch.long),
            'context_mention_ends': torch.tensor([m[1] for m in inst.mentions], dtype=torch.long)
        }
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

        # self.model.eval()
        with torch.no_grad():
            span_starts, span_ends, span_mention_scores, antecedent_idx, antecedent_scores = self.model(**inputs)
        inst = self._get_predicted_clusters(inst, span_starts, span_ends, span_mention_scores, antecedent_idx,
                                            antecedent_scores, inst.uttr_start_idx[-1],
                                            allow_singleton=allow_singleton, return_prob=data.return_prob)
        output = inst.generate_output(data.doc_or_uttr, verbose=data.verbose, online=True)
        return output

    @classmethod
    def _get_predicted_antecedents(cls, antecedent_idx: List[List[int]], antecedent_scores: torch.Tensor):
        predicted_antecedents = []
        for i, idx in enumerate((torch.argmax(antecedent_scores, dim=1) - 1).tolist()):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    @classmethod
    def _get_predicted_clusters(cls, inst: CorefInstance, span_starts, span_ends, span_mention_scores, antecedent_idx,
                                antecedent_scores: torch.Tensor, uttr_start_idx: int,
                                allow_singleton: bool, return_prob: bool) -> CorefInstance:
        subtoken_map = inst.subtoken_map
        span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
        antecedent_idx = antecedent_idx.tolist()
        mentions = set(inst.mentions)  # Subtoken idx
        predicted_clusters, mention_to_cluster_id = [], {}  # Original token idx
        linking_prob = defaultdict(lambda: defaultdict(float))  # Original token idx

        predicted_antecedents = cls._get_predicted_antecedents(antecedent_idx, antecedent_scores)
        for i, predicted_idx in enumerate(predicted_antecedents):
            # Skip context
            if span_starts[i] < uttr_start_idx:
                continue

            mention = (int(span_starts[i]), int(span_ends[i]))
            mention_tok = (subtoken_map[mention[0]], subtoken_map[mention[1]])  # Original token idx
            if mention_tok in mention_to_cluster_id:
                continue  # Skip rare cases where two mentions map to same original tokens

            if return_prob:
                probs = torch.nn.functional.softmax(antecedent_scores[i]).tolist()
                for antecedent_i, prob in zip([i] + antecedent_idx[i], probs):
                    if prob > 0.1:
                        antecedent_tok = (subtoken_map[span_starts[antecedent_i]],subtoken_map[span_ends[antecedent_i]])
                        linking_prob[mention_tok][antecedent_tok] = prob
                        linking_prob[antecedent_tok][mention_tok] = prob

            # Create cluster for new mention; can result in singleton
            if predicted_idx < 0:
                if allow_singleton and span_mention_scores[i] > 0:
                    cluster_id = len(predicted_clusters)
                    predicted_clusters.append([mention_tok])
                    mention_to_cluster_id[mention_tok] = cluster_id
                    mentions.add(mention)
                continue

            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_tok = (subtoken_map[antecedent[0]], subtoken_map[antecedent[1]])  # Original token idx
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent_tok, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent_tok])
                mention_to_cluster_id[antecedent_tok] = antecedent_cluster_id

            predicted_clusters[antecedent_cluster_id].append(mention_tok)
            mention_to_cluster_id[mention_tok] = antecedent_cluster_id

            mentions.add(antecedent)
            mentions.add(mention)

        # Update mentions and clusters
        inst.mentions = sorted(list(mentions))
        inst.clusters = predicted_clusters
        inst.mention_to_cluster_id = mention_to_cluster_id
        inst.linking_prob = linking_prob if return_prob else None
        return inst

    def build_dataloader(self, data, batch_size, shuffle=False, device=None, logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        pass

    def build_optimizer(self, **kwargs):
        pass

    def build_criterion(self, decoder, **kwargs):
        pass

    def build_metric(self, **kwargs):
        pass

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, **kwargs):
        pass

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, **kwargs):
        pass

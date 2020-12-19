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
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Callable, Any
import re
import torch
from torch import nn
from torch.utils.data import DataLoader

import elit
from elit.common.component import Component
from elit.common.constant import IDX
from elit.common.dataset import TransformDataset
from elit.common.structure import SerializableDict, Configurable
from elit.common.transform import VocabDict
from elit.utils.io_util import get_resource, basename_no_ext
from elit.utils.log_util import init_logger, flash
from elit.utils.reflection import classpath_of
from elit.utils.torch_util import cuda_devices, set_seed
from elit.utils.util import merge_dict, isdebugging, merge_list_of_dict


class TorchComponent(Component, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model: Optional[torch.nn.Module] = None
        self.config = SerializableDict(**kwargs)
        self.vocabs = VocabDict()

    def _capture_config(self, locals_: Dict,
                        exclude=(
                                'trn_data', 'dev_data', 'save_dir', 'kwargs', 'self', 'logger', 'verbose',
                                'dev_batch_size', '__class__', 'devices', 'eval_trn')):
        """Save arguments to config

        Args:
          locals_: Dict: 
          exclude:  (Default value = ('trn_data')
          'dev_data': 
          'save_dir': 
          'kwargs': 
          'self': 
          'logger': 
          'verbose': 
          'dev_batch_size': 
          '__class__': 
          'devices'): 

        Returns:

        
        """
        if 'kwargs' in locals_:
            locals_.update(locals_['kwargs'])
        locals_ = dict((k, v) for k, v in locals_.items() if k not in exclude and not k.startswith('_'))
        self.config.update(locals_)
        return self.config

    def save_weights(self, save_dir, filename='model.pt', trainable_only=True, **kwargs):
        model = self.model_
        state_dict = model.state_dict()
        if trainable_only:
            trainable_names = set(n for n, p in model.named_parameters() if p.requires_grad)
            state_dict = dict((n, p) for n, p in state_dict.items() if n in trainable_names)
        torch.save(state_dict, os.path.join(save_dir, filename))

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        save_dir = get_resource(save_dir)
        filename = os.path.join(save_dir, filename)
        # flash(f'Loading model: {filename} [blink]...[/blink][/yellow]')
        self.model_.load_state_dict(torch.load(filename, map_location='cpu'), strict=False)
        # flash('')

    def save_config(self, save_dir, filename='config.json'):
        self._savable_config.save_json(os.path.join(save_dir, filename))

    def load_config(self, save_dir, filename='config.json', **kwargs):
        save_dir = get_resource(save_dir)
        self.config.load_json(os.path.join(save_dir, filename))
        self.config.update(kwargs)  # overwrite config loaded from disk
        for k, v in self.config.items():
            if isinstance(v, dict) and 'classpath' in v:
                self.config[k] = Configurable.from_config(v)
        self.on_config_ready(**self.config)

    def save_vocabs(self, save_dir, filename='vocabs.json'):
        if hasattr(self, 'vocabs'):
            self.vocabs.save_vocabs(save_dir, filename)

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        if hasattr(self, 'vocabs'):
            self.vocabs = VocabDict()
            self.vocabs.load_vocabs(save_dir, filename)

    def save(self, save_dir: str, **kwargs):
        self.save_config(save_dir)
        self.save_vocabs(save_dir)
        self.save_weights(save_dir)

    def load(self, save_dir: str, devices=None, **kwargs):
        save_dir = get_resource(save_dir)
        # flash('Loading config and vocabs [blink][yellow]...[/yellow][/blink]')
        if devices is None and self.model:
            devices = self.devices
        self.load_config(save_dir, **kwargs)
        self.load_vocabs(save_dir)
        flash('Building model [blink][yellow]...[/yellow][/blink]')
        self.model = self.build_model(
            **merge_dict(self.config, training=False, **kwargs, overwrite=True,
                         inplace=True))
        flash('')
        self.load_weights(save_dir, **kwargs)
        self.to(devices)
        self.model.eval()

    def fit(self, trn_data, dev_data, save_dir, batch_size, epochs, devices=None, logger=None, seed=None,
            finetune=False, eval_trn=True, _device_placeholder=False, **kwargs):
        # Common initialization steps
        config = self._capture_config(locals())
        if not logger:
            logger = self.build_logger('train', save_dir)
        if not seed:
            self.config.seed = 233 if isdebugging() else int(time.time())
        set_seed(self.config.seed)
        logger.info(self._savable_config.to_json(sort=True))
        if isinstance(devices, list) or devices is None or isinstance(devices, float):
            flash('[yellow]Querying CUDA devices [blink]...[/blink][/yellow]')
            devices = -1 if isdebugging() else cuda_devices(devices)
            flash('')
        # flash(f'Available GPUs: {devices}')
        if isinstance(devices, list):
            first_device = (devices[0] if devices else -1)
        elif isinstance(devices, dict):
            first_device = next(iter(devices.values()))
        elif isinstance(devices, int):
            first_device = devices
        else:
            first_device = -1
        if _device_placeholder and first_device >= 0:
            _dummy_placeholder = self._create_dummy_placeholder_on(first_device)
        if finetune:
            if isinstance(finetune, str):
                self.load(finetune, devices=devices)
            else:
                self.load(save_dir, devices=devices)
            logger.info(
                f'Finetune model loaded with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}'
                f'/{sum(p.numel() for p in self.model.parameters())} trainable/total parameters.')
        self.on_config_ready(**self.config)
        trn = self.build_dataloader(**merge_dict(config, data=trn_data, batch_size=batch_size, shuffle=True,
                                                 training=True, device=first_device, logger=logger, vocabs=self.vocabs,
                                                 overwrite=True))
        dev = self.build_dataloader(**merge_dict(config, data=dev_data, batch_size=batch_size, shuffle=False,
                                                 training=None, device=first_device, logger=logger, vocabs=self.vocabs,
                                                 overwrite=True)) if dev_data else None
        if not finetune:
            flash('[yellow]Building model [blink]...[/blink][/yellow]')
            self.model = self.build_model(**merge_dict(config, training=True))
            flash('')
            logger.info(f'Model built with {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}'
                        f'/{sum(p.numel() for p in self.model.parameters())} trainable/total parameters.')
            assert self.model, 'build_model is not properly implemented.'
        _description = repr(self.model)
        if len(_description.split('\n')) < 10:
            logger.info(_description)
        self.save_config(save_dir)
        self.save_vocabs(save_dir)
        self.to(devices, logger)
        if _device_placeholder and first_device >= 0:
            del _dummy_placeholder
        criterion = self.build_criterion(**merge_dict(config, trn=trn))
        optimizer = self.build_optimizer(**merge_dict(config, trn=trn, criterion=criterion))
        metric = self.build_metric(**self.config)
        if hasattr(trn.dataset, '__len__') and dev and hasattr(dev.dataset, '__len__'):
            logger.info(f'{len(trn.dataset)}/{len(dev.dataset)} samples in trn/dev set.')
            trn_size = len(trn) // self.config.get('gradient_accumulation', 1)
            ratio_width = len(f'{trn_size}/{trn_size}')
        else:
            ratio_width = None
        return self.execute_training_loop(**merge_dict(config, trn=trn, dev=dev, epochs=epochs, criterion=criterion,
                                                       optimizer=optimizer, metric=metric, logger=logger,
                                                       save_dir=save_dir,
                                                       devices=devices,
                                                       ratio_width=ratio_width,
                                                       trn_data=trn_data,
                                                       dev_data=dev_data,
                                                       eval_trn=eval_trn,
                                                       overwrite=True))

    def build_logger(self, name, save_dir):
        logger = init_logger(name=name, root_dir=save_dir, level=logging.INFO, fmt="%(message)s")
        return logger

    @abstractmethod
    def build_dataloader(self, data, batch_size, shuffle=False, device=None, logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        pass

    def build_vocabs(self, **kwargs):
        pass

    @property
    def _savable_config(self):
        def convert(k, v):
            if hasattr(v, 'config'):
                v = v.config
            if isinstance(v, (set, tuple)):
                v = list(v)
            return k, v

        config = SerializableDict(
            convert(k, v) for k, v in sorted(self.config.items()))
        config.update({
            # 'create_time': now_datetime(),
            'classpath': classpath_of(self),
            'elit_version': elit.__version__,
        })
        return config

    @abstractmethod
    def build_optimizer(self, **kwargs):
        pass

    @abstractmethod
    def build_criterion(self, decoder, **kwargs):
        pass

    @abstractmethod
    def build_metric(self, **kwargs):
        pass

    @abstractmethod
    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None,
                              **kwargs):
        pass

    @abstractmethod
    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        pass

    @abstractmethod
    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, **kwargs):
        pass

    @abstractmethod
    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def evaluate(self, tst_data, save_dir=None, logger: logging.Logger = None, batch_size=None, output=False, **kwargs):
        if not self.model:
            raise RuntimeError('Call fit or load before evaluate.')
        if isinstance(tst_data, str):
            tst_data = get_resource(tst_data)
            filename = os.path.basename(tst_data)
        else:
            filename = None
        if output is True:
            output = self.generate_prediction_filename(tst_data if isinstance(tst_data, str) else 'test.txt', save_dir)
        if logger is None:
            _logger_name = basename_no_ext(filename) if filename else None
            logger = self.build_logger(_logger_name, save_dir)
        if not batch_size:
            batch_size = self.config.get('batch_size', 32)
        data = self.build_dataloader(**merge_dict(self.config, data=tst_data, batch_size=batch_size, shuffle=False,
                                                  device=self.devices[0], logger=logger, overwrite=True))
        dataset = data
        while dataset and hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        num_samples = len(dataset) if dataset else None
        if output and isinstance(dataset, TransformDataset):
            def add_idx(samples):
                for idx, sample in enumerate(samples):
                    if sample:
                        sample[IDX] = idx

            add_idx(dataset.data)
            if dataset.cache:
                add_idx(dataset.cache)

        criterion = self.build_criterion(**self.config)
        metric = self.build_metric(**self.config)
        start = time.time()
        outputs = self.evaluate_dataloader(data, criterion=criterion, filename=filename, output=output, input=tst_data,
                                           save_dir=save_dir,
                                           test=True,
                                           num_samples=num_samples,
                                           **merge_dict(self.config, batch_size=batch_size, metric=metric,
                                                        logger=logger, **kwargs))
        elapsed = time.time() - start
        if logger:
            if num_samples:
                logger.info(f'speed: {num_samples / elapsed:.0f} samples/second')
            else:
                logger.info(f'speed: {len(data) / elapsed:.0f} batches/second')
        return metric, outputs

    def generate_prediction_filename(self, tst_data, save_dir):
        assert isinstance(tst_data,
                          str), 'tst_data has be a str in order to infer the output name'
        output = os.path.splitext(os.path.basename(tst_data))
        output = os.path.join(save_dir, output[0] + '.pred' + output[1])
        return output

    def to(self, devices=Union[int, float, List[int], Dict[str, Union[int, torch.device]]],
           logger: logging.Logger = None):
        if devices == -1 or devices == [-1]:
            devices = []
        elif isinstance(devices, (int, float)) or devices is None:
            devices = cuda_devices(devices)
        if devices:
            if logger:
                logger.info(f'Using GPUs: [on_blue][cyan][bold]{devices}[/bold][/cyan][/on_blue]')
            if isinstance(devices, list):
                flash(f'Moving model to GPUs {devices} [blink][yellow]...[/yellow][/blink]')
                self.model = self.model.to(devices[0])
                if len(devices) > 1 and not isdebugging() and not isinstance(self.model, nn.DataParallel):
                    self.model = self.parallelize(devices)
            elif isinstance(devices, dict):
                for name, module in self.model.named_modules():
                    for regex, device in devices.items():
                        try:
                            on_device: torch.device = next(module.parameters()).device
                        except StopIteration:
                            continue
                        if on_device == device:
                            continue
                        if isinstance(device, int):
                            if on_device.index == device:
                                continue
                        if re.match(regex, name):
                            if not name:
                                name = '*'
                            flash(f'Moving module [yellow]{name}[/yellow] to [on_yellow][magenta][bold]{device}'
                                  f'[/bold][/magenta][/on_yellow]: [red]{regex}[/red]\n')
                            module.to(device)
            else:
                raise ValueError(f'Unrecognized devices {devices}')
            flash('')
        else:
            if logger:
                logger.info('Using CPU')

    def parallelize(self, devices: List[Union[int, torch.device]]):
        return nn.DataParallel(self.model, device_ids=devices)

    @property
    def devices(self):
        if self.model is None:
            return None
        # next(parser.model.parameters()).device
        if hasattr(self.model, 'device_ids'):
            return self.model.device_ids
        device: torch.device = next(self.model.parameters()).device
        return [device]

    @property
    def device(self):
        devices = self.devices
        if not devices:
            return None
        return devices[0]

    def on_config_ready(self, **kwargs):
        pass

    @property
    def model_(self) -> nn.Module:
        """
        The actual model when it's wrapped by a `DataParallel`

        Returns: The "real" model

        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    # noinspection PyMethodOverriding
    @abstractmethod
    def predict(self, data: Union[str, List[str]], batch_size: int = None, **kwargs):
        pass

    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = merge_list_of_dict(samples)
        return batch

    @staticmethod
    def _create_dummy_placeholder_on(device):
        if device < 0:
            device = 'cpu:0'
        return torch.zeros(16, 16, device=device)

    @torch.no_grad()
    def __call__(self, data, batch_size=None, **kwargs):
        return super().__call__(data, **merge_dict(self.config, overwrite=True,
                                                   batch_size=batch_size or self.config.get('batch_size', None),
                                                   **kwargs))

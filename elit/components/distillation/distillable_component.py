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
from abc import ABC
from copy import copy

import elit
from elit.common.torch_component import TorchComponent
from elit.components.distillation.losses import KnowledgeDistillationLoss
from elit.components.distillation.schedulers import TemperatureScheduler
from elit.utils.torch_util import cuda_devices
from elit.utils.util import merge_locals_kwargs


class DistillableComponent(TorchComponent, ABC):

    # noinspection PyMethodMayBeStatic,PyTypeChecker
    def build_teacher(self, teacher: str, devices) -> TorchComponent:
        return elit.load(teacher, load_kwargs={'devices': devices})

    def distill(self,
                teacher: str,
                trn_data,
                dev_data,
                save_dir,
                batch_size=None,
                epochs=None,
                kd_criterion='kd_ce_loss',
                temperature_scheduler='flsw',
                devices=None,
                logger=None,
                seed=None,
                **kwargs):
        devices = devices or cuda_devices()
        if isinstance(kd_criterion, str):
            kd_criterion = KnowledgeDistillationLoss(kd_criterion)
        if isinstance(temperature_scheduler, str):
            temperature_scheduler = TemperatureScheduler.from_name(temperature_scheduler)
        teacher = self.build_teacher(teacher, devices=devices)
        self.vocabs = teacher.vocabs
        config = copy(teacher.config)
        batch_size = batch_size or config.get('batch_size', None)
        epochs = epochs or config.get('epochs', None)
        config.update(kwargs)
        return super().fit(**merge_locals_kwargs(locals(),
                                                 config,
                                                 excludes=('self', 'kwargs', '__class__', 'config')))

    @property
    def _savable_config(self):
        config = super(DistillableComponent, self)._savable_config
        if 'teacher' in config:
            config.teacher = config.teacher.load_path
        return config

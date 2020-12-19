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
from typing import Callable, Any

from elit.common.component import Component
from elit.utils.reflection import classpath_of, object_from_classpath, str_to_type


class LambdaComponent(Component):
    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function
        self.config['function'] = classpath_of(function)

    def predict(self, data: Any, **kwargs):
        unpack = kwargs.pop('_elit_unpack', None)
        if unpack:
            return self.function(*data, **kwargs)
        return self.function(data, **kwargs)

    @staticmethod
    def from_config(meta: dict, **kwargs):
        cls = str_to_type(meta['classpath'])
        function = meta['function']
        function = object_from_classpath(function)
        return cls(function)

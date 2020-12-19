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
import inspect
from abc import ABC, abstractmethod
from typing import Any

from elit.common.structure import Configurable


class Component(Configurable, ABC):
    @abstractmethod
    def predict(self, data: Any, **kwargs):
        """Predict on data

        Args:
          data: Any type of data subject to sub-classes
          kwargs: Additional arguments
          data: Any: 
          **kwargs: 

        Returns:

        """
        raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def __call__(self, data, **kwargs):
        return self.predict(data, **kwargs)

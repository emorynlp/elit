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
import json
from collections import OrderedDict
from typing import Dict

from elit.utils.io_util import save_json, save_pickle, load_pickle, load_json, filename_is_json
from elit.utils.reflection import str_to_type, classpath_of


class Serializable(object):
    """A super class for save/load operations."""

    def save(self, path, fmt=None):
        if not fmt:
            if filename_is_json(path):
                self.save_json(path)
            else:
                self.save_pickle(path)
        elif fmt in ['json', 'jsonl']:
            self.save_json(path)
        else:
            self.save_pickle(path)

    def load(self, path, fmt=None):
        if not fmt:
            if filename_is_json(path):
                self.load_json(path)
            else:
                self.load_pickle(path)
        elif fmt in ['json', 'jsonl']:
            self.load_json(path)
        else:
            self.load_pickle(path)

    def save_pickle(self, path):
        """Save to path

        Args:
          path: 

        Returns:

        
        """
        save_pickle(self, path)

    def load_pickle(self, path):
        """Load from path

        Args:
          path(str): file path

        Returns:

        
        """
        item = load_pickle(path)
        return self.copy_from(item)

    def save_json(self, path):
        save_json(self.to_dict(), path)

    def load_json(self, path):
        item = load_json(path)
        return self.copy_from(item)

    # @abstractmethod
    def copy_from(self, item):
        self.__dict__ = item.__dict__
        # raise NotImplementedError('%s.%s()' % (self.__class__.__name__, inspect.stack()[0][3]))

    def to_json(self, ensure_ascii=False, indent=2, sort=False) -> str:
        d = self.to_dict()
        if sort:
            d = OrderedDict(sorted(d.items()))
        return json.dumps(d, ensure_ascii=ensure_ascii, indent=indent, default=lambda o: repr(o))

    def to_dict(self) -> dict:
        return self.__dict__


class SerializableDict(Serializable, dict):

    def save_json(self, path):
        save_json(self, path)

    def copy_from(self, item):
        if isinstance(item, dict):
            self.clear()
            self.update(item)

    def __getattr__(self, key):
        if key.startswith('__'):
            return dict.__getattr__(key)
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def to_dict(self) -> dict:
        return self


class Configurable(object):
    @staticmethod
    def from_config(config: dict, **kwargs):
        """

        Args:
          config: 
          kwargs: 
          config: dict: 
          **kwargs: 

        Returns:

        
        """
        cls = config.get('classpath', None)
        assert cls, f'{config} doesn\'t contain classpath field'
        cls = str_to_type(cls)
        deserialized_config = dict(config)
        for k, v in config.items():
            if isinstance(v, dict) and 'classpath' in v:
                deserialized_config[k] = Configurable.from_config(v)
        if cls.from_config == Configurable.from_config:
            deserialized_config.pop('classpath')
            return cls(**deserialized_config)
        else:
            return cls.from_config(deserialized_config)


class AutoConfigurable(Configurable):
    @property
    def config(self):
        return dict([('classpath', classpath_of(self))] +
                    [(k, v.config if hasattr(v, 'config') else v)
                     for k, v in self.__dict__.items() if
                     not k.startswith('_')])

    def __repr__(self) -> str:
        return repr(self.config)


class ConfigTracker(Configurable):

    def __init__(self, locals_: Dict, exclude=('kwargs', 'self', '__class__', 'locals_')) -> None:
        if 'kwargs' in locals_:
            locals_.update(locals_['kwargs'])
        self.config = SerializableDict(
            (k, v.config if hasattr(v, 'config') else v) for k, v in locals_.items() if k not in exclude)
        self.config['classpath'] = classpath_of(self)


class History(object):
    def __init__(self):
        self.num_mini_batches = 0

    def step(self, gradient_accumulation):
        self.num_mini_batches += 1
        return self.num_mini_batches % gradient_accumulation == 0

    def num_training_steps(self, num_batches, gradient_accumulation):
        return len(
            [i for i in range(self.num_mini_batches + 1, self.num_mini_batches + num_batches + 1) if
             i % gradient_accumulation == 0])

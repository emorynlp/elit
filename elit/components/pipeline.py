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
import types
from typing import Callable, List, Generator, Union, Any, Tuple, Iterable
from elit.components.lambda_wrapper import LambdaComponent
from elit.common.component import Component
from elit.common.document import Document
from elit.utils.component_util import load_from_meta
from elit.utils.io_util import save_json, load_json
from elit.utils.reflection import module_path_of, str_to_type, classpath_of
import elit


class Pipe(Component):

    def __init__(self, component: Component, input_key: str = None, output_key: str = None, **kwargs) -> None:
        super().__init__()
        self.output_key = output_key
        self.input_key = input_key
        self.component = component
        self.kwargs = kwargs
        self.config.update({
            'component': component.config,
            'input_key': self.input_key,
            'output_key': self.output_key,
            'kwargs': self.kwargs
        })

    # noinspection PyShadowingBuiltins
    def predict(self, doc: Document, **kwargs) -> Document:

        unpack = False
        if self.input_key:
            if isinstance(self.input_key, (tuple, list)):
                if isinstance(self.component, LambdaComponent):  # assume functions take multiple arguments
                    input = [doc[key] for key in self.input_key]
                    unpack = True
                else:
                    input = list(list(zip(*sent)) for sent in zip(*[doc[key] for key in self.input_key]))
            else:
                input = doc[self.input_key]
        else:
            input = doc

        if self.kwargs:
            kwargs.update(self.kwargs)
        if unpack:
            kwargs['_elit_unpack'] = True
        output = self.component(input, **kwargs)
        if isinstance(output, types.GeneratorType):
            output = list(output)
        if self.output_key:
            if not isinstance(doc, Document):
                doc = Document()
            if isinstance(self.output_key, tuple):
                for key, value in zip(self.output_key, output):
                    doc[key] = value
            else:
                doc[self.output_key] = output
            return doc
        return output

    def __repr__(self):
        return f'{self.input_key}->{self.component.__class__.__name__}->{self.output_key}'

    @staticmethod
    def from_config(meta: dict, **kwargs):
        cls = str_to_type(meta['classpath'])
        component = load_from_meta(meta['component'])
        return cls(component, meta['input_key'], meta['output_key'], **meta['kwargs'])


class Pipeline(Component, list):
    def __init__(self, *pipes: Pipe) -> None:
        super().__init__()
        if pipes:
            self.extend(pipes)

    def append(self, component: Callable, input_key: Union[str, Iterable[str]] = None,
               output_key: Union[str, Iterable[str]] = None, **kwargs):
        self.insert(len(self), component, input_key, output_key, **kwargs)
        return self

    def insert(self, index: int, component: Callable, input_key: Union[str, Iterable[str]] = None,
               output_key: Union[str, Iterable[str]] = None,
               **kwargs):
        if not input_key and len(self):
            input_key = self[-1].output_key
        if not isinstance(component, Component):
            component = LambdaComponent(component)
        super().insert(index, Pipe(component, input_key, output_key, **kwargs))
        return self

    def __call__(self, doc: Document, **kwargs) -> Document:
        for component in self:
            doc = component(doc)
        return doc

    @property
    def meta(self):
        return {
            'classpath': classpath_of(self),
            'elit_version': elit.version.__version__,
            'pipes': [pipe.config for pipe in self]
        }

    @meta.setter
    def meta(self, value):
        pass

    def save(self, filepath):
        save_json(self.meta, filepath)

    def load(self, filepath):
        meta = load_json(filepath)
        self.clear()
        self.extend(Pipeline.from_config(meta))

    @staticmethod
    def from_config(meta: Union[dict, str], **kwargs):
        if isinstance(meta, str):
            meta = load_json(meta)
        return Pipeline(*[load_from_meta(pipe) for pipe in meta['pipes']])

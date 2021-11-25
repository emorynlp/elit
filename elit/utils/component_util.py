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
import os
import traceback
from sys import exit

from elit import pretrained
from elit.common.component import Component
from elit.utils.io_util import get_resource, load_json, eprint, get_latest_info_from_pypi
from elit.utils.reflection import object_from_classpath, str_to_type
from elit import version


def load_from_meta_file(save_dir: str, meta_filename='meta.json', transform_only=False, load_kwargs=None,
                        **kwargs) -> Component:
    """

    Args:
        save_dir:
        meta_filename (str): The meta file of that saved component, which stores the classpath and version.
        transform_only:
        **kwargs:

    Returns:

    """
    identifier = save_dir
    load_path = save_dir
    save_dir = get_resource(save_dir)
    if save_dir.endswith('.json'):
        meta_filename = os.path.basename(save_dir)
        save_dir = os.path.dirname(save_dir)
    metapath = os.path.join(save_dir, meta_filename)
    if not os.path.isfile(metapath):
        metapath = os.path.join(save_dir, 'config.json')
    if not os.path.isfile(metapath):
        tips = ''
        if save_dir.isupper():
            from difflib import SequenceMatcher
            similar_keys = sorted(pretrained.ALL.keys(),
                                  key=lambda k: SequenceMatcher(None, save_dir, metapath).ratio(),
                                  reverse=True)[:5]
            tips = f'Check its spelling based on the available keys:\n' + \
                   f'{sorted(pretrained.ALL.keys())}\n' + \
                   f'Tips: it might be one of {similar_keys}'
        raise FileNotFoundError(f'The identifier {save_dir} resolves to a non-exist meta file {metapath}. {tips}')
    meta: dict = load_json(metapath)
    cls = meta.get('classpath', None)
    if not cls:
        cls = meta.get('class_path', None)  # For older version
    assert cls, f'{meta_filename} doesn\'t contain classpath field'
    try:
        obj: Component = object_from_classpath(cls)
        if hasattr(obj, 'load'):
            if transform_only:
                # noinspection PyUnresolvedReferences
                obj.load_transform(save_dir)
            else:
                if load_kwargs is None:
                    load_kwargs = {}
                if os.path.isfile(os.path.join(save_dir, 'config.json')):
                    obj.load(save_dir, **kwargs)
                else:
                    obj.load(metapath, **kwargs)
            obj.config['load_path'] = load_path
        return obj
    except Exception as e:
        eprint(f'Failed to load {identifier}. See traceback below:')
        eprint(f'{"ERROR LOG BEGINS":=^80}')
        traceback.print_exc()
        eprint(f'{"ERROR LOG ENDS":=^80}')
        from pkg_resources import parse_version
        model_version = meta.get("elit_version", "unknown")
        if model_version == '2.0.0':  # Quick fix: the first version used a wrong string
            model_version = '2.0.0-alpha.0'
        model_version = parse_version(model_version)
        installed_version = parse_version(version.__version__)
        try:
            latest_version = get_latest_info_from_pypi()
        except:
            latest_version = None
        if model_version > installed_version:
            eprint(f'{identifier} was created with elit-{model_version}, '
                   f'while you are running a lower version: {installed_version}. ')
        if installed_version < latest_version:
            eprint(
                f'Please upgrade elit with:\n'
                f'pip install --upgrade elit\n')
        eprint(
            'If the problem still persists, please submit an issue to https://github.com/emorynlp/elit/issues\n'
            'When reporting an issue, make sure to paste the FULL ERROR LOG above.')
        exit(1)


def load_from_meta(meta: dict) -> Component:
    cls = meta.get('class_path', None)
    assert cls, f'{meta} doesn\'t contain class_path field'
    cls = str_to_type(cls)
    return cls.from_config(meta)

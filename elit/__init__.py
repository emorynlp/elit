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
__author__ = "Jinho D. Choi"
import elit.common
import elit.components
import elit.pretrained
import elit.utils
from elit.version import __version__

elit.utils.util.ls_resource_in_module(elit.pretrained)


def load(save_dir: str, transform_only: bool = False, **kwargs) -> elit.common.component.Component:
    """Load pretrained component from an identifier.

    Args:
      save_dir (str): The identifier to the saved component. It could be a remote URL or a local path.
      transform_only: Whether to load transform only for TensorFlow components. Default: ``False``.
      **kwargs: Arguments passed to `Component.load`

    Returns:
      A pretrained component.

    """
    save_dir = elit.pretrained.ALL.get(save_dir, save_dir)
    from elit.utils.component_util import load_from_meta_file
    return load_from_meta_file(save_dir, 'meta.json', transform_only=transform_only, **kwargs)


def pipeline(*pipes) -> elit.components.pipeline.Pipeline:
    """Creates a pipeline of components.

    Args:
      *pipes: Components if pre-defined any.

    Returns:
      A pipeline

    """
    return elit.components.pipeline.Pipeline(*pipes)

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

from elit.common.constant import ELIT_URL
# mute transformers
import logging

logging.getLogger('transformers.file_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.filelock').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_tf_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

zh_albert_models_google = {
    'albert_base_zh': ELIT_URL + 'embeddings/albert_base_zh.tar.gz',  # Provide mirroring
    'albert_large_zh': 'https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz',
    'albert_xlarge_zh': 'https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz',
    'albert_xxlarge_zh': 'https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz',
}
# bert_models_google['chinese_L-12_H-768_A-12'] = elit_URL + 'embeddings/chinese_L-12_H-768_A-12.zip'

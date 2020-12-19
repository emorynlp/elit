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

_PTB_HOME = 'https://github.com/KhalilMrini/LAL-Parser/archive/master.zip#data/'

PTB_TRAIN = _PTB_HOME + '02-21.10way.clean'
PTB_DEV = _PTB_HOME + '22.auto.clean'
PTB_TEST = _PTB_HOME + '23.auto.clean'

PTB_SD330_TRAIN = _PTB_HOME + 'ptb_train_3.3.0.sd.clean'
PTB_SD330_DEV = _PTB_HOME + 'ptb_dev_3.3.0.sd.clean'
PTB_SD330_TEST = _PTB_HOME + 'ptb_test_3.3.0.sd.clean'

PTB_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}

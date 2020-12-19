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

if os.environ.get('USE_TF', None) is None:
    os.environ["USE_TF"] = 'NO'  # saves time loading transformers
if os.environ.get('TOKENIZERS_PARALLELISM', None) is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BertTokenizer, BertConfig, PretrainedConfig, \
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, BertTokenizerFast, AlbertConfig, BertModel, AutoModel, \
    PreTrainedModel, get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, optimization, BartModel


class AutoModel_(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, training=True, **kwargs):
        if training:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            if isinstance(pretrained_model_name_or_path, str):
                return super().from_config(AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs))
            else:
                assert not kwargs
                return super().from_config(pretrained_model_name_or_path)

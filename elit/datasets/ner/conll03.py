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
from typing import Union, List, Callable

from elit.common.dataset import TransformDataset
from elit.utils.io_util import get_resource, generate_words_tags_from_tsv
from elit.utils.string_util import split_long_sentence_into


class TSVTaggingDataset(TransformDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 generate_idx=None,
                 delimiter=None, max_seq_len=None, sent_delimiter=None, char_level=False, hard_constraint=False,
                 ) -> None:
        self.char_level = char_level
        self.hard_constraint = hard_constraint
        self.sent_delimiter = sent_delimiter
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath):
        filepath = get_resource(filepath)
        # idx = 0
        for words, tags in generate_words_tags_from_tsv(filepath, lower=False):
            # idx += 1
            # if idx % 1000 == 0:
            #     print(f'\rRead instances {idx // 1000}k', end='')
            if self.max_seq_len:
                start = 0
                for short_sents in split_long_sentence_into(words, self.max_seq_len, self.sent_delimiter,
                                                            char_level=self.char_level,
                                                            hard_constraint=self.hard_constraint):
                    end = start + len(short_sents)
                    yield {'token': short_sents, 'tag': tags[start:end]}
                    start = end
            else:
                yield {'token': words, 'tag': tags}
        # print('\r', end='')


def main():
    dataset = TSVTaggingDataset(CONLL03_EN_DEV)
    print(dataset[3])


if __name__ == '__main__':
    main()

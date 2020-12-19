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
from typing import Union, List, Callable

from elit.common.constant import NULL
from elit.common.dataset import TransformDataset
import json
from alnlp.metrics import span_utils
from elit.utils.io_util import TimingFileIterator, read_tsv_as_sents


class JsonNERDataset(TransformDataset):

    def __init__(self, data: Union[str, List], transform: Union[Callable, List] = None, cache=None,
                 generate_idx=None, doc_level_offset=True, tagset=None) -> None:
        self.tagset = tagset
        self.doc_level_offset = doc_level_offset
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        filename = os.path.basename(filepath)
        reader = TimingFileIterator(filepath)
        num_docs, num_sentences = 0, 0
        for line in reader:
            doc = json.loads(line)
            num_docs += 1
            num_tokens_in_doc = 0
            for sentence, ner in zip(doc['sentences'], doc['ner']):
                if self.doc_level_offset:
                    ner = [(x[0] - num_tokens_in_doc, x[1] - num_tokens_in_doc, x[2]) for x in ner]
                else:
                    ner = [(x[0], x[1], x[2]) for x in ner]
                if self.tagset:
                    ner = [x for x in ner if x[2] in self.tagset]
                    if isinstance(self.tagset, dict):
                        ner = [(x[0], x[1], self.tagset[x[2]]) for x in ner]
                deduplicated_srl = []
                be_set = set()
                for b, e, l in ner:
                    be = (b, e)
                    if be in be_set:
                        continue
                    be_set.add(be)
                    deduplicated_srl.append((b, e, l))
                yield {
                    'token': sentence,
                    'ner': deduplicated_srl
                }
                num_sentences += 1
                num_tokens_in_doc += len(sentence)
            reader.log(
                f'{filename} {num_docs} documents, {num_sentences} sentences [blink][yellow]...[/yellow][/blink]')
        reader.erase()


def convert_conll03_to_json(file_path):
    dataset = []
    num_docs = [0]

    def new_doc():
        doc_key = num_docs[0]
        num_docs[0] += 1
        return {
            'doc_key': doc_key,
            'sentences': [],
            'ner': [],
        }

    doc = new_doc()
    offset = 0
    for cells in read_tsv_as_sents(file_path):
        if cells[0][0] == '-DOCSTART-' and doc['ner']:
            dataset.append(doc)
            doc = new_doc()
            offset = 0
        sentence = [x[0] for x in cells]
        ner = [x[-1] for x in cells]
        ner = span_utils.iobes_tags_to_spans(ner)
        adjusted_ner = []
        for label, (span_start, span_end) in ner:
            adjusted_ner.append([span_start + offset, span_end + offset, label])
        doc['sentences'].append(sentence)
        doc['ner'].append(adjusted_ner)
        offset += len(sentence)
    if doc['ner']:
        dataset.append(doc)
    output_path = os.path.splitext(file_path)[0] + '.json'
    with open(output_path, 'w') as out:
        for each in dataset:
            json.dump(each, out)
            out.write('\n')


def unpack_ner(sample: dict) -> dict:
    ner: list = sample.get('ner', None)
    if ner is not None:
        if ner:
            sample['begin_offset'], sample['end_offset'], sample['label'] = zip(*ner)
        else:
            # It's necessary to create a null label when there is no NER in the sentence for the sake of padding.
            sample['begin_offset'], sample['end_offset'], sample['label'] = [0], [0], [NULL]
    return sample

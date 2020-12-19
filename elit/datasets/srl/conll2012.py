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
import glob
import json
import os
from typing import Union, List, Callable

from alnlp.metrics.span_utils import enumerate_spans

from elit.common.dataset import TransformDataset
from elit.common.transform import NamedTransform
from elit.utils.io_util import read_tsv_as_sents, get_resource, merge_files, TimingFileIterator
from elit.utils.time_util import CountdownTimer

_CONLL2012_EN_HOME = 'https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/archive/master.zip#conll-formatted-ontonotes-5.0/data'
# These are v4 of OntoNotes
CONLL2012_EN_TRAIN = _CONLL2012_EN_HOME + '/train/data/english/annotations'
CONLL2012_EN_DEV = _CONLL2012_EN_HOME + '/development/data/english/annotations'
CONLL2012_EN_TEST = _CONLL2012_EN_HOME + '/conll-2012-test/data/english/annotations'


def conll_2012_en_combined():
    home = get_resource(_CONLL2012_EN_HOME)
    outputs = ['train', 'dev', 'test']
    for i in range(len(outputs)):
        outputs[i] = f'{home}/conll12_en/{outputs[i]}.conll'
    if all(os.path.isfile(x) for x in outputs):
        return outputs
    os.makedirs(os.path.dirname(outputs[0]), exist_ok=True)
    for in_path, out_path in zip([CONLL2012_EN_TRAIN, CONLL2012_EN_DEV, CONLL2012_EN_TEST], outputs):
        in_path = get_resource(in_path)
        files = sorted(glob.glob(f'{in_path}/**/*gold_conll', recursive=True))
        merge_files(files, out_path)
    return outputs


# conll_2012_en_combined()


class CoNLL2012BIOSRLDataset(TransformDataset):
    def load_file(self, filepath: str):
        filepath = get_resource(filepath)
        if os.path.isfile(filepath):
            files = [filepath]
        else:
            assert os.path.isdir(filepath), f'{filepath} has to be a directory of CoNLL 2012'
            files = sorted(glob.glob(f'{filepath}/**/*gold_conll', recursive=True))
        timer = CountdownTimer(len(files))
        for fid, f in enumerate(files):
            timer.log(f'files loading[blink][yellow]...[/yellow][/blink]')
            # 0:DOCUMENT 1:PART 2:INDEX 3:WORD 4:POS 5:PARSE 6:LEMMA 7:FRAME 8:SENSE 9:SPEAKER 10:NE 11-N:ARGS N:COREF
            for sent in read_tsv_as_sents(f, ignore_prefix='#'):
                sense = [cell[7] for cell in sent]
                props = [cell[11:-1] for cell in sent]
                props = map(lambda p: p, zip(*props))
                prd_bio_labels = [self._make_bio_labels(prop) for prop in props]
                prd_bio_labels = [self._remove_B_V(x) for x in prd_bio_labels]
                prd_indices = [i for i, x in enumerate(sense) if x != '-']
                token = [x[3] for x in sent]
                srl = [None for x in token]
                for idx, labels in zip(prd_indices, prd_bio_labels):
                    srl[idx] = labels
                srl = [x if x else ['O'] * len(token) for x in srl]
                yield {'token': token, 'srl': srl}

    @staticmethod
    def _make_bio_labels(prop):
        """Copied from https://github.com/hiroki13/span-based-srl/blob/2c8b677c4e00b6c607e09ef4f9fe3d54961e4f2e/src/utils/sent.py#L42

        Args:
          prop: 1D: n_words; elem=bracket label

        Returns:
          1D: n_words; elem=BIO label

        """
        labels = []
        prev = None
        for arg in prop:
            if arg.startswith('('):
                if arg.endswith(')'):
                    prev = arg.split("*")[0][1:]
                    label = 'B-' + prev
                    prev = None
                else:
                    prev = arg[1:-1]
                    label = 'B-' + prev
            else:
                if prev:
                    label = 'I-' + prev
                    if arg.endswith(')'):
                        prev = None
                else:
                    label = 'O'
            labels.append(label)
        return labels

    @staticmethod
    def _remove_B_V(labels):
        return ['O' if x == 'B-V' else x for x in labels]


class CoNLL2012SRLDataset(TransformDataset):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 doc_level_offset=True,
                 generate_idx=None) -> None:
        self.doc_level_offset = doc_level_offset
        super().__init__(data, transform, cache, generate_idx=generate_idx)

    def load_file(self, filepath: str):
        filename = os.path.basename(filepath)
        reader = TimingFileIterator(filepath)
        num_docs, num_sentences = 0, 0
        for line in reader:
            doc = json.loads(line)
            num_docs += 1
            num_tokens_in_doc = 0
            for sid, (sentence, srl) in enumerate(zip(doc['sentences'], doc['srl'])):
                if self.doc_level_offset:
                    srl = [(x[0] - num_tokens_in_doc, x[1] - num_tokens_in_doc, x[2] - num_tokens_in_doc, x[3]) for x in
                           srl]
                else:
                    srl = [(x[0], x[1], x[2], x[3]) for x in srl]
                for x in srl:
                    if any([o < 0 for o in x[:3]]):
                        raise ValueError(f'Negative offset occurred, maybe doc_level_offset=False')
                    if any([o >= len(sentence) for o in x[:3]]):
                        raise ValueError('Offset exceeds sentence length, maybe doc_level_offset=True')
                deduplicated_srl = set()
                pa_set = set()
                for p, b, e, l in srl:
                    pa = (p, b, e)
                    if pa in pa_set:
                        continue
                    pa_set.add(pa)
                    deduplicated_srl.add((p, b, e, l))
                yield self.build_sample(sentence, deduplicated_srl, doc, sid)
                num_sentences += 1
                num_tokens_in_doc += len(sentence)
            reader.log(
                f'{filename} {num_docs} documents, {num_sentences} sentences [blink][yellow]...[/yellow][/blink]')
        reader.erase()

    # noinspection PyMethodMayBeStatic
    def build_sample(self, sentence, deduplicated_srl, doc, sid):
        return {
            'token': sentence,
            'srl': deduplicated_srl
        }


def group_pa_by_p(sample: dict) -> dict:
    if 'srl' in sample:
        srl: list = sample['srl']
        grouped_srl = group_pa_by_p_(srl)
        sample['srl'] = grouped_srl
    return sample


def group_pa_by_p_(srl):
    grouped_srl = {}
    for p, b, e, l in srl:
        bel = grouped_srl.get(p, None)
        if not bel:
            bel = grouped_srl[p] = set()
        bel.add((b, e, l))
    return grouped_srl


def filter_v_args(sample: dict) -> dict:
    if 'srl' in sample:
        sample['srl'] = [t for t in sample['srl'] if t[-1] not in ["V", "C-V"]]
    return sample


def unpack_srl(sample: dict) -> dict:
    if 'srl' in sample:
        srl = sample['srl']
        predicate_offset = [x[0] for x in srl]
        argument_begin_offset = [x[1] for x in srl]
        argument_end_offset = [x[2] for x in srl]
        srl_label = [x[-1] for x in srl]
        sample.update({
            'predicate_offset': predicate_offset,
            'argument_begin_offset': argument_begin_offset,
            'argument_end_offset': argument_end_offset,
            'srl_label': srl_label,  # We can obtain mask by srl_label > 0
            # 'srl_mask': len(srl_label),
        })
    return sample


class SpanCandidatesGenerator(NamedTransform):

    def __init__(self, src: str, dst: str = None, max_span_width=None) -> None:
        if not dst:
            dst = f'{src}_span'
        super().__init__(src, dst)
        self.max_span_width = max_span_width

    def __call__(self, sample: dict) -> dict:
        sample[self.dst] = list(enumerate_spans(sample[self.src], max_span_width=self.max_span_width))
        return sample


class CoNLL2012SRLBIODataset(CoNLL2012SRLDataset):
    def build_sample(self, tokens, deduplicated_srl, doc, sid):
        # Convert srl to exclusive format
        deduplicated_srl = set((x[0], x[1], x[2] + 1, x[3]) for x in deduplicated_srl if x[3] != 'V')
        labels = [['O'] * len(tokens) for _ in range(len(tokens))]
        srl = group_pa_by_p_(deduplicated_srl)
        for p, args in sorted(srl.items()):
            labels_per_p = labels[p]
            for start, end, label in args:
                assert end > start
                assert label != 'V'  # We don't predict predicate
                labels_per_p[start] = 'B-' + label
                for j in range(start + 1, end):
                    labels_per_p[j] = 'I-' + label
        sample = {
            'token': tokens,
            'srl': labels,
            'srl_set': deduplicated_srl,
        }
        if 'pos' in doc:
            sample['pos'] = doc['pos'][sid]
        return sample

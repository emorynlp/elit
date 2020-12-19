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
from collections import Counter
from typing import List, Dict, Union, Iterable

from elit.common.constant import UNK, PAD
from elit.common.structure import Serializable
from elit.utils.reflection import classpath_of


class Vocab(Serializable):
    def __init__(self, idx_to_token: List[str] = None, token_to_idx: Dict = None, mutable=True, pad_token=PAD,
                 unk_token=UNK) -> None:
        super().__init__()
        if idx_to_token:
            t2i = dict((token, idx) for idx, token in enumerate(idx_to_token))
            if token_to_idx:
                t2i.update(token_to_idx)
            token_to_idx = t2i
        if token_to_idx is None:
            token_to_idx = {}
            if pad_token is not None:
                token_to_idx[pad_token] = len(token_to_idx)
            if unk_token is not None:
                token_to_idx[unk_token] = token_to_idx.get(unk_token, len(token_to_idx))
        self.token_to_idx = token_to_idx
        self.idx_to_token: List[str] = None
        self.mutable = mutable
        self.pad_token = pad_token
        self.unk_token = unk_token

    def __setitem__(self, token: str, idx: int):
        assert self.mutable, 'Update an immutable Vocab object is not allowed'
        self.token_to_idx[token] = idx

    def __getitem__(self, key: Union[str, int, List]) -> Union[int, str, List]:
        if isinstance(key, str):
            return self.get_idx(key)
        elif isinstance(key, int):
            return self.get_token(key)
        elif isinstance(key, list):
            if len(key) == 0:
                return []
            elif isinstance(key[0], str):
                return [self.get_idx(x) for x in key]
            elif isinstance(key[0], int):
                return [self.get_token(x) for x in key]

    def __contains__(self, key: Union[str, int]):
        if isinstance(key, str):
            return key in self.token_to_idx
        elif isinstance(key, int):
            return 0 <= key < len(self.idx_to_token)
        else:
            return False

    def add(self, token: str) -> int:
        assert self.mutable, 'It is not allowed to call add on an immutable Vocab'
        assert isinstance(token, str), f'Token type must be str but got {type(token)} from {token}'
        assert token is not None, 'Token must not be None'
        idx = self.token_to_idx.get(token, None)
        if idx is None:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
        return idx

    def update(self, tokens: Iterable[str]) -> None:
        """Update the vocab with these tokens by adding them to vocab one by one.

        Args:
          tokens: Iterable[str]: 

        Returns:

        
        """
        assert self.mutable, 'It is not allowed to update an immutable Vocab'
        for token in tokens:
            self.add(token)

    def get_idx(self, token: str) -> int:
        assert isinstance(token, str), 'token has to be `str`'
        idx = self.token_to_idx.get(token, None)
        if idx is None:
            if self.mutable:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
            else:
                idx = self.token_to_idx.get(self.unk_token, None)
        return idx

    def get_idx_without_add(self, token: str) -> int:
        idx = self.token_to_idx.get(token, None)
        if idx is None:
            idx = self.token_to_idx.get(self.safe_unk_token, None)
        return idx

    def get_token(self, idx: int) -> str:
        if self.idx_to_token:
            return self.idx_to_token[idx]

        if self.mutable:
            for token in self.token_to_idx:
                if self.token_to_idx[token] == idx:
                    return token

    def has_key(self, token):
        return token in self.token_to_idx

    def __len__(self):
        return len(self.token_to_idx)

    def lock(self):
        if self.locked:
            return self
        self.mutable = False
        self.build_idx_to_token()
        return self

    def build_idx_to_token(self):
        max_idx = max(self.token_to_idx.values())
        self.idx_to_token = [None] * (max_idx + 1)
        for token, idx in self.token_to_idx.items():
            self.idx_to_token[idx] = token

    def unlock(self):
        if not self.locked:
            return
        self.mutable = True
        self.idx_to_token = None
        return self

    @property
    def locked(self):
        return not self.mutable

    @property
    def unk_idx(self):
        if self.unk_token is None:
            return None
        else:
            return self.token_to_idx.get(self.unk_token, None)

    @property
    def pad_idx(self):
        if self.pad_token is None:
            return None
        else:
            return self.token_to_idx.get(self.pad_token, None)

    @property
    def tokens(self):
        return self.token_to_idx.keys()

    def __str__(self) -> str:
        return self.token_to_idx.__str__()

    def summary(self, verbose=True) -> str:
        # report = 'Length: {}\n'.format(len(self))
        # report += 'Samples: {}\n'.format(str(list(self.token_to_idx.keys())[:min(50, len(self))]))
        # report += 'Mutable: {}'.format(self.mutable)
        # report = report.strip()
        report = '[{}] = '.format(len(self))
        report += str(list(self.token_to_idx.keys())[:min(50, len(self))])
        if verbose:
            print(report)
        return report

    def __call__(self, some_token: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        if isinstance(some_token, (list, tuple, set)):
            indices = []
            if len(some_token) and isinstance(some_token[0], (list, tuple, set)):
                for sent in some_token:
                    inside = []
                    for token in sent:
                        inside.append(self.get_idx(token))
                    indices.append(inside)
                return indices
            for token in some_token:
                indices.append(self.get_idx(token))
            return indices
        else:
            return self.get_idx(some_token)

    def to_dict(self) -> dict:
        idx_to_token = self.idx_to_token
        pad_token = self.pad_token
        unk_token = self.unk_token
        mutable = self.mutable
        items = locals().copy()
        items.pop('self')
        return items

    def copy_from(self, item: dict):
        for key, value in item.items():
            setattr(self, key, value)
        self.token_to_idx = {k: v for v, k in enumerate(self.idx_to_token)}
        return self

    def lower(self):
        self.unlock()
        token_to_idx = self.token_to_idx
        self.token_to_idx = {}
        for token in token_to_idx.keys():
            self.add(token.lower())
        return self

    @property
    def first_token(self):
        if self.idx_to_token:
            return self.idx_to_token[0]
        if self.token_to_idx:
            return next(iter(self.token_to_idx))
        return None

    def merge(self, other):
        for word, idx in other.token_to_idx.items():
            self.get_idx(word)

    @property
    def safe_pad_token(self) -> str:
        """Get the pad token safely. It always returns a pad token, which is the token
        closest to pad if not presented in the vocab.

        Args:

        Returns:

        
        """
        if self.pad_token:
            return self.pad_token
        if self.first_token:
            return self.first_token
        return PAD

    @property
    def safe_pad_token_idx(self) -> int:
        return self.token_to_idx.get(self.safe_pad_token, 0)

    @property
    def safe_unk_token(self) -> str:
        """Get the unk token safely. It always returns a unk token, which is the token
        closest to unk if not presented in the vocab.

        Args:

        Returns:

        
        """
        if self.unk_token:
            return self.unk_token
        if self.first_token:
            return self.first_token
        return UNK

    def __repr__(self) -> str:
        if self.idx_to_token is not None:
            return self.idx_to_token.__repr__()
        return self.token_to_idx.__repr__()

    def extend(self, tokens: Iterable[str]):
        self.unlock()
        self(tokens)

    def reload_idx_to_token(self, idx_to_token: List[str], pad_idx=0, unk_idx=1):
        self.idx_to_token = idx_to_token
        self.token_to_idx = dict((s, i) for i, s in enumerate(idx_to_token))
        if pad_idx is not None:
            self.pad_token = idx_to_token[pad_idx]
        if unk_idx is not None:
            self.unk_token = idx_to_token[unk_idx]

    def set_unk_as_safe_unk(self):
        self.unk_token = self.safe_unk_token

    def clear(self):
        self.unlock()
        self.token_to_idx.clear()


class CustomVocab(Vocab):
    def to_dict(self) -> dict:
        d = super().to_dict()
        d['type'] = classpath_of(self)
        return d


class LowercaseVocab(CustomVocab):
    def get_idx(self, token: str) -> int:
        idx = self.token_to_idx.get(token, None)
        if idx is None:
            idx = self.token_to_idx.get(token.lower(), None)
        if idx is None:
            if self.mutable:
                idx = len(self.token_to_idx)
                self.token_to_idx[token] = idx
            else:
                idx = self.token_to_idx.get(self.unk_token, None)
        return idx


class VocabWithNone(CustomVocab):
    def get_idx(self, token: str) -> int:
        if token is None:
            return -1
        return super().get_idx(token)


class VocabWithFrequency(CustomVocab):

    def __init__(self, counter: Counter = None, min_occur_cnt=0, pad_token=PAD, unk_token=UNK, specials=None) -> None:
        super().__init__(None, None, True, pad_token, unk_token)
        if specials:
            for each in specials:
                counter.pop(each, None)
                self.add(each)
        self.frequencies = [1] * len(self)
        if counter:
            for token, freq in counter.most_common():
                if freq >= min_occur_cnt:
                    self.add(token)
                    self.frequencies.append(freq)
        self.lock()

    def to_dict(self) -> dict:
        d = super().to_dict()
        d['frequencies'] = self.frequencies
        return d

    def copy_from(self, item: dict):
        super().copy_from(item)
        self.frequencies = item['frequencies']

    def get_frequency(self, token):
        idx = self.get_idx(token)
        if idx is not None:
            return self.frequencies[idx]
        return 0


class VocabCounter(CustomVocab):

    def __init__(self, idx_to_token: List[str] = None, token_to_idx: Dict = None, mutable=True, pad_token=PAD,
                 unk_token=UNK) -> None:
        super().__init__(idx_to_token, token_to_idx, mutable, pad_token, unk_token)
        self.counter = Counter()

    def get_idx(self, token: str) -> int:
        if self.mutable:
            self.counter[token] += 1
        return super().get_idx(token)

    def trim(self, min_frequency):
        assert self.mutable
        specials = {self.unk_token, self.pad_token}
        survivors = list((token, freq) for token, freq in self.counter.most_common()
                         if freq >= min_frequency and token not in specials)
        survivors = [(x, -1) for x in specials if x] + survivors
        self.counter = Counter(dict(survivors))
        self.token_to_idx = dict()
        self.idx_to_token = None
        for token, freq in survivors:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx

    def copy_from(self, item: dict):
        super().copy_from(item)
        self.counter = Counter(item['counter'].items()) if 'counter' in item else Counter()

    def to_dict(self) -> dict:
        d = super().to_dict()
        d['counter'] = dict(self.counter.items())
        return d


def create_label_vocab() -> Vocab:
    return Vocab(pad_token=None, unk_token=None)

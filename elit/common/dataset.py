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
import math
import random
import warnings
from abc import ABC, abstractmethod
from copy import copy
from logging import Logger
from typing import Union, List, Callable, Iterable

import torch
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataset import IterableDataset

from elit.common.constant import IDX
from elit.common.structure import AutoConfigurable
from elit.common.transform import TransformList, VocabDict, EmbeddingNamedTransform
from elit.common.vocab import Vocab
from elit.components.parsers.alg import kmeans
from elit.utils.io_util import read_cells, get_resource
from elit.utils.torch_util import dtype_of
from elit.utils.util import isdebugging, merge_list_of_dict, k_fold


class Transformable(ABC):
    def __init__(self, transform: Union[Callable, List] = None) -> None:
        super().__init__()
        if isinstance(transform, list) and not isinstance(transform, TransformList):
            transform = TransformList(*transform)
        self.transform: Union[Callable, TransformList] = transform

    def append_transform(self, transform: Callable):
        assert transform is not None, 'None transform not allowed'
        if not self.transform:
            self.transform = TransformList(transform)
        elif not isinstance(self.transform, TransformList):
            if self.transform != transform:
                self.transform = TransformList(self.transform, transform)
        else:
            if transform not in self.transform:
                self.transform.append(transform)
        return self

    def insert_transform(self, index: int, transform: Callable):
        assert transform is not None, 'None transform not allowed'
        if not self.transform:
            self.transform = TransformList(transform)
        elif not isinstance(self.transform, TransformList):
            if self.transform != transform:
                self.transform = TransformList(self.transform)
                self.transform.insert(index, transform)
        else:
            if transform not in self.transform:
                self.transform.insert(index, transform)
        return self

    def transform_sample(self, sample: dict, inplace=False) -> dict:
        if not inplace:
            sample = copy(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample


class TransformDataset(Transformable, Dataset, ABC):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None) -> None:
        super().__init__(transform)
        if generate_idx is None:
            generate_idx = isinstance(data, list)
        data = self.load_data(data, generate_idx)
        assert data, 'No samples loaded'
        assert isinstance(data[0],
                          dict), f'TransformDataset expects each sample to be a dict but got {type(data[0])} instead.'
        self.data = data
        if cache:
            self.cache = [None] * len(data)
        else:
            self.cache = None

    def load_data(self, data, generate_idx=False):
        if self.should_load_file(data):
            if isinstance(data, str):
                data = get_resource(data)
            data = list(self.load_file(data))
        if generate_idx:
            for i, each in enumerate(data):
                each[IDX] = i
        # elif isinstance(data, list):
        #     data = self.load_list(data)
        return data

    # noinspection PyMethodMayBeStatic
    # def load_list(self, data: list) -> List[Dict[str, Any]]:
    #     return data

    def should_load_file(self, data) -> bool:
        return isinstance(data, str)

    @abstractmethod
    def load_file(self, filepath: str):
        pass

    def __getitem__(self, index: Union[int, slice]) -> Union[dict, List[dict]]:
        # if isinstance(index, (list, tuple)):
        #     assert len(index) == 1
        #     index = index[0]
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]

        if self.cache:
            cache = self.cache[index]
            if cache:
                return cache
        sample = self.data[index]
        sample = self.transform_sample(sample)
        if self.cache:
            self.cache[index] = sample
        return sample

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f'{len(self)} samples: {self[0]} ...'

    def purge_cache(self):
        self.cache = [None] * len(self.data)

    def split(self, *ratios):
        ratios = [x / sum(ratios) for x in ratios]
        chunks = []
        prev = 0
        for r in ratios:
            cur = prev + math.ceil(len(self) * r)
            chunks.append([prev, cur])
            prev = cur
        chunks[-1][1] = len(self)
        outputs = []
        for b, e in chunks:
            dataset = copy(self)
            dataset.data = dataset.data[b:e]
            if dataset.cache:
                dataset.cache = dataset.cache[b:e]
            outputs.append(dataset)
        return outputs

    def k_fold(self, k, i):
        assert 0 <= i <= k, f'Invalid split {i}'
        train_indices, test_indices = k_fold(k, len(self), i)
        return self.subset(train_indices), self.subset(test_indices)

    def subset(self, indices):
        dataset = copy(self)
        dataset.data = [dataset.data[i] for i in indices]
        if dataset.cache:
            dataset.cache = [dataset.cache[i] for i in indices]
        return dataset

    def shuffle(self):
        if not self.cache:
            random.shuffle(self.data)
        else:
            z = list(zip(self.data, self.cache))
            random.shuffle(z)
            self.data, self.cache = zip(*z)

    def prune(self, criterion: Callable, logger: Logger = None):
        # noinspection PyTypeChecker
        size_before = len(self)
        good_ones = [i for i, s in enumerate(self) if not criterion(s)]
        self.data = [self.data[i] for i in good_ones]
        if self.cache:
            self.cache = [self.cache[i] for i in good_ones]
        if logger:
            size_after = len(self)
            num_pruned = size_before - size_after
            logger.info(f'Pruned [yellow]{num_pruned} ({num_pruned / size_before:.1%})[/yellow] '
                        f'samples out of {size_before}.')
        return size_before


class TransformSequentialDataset(Transformable, IterableDataset, ABC):
    pass


class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=None, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 device=None, **kwargs):
        if batch_sampler is not None:
            batch_size = 1
        if num_workers is None:
            if isdebugging():
                num_workers = 0
            else:
                num_workers = 2
        # noinspection PyArgumentList
        super(DeviceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                               sampler=sampler,
                                               batch_sampler=batch_sampler, num_workers=num_workers,
                                               collate_fn=collate_fn,
                                               pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                               worker_init_fn=worker_init_fn,
                                               multiprocessing_context=multiprocessing_context, **kwargs)
        self.device = device

    def __iter__(self):
        for raw_batch in super(DeviceDataLoader, self).__iter__():
            if self.device is not None:
                for field, data in raw_batch.items():
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                        raw_batch[field] = data
            yield raw_batch

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)


class PadSequenceDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 pad: dict = None, vocabs: VocabDict = None, device=None, **kwargs):
        if device == -1:
            device = None
        if collate_fn is None:
            collate_fn = self.collate_fn
        if num_workers is None:
            if isdebugging():
                num_workers = 0
            else:
                num_workers = 2
        if batch_sampler is None:
            assert batch_size, 'batch_size has to be specified when batch_sampler is None'
        else:
            batch_size = 1
            shuffle = None
            drop_last = None
        # noinspection PyArgumentList
        super(PadSequenceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                    sampler=sampler,
                                                    batch_sampler=batch_sampler, num_workers=num_workers,
                                                    collate_fn=collate_fn,
                                                    pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                                    worker_init_fn=worker_init_fn,
                                                    multiprocessing_context=multiprocessing_context, **kwargs)
        self.vocabs = vocabs
        if isinstance(dataset, TransformDataset) and dataset.transform:
            transform = dataset.transform
            if not isinstance(transform, TransformList):
                transform = []
            for each in transform:
                if isinstance(each, EmbeddingNamedTransform):
                    if pad is None:
                        pad = {}
                    if each.dst not in pad:
                        pad[each.dst] = 0
        self.pad = pad
        self.device = device

    def __iter__(self):
        for raw_batch in super(PadSequenceDataLoader, self).__iter__():
            for field, data in raw_batch.items():
                if isinstance(data, torch.Tensor):
                    continue
                vocab_key = field[:-len('_id')] if field.endswith('_id') else None
                vocab: Vocab = self.vocabs.get(vocab_key, None) if self.vocabs and vocab_key else None
                if vocab:
                    pad = vocab.safe_pad_token_idx
                    dtype = torch.long
                elif self.pad is not None and field in self.pad:
                    pad = self.pad[field]
                    dtype = dtype_of(pad)
                elif field.endswith('_offset') or field.endswith('_id') or field.endswith(
                        '_count') or field.endswith('_ids') or field.endswith('_score') or field.endswith(
                    '_length') or field.endswith('_span'):
                    # guess some common fields to pad
                    pad = 0
                    dtype = torch.long
                elif field.endswith('_mask'):
                    pad = False
                    dtype = torch.bool
                else:
                    # no need to pad
                    continue
                data = self.pad_data(data, pad, dtype)
                raw_batch[field] = data
            if self.device is not None:
                for field, data in raw_batch.items():
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                        raw_batch[field] = data
            yield raw_batch

    @staticmethod
    def pad_data(data, pad, dtype=None, device=None):
        if isinstance(data[0], torch.Tensor):
            data = pad_sequence(data, True, pad)
        elif isinstance(data[0], Iterable):
            inner_is_iterable = False
            for each in data:
                if len(each):
                    if isinstance(each[0], Iterable):
                        inner_is_iterable = True
                        if len(each[0]):
                            if not dtype:
                                dtype = dtype_of(each[0][0])
                    else:
                        inner_is_iterable = False
                        if not dtype:
                            dtype = dtype_of(each[0])
                    break
            if inner_is_iterable:
                max_seq_len = len(max(data, key=len))
                max_word_len = len(max([chars for words in data for chars in words], key=len))
                ids = torch.zeros(len(data), max_seq_len, max_word_len, dtype=dtype, device=device)
                for i, words in enumerate(data):
                    for j, chars in enumerate(words):
                        ids[i][j][:len(chars)] = torch.tensor(chars, dtype=dtype, device=device)
                data = ids
            else:
                data = pad_sequence([torch.tensor(x, dtype=dtype, device=device) for x in data], True, pad)
        elif isinstance(data, list):
            data = torch.tensor(data, dtype=dtype, device=device)
        return data

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)


def _prefetch_generator(dataloader, queue, batchify=None):
    while True:
        for batch in dataloader:
            if batchify:
                batch = batchify(batch)
            queue.put(batch)


class PrefetchDataLoader(DataLoader):
    def __init__(self, dataloader, prefetch=10, batchify=None) -> None:
        """
        PrefetchDataLoader only works in spawn mode with the following initialization code:

        if __name__ == '__main__':
            import torch

            torch.multiprocessing.set_start_method('spawn')

        And these 2 lines MUST be put into `if __name__ == '__main__':` block.

        Args:
            dataloader:
            prefetch:
            batchify:
        """
        super().__init__(dataset=dataloader)
        self._batchify = batchify
        self.prefetch = None if isdebugging() else prefetch
        if self.prefetch:
            self._fire_process(dataloader, prefetch)

    def _fire_process(self, dataloader, prefetch):
        self.queue = mp.Queue(prefetch)
        self.process = mp.Process(target=_prefetch_generator, args=(dataloader, self.queue, self._batchify))
        self.process.start()

    def __iter__(self):
        if not self.prefetch:
            for batch in self.dataset:
                if self._batchify:
                    batch = self._batchify(batch)
                yield batch
        else:
            size = len(self)
            while size:
                batch = self.queue.get()
                yield batch
                size -= 1

    def close(self):
        if self.prefetch:
            self.queue.close()
            self.process.terminate()

    @property
    def batchify(self):
        return self._batchify

    @batchify.setter
    def batchify(self, batchify):
        self._batchify = batchify
        if not self.prefetch:
            prefetch = vars(self.queue).get('maxsize', 10)
            self.close()
            self._fire_process(self.dataset, prefetch)


class BucketSampler(Sampler):
    # noinspection PyMissingConstructor
    def __init__(self, buckets, batch_max_tokens, batch_size=None, shuffle=False):
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[
            (size, bucket) for size, bucket in buckets.items()
        ])
        # the number of chunks in each bucket, which is clipped by
        # range [1, len(bucket)]
        if batch_size:
            self.chunks = [
                max(batch_size, min(len(bucket), max(round(size * len(bucket) / batch_max_tokens), 1)))
                for size, bucket in zip(self.sizes, self.buckets)
            ]
        else:
            self.chunks = [
                min(len(bucket), max(round(size * len(bucket) / batch_max_tokens), 1))
                for size, bucket in zip(self.sizes, self.buckets)
            ]

    def __iter__(self):
        # if shuffle, shuffle both the buckets and samples in each bucket
        range_fn = torch.randperm if self.shuffle else torch.arange
        for i in range_fn(len(self.buckets)).tolist():
            split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1 for j in range(self.chunks[i])]
            # DON'T use `torch.chunk` which may return wrong number of chunks
            for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                yield [self.buckets[i][j] for j in batch.tolist()]

    def __len__(self):
        return sum(self.chunks)


class KMeansSampler(BucketSampler):
    def __init__(self, lengths, batch_max_tokens, batch_size=None, shuffle=False, n_buckets=1):
        if n_buckets > len(lengths):
            n_buckets = 1
        self.n_buckets = n_buckets
        self.lengths = lengths
        buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        super().__init__(buckets, batch_max_tokens, batch_size, shuffle)


class SortingSampler(Sampler):
    def __init__(self, lengths: List[int], batch_size=None, batch_max_tokens=None, shuffle=False) -> None:
        # assert any([batch_size, batch_max_tokens]), 'At least one of batch_size and batch_max_tokens is required'
        self.shuffle = shuffle
        self.batch_size = batch_size
        # self.batch_max_tokens = batch_max_tokens
        self.batch_indices = []
        num_tokens = 0
        mini_batch = []
        for i in torch.argsort(torch.tensor(lengths), descending=True).tolist():
            # if batch_max_tokens:
            if (batch_max_tokens is None or num_tokens + lengths[i] <= batch_max_tokens) and (
                    batch_size is None or len(mini_batch) < batch_size):
                mini_batch.append(i)
                num_tokens += lengths[i]
            else:
                if not mini_batch:  # this sequence is longer than  batch_max_tokens
                    mini_batch.append(i)
                    self.batch_indices.append(mini_batch)
                    mini_batch = []
                    num_tokens = 0
                else:
                    self.batch_indices.append(mini_batch)
                    mini_batch = [i]
                    num_tokens = lengths[i]
        if mini_batch:
            self.batch_indices.append(mini_batch)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_indices)
        for batch in self.batch_indices:
            yield batch

    def __len__(self) -> int:
        return len(self.batch_indices)


class SamplerBuilder(AutoConfigurable, ABC):
    @abstractmethod
    def build(self, lengths: List[int], shuffle=False, gradient_accumulation=1, **kwargs) -> Sampler:
        pass

    def __call__(self, lengths: List[int], shuffle=False, **kwargs) -> Sampler:
        return self.build(lengths, shuffle, **kwargs)

    def scale(self, gradient_accumulation):
        batch_size = self.batch_size
        batch_max_tokens = self.batch_max_tokens
        if gradient_accumulation:
            if batch_size:
                batch_size //= gradient_accumulation
            if batch_max_tokens:
                batch_max_tokens //= gradient_accumulation
        return batch_size, batch_max_tokens


class SortingSamplerBuilder(SortingSampler, SamplerBuilder):
    # noinspection PyMissingConstructor
    def __init__(self, batch_size=None, batch_max_tokens=None) -> None:
        self.batch_max_tokens = batch_max_tokens
        self.batch_size = batch_size

    def build(self, lengths: List[int], shuffle=False, gradient_accumulation=1, **kwargs) -> Sampler:
        batch_size, batch_max_tokens = self.scale(gradient_accumulation)
        return SortingSampler(lengths, batch_size, batch_max_tokens, shuffle)

    def __len__(self) -> int:
        return 1


class KMeansSamplerBuilder(KMeansSampler, SamplerBuilder):
    # noinspection PyMissingConstructor
    def __init__(self, batch_max_tokens, batch_size=None, n_buckets=1):
        self.n_buckets = n_buckets
        self.batch_size = batch_size
        self.batch_max_tokens = batch_max_tokens

    def build(self, lengths: List[int], shuffle=False, gradient_accumulation=1, **kwargs) -> Sampler:
        batch_size, batch_max_tokens = self.scale(gradient_accumulation)
        return KMeansSampler(lengths, batch_max_tokens, batch_size, shuffle, self.n_buckets)

    def __len__(self) -> int:
        return 1


class TableDataset(TransformDataset):
    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 delimiter='auto',
                 strip=True,
                 headers=None) -> None:
        self.headers = headers
        self.strip = strip
        self.delimiter = delimiter
        super().__init__(data, transform, cache)

    def load_file(self, filepath: str):
        for idx, cells in enumerate(read_cells(filepath, strip=self.strip, delimiter=self.delimiter)):
            if not idx and not self.headers:
                self.headers = cells
                if any(len(h) > 32 for h in self.headers):
                    warnings.warn('As you did not pass in `headers` to `TableDataset`, the first line is regarded as '
                                  'headers. However, the length for some headers are too long (>32), which might be '
                                  'wrong. To make sure, pass `headers=...` explicitly.')
            else:
                yield dict(zip(self.headers, cells))

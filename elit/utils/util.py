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
from typing import Union, Any, List, Optional, Tuple, Iterable, Dict, Set
import inspect
from itertools import chain, combinations


def powerset(iterable, descending=False):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Args:
        iterable:

    Returns:

    """
    s = list(iterable)
    sizes = range(len(s), -1, -1) if descending else range(len(s) + 1)
    return chain.from_iterable(combinations(s, r) for r in sizes)


def isdebugging():
    """See Also https://stackoverflow.com/questions/333995/how-to-detect-that-python-code-is-being-executed-through-the-debugger"""
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def list_is_list_of_lists(sent: Union[Any, List[Any]]) -> Optional[bool]:
    if not sent:
        return None
    return isinstance(sent[0], list)


def set_tuple_with(t: Tuple, v, at=0) -> Tuple:
    t = list(t)
    t[at] = v
    return tuple(t)


def consume_keys_from_dict(keys: Iterable, d: dict) -> dict:
    consumed = {}
    for k in keys:
        if k in d:
            consumed[k] = d.pop(k)
    return consumed


def merge_dict(d: dict, overwrite=False, inplace=False, **kwargs):
    """Merging the provided dict with other kvs

    Args:
      d: 
      kwargs: 
      d: dict: 
      overwrite:  (Default value = False)
      inplace:  (Default value = False)
      **kwargs: 

    Returns:

    
    """
    nd = dict([(k, v) for k, v in d.items()] + [(k, v) for k, v in kwargs.items() if overwrite or k not in d])
    if inplace:
        d.update(nd)
        return d
    return nd


def merge_locals_kwargs(locals: dict, kwargs: dict = None, excludes=('self', 'kwargs', '__class__')):
    if not kwargs:
        kwargs = dict()
    return merge_dict(dict((k, v) for k, v in list(locals.items())
                           if k not in excludes), **kwargs)


def infer_space_after(sent: List[str]):
    last_token = None
    quote_count: int = 0
    # infer whitespace after field
    whitespace_after = [True] * len(sent)
    for token in range(len(sent)):
        if sent[token] == '"':
            quote_count += 1
            if quote_count % 2 != 0:
                whitespace_after[token] = False
            elif last_token is not None:
                whitespace_after[last_token] = False

        if last_token is not None:

            if sent[token] in [".", ":", ",", ";", ")", "n't", "!", "?"]:
                whitespace_after[last_token] = False

            if sent[token].startswith("'"):
                whitespace_after[last_token] = False

        if sent[token] in ["("]:
            whitespace_after[token] = False

        last_token = token
    return whitespace_after


def ls_resource_in_module(root) -> dict:
    res = dict()
    for k, v in root.__dict__.items():
        if k.startswith('_') or v == root:
            continue
        if isinstance(v, str):
            if v.startswith('http') and not v.endswith('/') and not v.endswith('#') and not v.startswith('_'):
                res[k] = v
        elif type(v).__name__ == 'module':
            res.update(ls_resource_in_module(v))
    if 'ALL' in root.__dict__ and isinstance(root.__dict__['ALL'], dict):
        root.__dict__['ALL'].update(res)
    return res


def collapse_json(text, indent=12):
    """Compacts a string of json data by collapsing whitespace after the
    specified indent level
    
    NOTE: will not produce correct results when indent level is not a multiple
    of the json indent level

    Args:
      text: 
      indent:  (Default value = 12)

    Returns:

    """
    initial = " " * indent
    out = []  # final json output
    sublevel = []  # accumulation list for sublevel entries
    pending = None  # holder for consecutive entries at exact indent level
    for line in text.splitlines():
        if line.startswith(initial):
            if line[indent] == " ":
                # found a line indented further than the indent level, so add
                # it to the sublevel list
                if pending:
                    # the first item in the sublevel will be the pending item
                    # that was the previous line in the json
                    sublevel.append(pending)
                    pending = None
                item = line.strip()
                sublevel.append(item)
                if item.endswith(","):
                    sublevel.append(" ")
            elif sublevel:
                # found a line at the exact indent level *and* we have sublevel
                # items. This means the sublevel items have come to an end
                sublevel.append(line.strip())
                out.append("".join(sublevel))
                sublevel = []
            else:
                # found a line at the exact indent level but no items indented
                # further, so possibly start a new sub-level
                if pending:
                    # if there is already a pending item, it means that
                    # consecutive entries in the json had the exact same
                    # indentation and that last pending item was not the start
                    # of a new sublevel.
                    out.append(pending)
                pending = line.rstrip()
        else:
            if pending:
                # it's possible that an item will be pending but not added to
                # the output yet, so make sure it's not forgotten.
                out.append(pending)
                pending = None
            if sublevel:
                out.append("".join(sublevel))
            out.append(line)
    return "\n".join(out)


class DummyContext(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def merge_list_of_dict(samples: List[Dict]) -> dict:
    batch = {}
    for each in samples:
        for k, v in each.items():
            vs = batch.get(k, None)
            if vs is None:
                vs = []
                batch[k] = vs
            vs.append(v)
    return batch


def reorder(samples: List, order: List[int]) -> List:
    return [samples[i] for i in sorted(range(len(order)), key=lambda k: order[k])]


def k_fold(k, total, i):
    trn = math.ceil(i / k * total)
    tst = math.ceil((i + 1) / k * total)
    return list(range(0, trn)) + list(range(tst, total)), list(range(trn, tst))


def dfs(graph, start):
    seen = set()
    path = []
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)
            path.append(v)
            q.extend(graph[v])

    return path


def topological_sort(graph, start):
    seen = set()
    stack = []
    order = []
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]


def prefix_match(target, sources: Iterable[str]):
    if target is None:
        return None
    if target in sources:
        return target
    for each in sources:
        if each.startswith(target):
            return each

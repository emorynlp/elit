from collections import Counter
from typing import Union, List, Callable

from penman import Graph

from elit.common.dataset import TransformDataset
from elit.components.amr.seq2seq.dataset.IO import read_raw_amr_data
from elit.components.amr.seq2seq.dataset.penman import role_is_reverted
from elit.components.amr.seq2seq.dataset.tokenization_bart import PENMANBartTokenizer


class AMRDataset(TransformDataset):

    def __init__(self,
                 data: Union[str, List],
                 use_recategorization=False,
                 remove_wiki=False,
                 dereify=False,
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None) -> None:
        self.dereify = dereify
        self.remove_wiki = remove_wiki
        self.use_recategorization = use_recategorization
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        graphs = read_raw_amr_data([filepath], self.use_recategorization, remove_wiki=self.remove_wiki,
                                   dereify=self.dereify)
        for g in graphs:
            yield {'amr': g}

    def get_roles(self):
        roles = Counter()
        for sample in self.data:
            g: Graph = sample['amr']
            for s, r, t in g.triples:
                if role_is_reverted(r):
                    r = r[:-3]
                roles[r] += 1
        return roles

    def get_frames(self):
        frames = Counter()
        for sample in self.data:
            g: Graph = sample['amr']
            for i in g.instances():
                t = i.target
                cells = t.split('-')
                if len(cells) == 2 and len(cells[1]) == 2 and cells[1].isdigit():
                    frames[t] += 1
        return frames




def dfs_linearize_tokenize(sample: dict, tokenizer: PENMANBartTokenizer, remove_space=False, text_key='snt') -> dict:
    amr = sample.get('amr', None)
    if amr:
        l, e = tokenizer.linearize(amr)
        sample['graph_tokens'] = e['linearized_graphs']
        sample['graph_token_ids'] = l
        text = amr.metadata[text_key]
    else:
        text = sample['text']
    if remove_space:
        text = ''.join(text.split())
    sample['text'] = text
    sample['text_token_ids'] = tokenizer.encode(text)
    return sample



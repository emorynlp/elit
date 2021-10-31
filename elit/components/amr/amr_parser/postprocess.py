# MIT License
#
# Copyright (c) 2020 Deng Cai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict

from stog.utils import penman
import re
import numpy as np

from elit.common.vocab import Vocab
from elit.components.amr.amr_parser.amr_graph import is_attr_or_abs_form, need_an_instance
from elit.datasets.parsing.amr import largest_connected_component, to_triples
from elit.utils.util import powerset


class PostProcessor(object):
    def __init__(self, rel_vocab):
        self.amr = penman.AMRCodec()
        self.rel_vocab: Vocab = rel_vocab

    def to_triple(self, res_concept, res_relation):
        """
        Take argmax of relations as prediction when prob of arc >= 0.5 (sigmoid).

        Args:
          res_concept: list of strings
          res_relation: list of (dep:int, head:int, arc_prob:float, rel_prob:list(vocab))

        Returns:

        """
        ret = []
        names = []
        for i, c in enumerate(res_concept):
            if need_an_instance(c):
                name = 'c' + str(i)
                ret.append((name, 'instance', c))
            else:
                if c.endswith('_'):
                    name = '"' + c[:-1] + '"'
                else:
                    name = c
                name = name + '@attr%d@' % i
            names.append(name)

        grouped_relation = defaultdict(list)
        # dep head arc rel
        for T in res_relation:
            if len(T) == 4:
                i, j, p, r = T
                r = self.rel_vocab.get_token(int(np.argmax(np.array(r))))
            else:
                i, j, r = T
                p = 1
            grouped_relation[i] = grouped_relation[i] + [(j, p, r)]
        for i, c in enumerate(res_concept):
            if i == 0:
                continue
            max_p, max_j, max_r = 0., 0, None
            for j, p, r in grouped_relation[i]:
                assert j < i
                if is_attr_or_abs_form(res_concept[j]):
                    continue
                if p >= 0.5:
                    if not is_attr_or_abs_form(res_concept[i]):
                        if r.endswith('_reverse_'):
                            ret.append((names[i], r[:-9], names[j]))
                        else:
                            ret.append((names[j], r, names[i]))
                if p > max_p:
                    max_p = p
                    max_j = j
                    max_r = r
            if not max_r:
                continue
            if max_p < 0.5 or is_attr_or_abs_form(res_concept[i]):
                if max_r.endswith('_reverse_'):
                    ret.append((names[i], max_r[:-9], names[max_j]))
                else:
                    ret.append((names[max_j], max_r, names[i]))
        return ret

    def get_string(self, x):
        return self.amr.encode(penman.Graph(x), top=x[0][0])

    def postprocess(self, concept, relation, check_connected=False):
        triples = self.to_triple(concept, relation)
        if check_connected:
            c, e = largest_connected_component(triples)
            triples = to_triples(c, e)
        if check_connected:
            if check_connected is True:
                check_connected = 1000
            # Sometimes penman still complains, so let's use this stupid workaround
            for subset in powerset(triples, descending=True):
                if not subset:
                    break
                try:
                    return self.get_string(subset)
                except penman.EncodeError:
                    pass
                check_connected -= 1
                if not check_connected:
                    break
            return '(c0 / multi-sentence)'
        else:
            mstr = self.get_string(triples)
        return re.sub(r'@attr\d+@', '', mstr)

    def to_amr(self, concept, relation):
        triples = self.to_triple(concept, relation)
        return penman.Graph(triples)

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

import tempfile

import torch

from elit.common.constant import IDX
from elit.components.amr.amr_parser.data import REL
from elit.components.amr.amr_parser.postprocess import PostProcessor
from elit.components.amr.amr_parser.utils import move_to_device
from elit.datasets.parsing.amr import unlinearize
from elit.utils.time_util import CountdownTimer
from elit.utils.util import reorder


def show_progress(model, dev_data):
    model.eval()
    loss_acm = 0.
    for batch in dev_data:
        batch = move_to_device(batch, model.device)
        concept_loss, arc_loss, rel_loss = model(batch)
        loss = concept_loss + arc_loss + rel_loss
        loss_acm += loss.item()
    print('total loss', loss_acm)
    return loss_acm


def parse_batch(model, batch, beam_size, alpha, max_time_step, h=None):
    res = dict()
    concept_batch = []
    relation_batch = []
    levi_graph = model.decoder.levi_graph if hasattr(model, 'decoder') else False
    beams = model.predict(batch, beam_size, max_time_step, h=h)
    score_batch = []
    device = model.device
    for beam in beams:
        best_hyp = beam.get_k_best(1, alpha)[0]
        predicted_concept = [token for token in best_hyp.seq[1:-1]]
        predicted_rel = []
        if levi_graph is True:
            last_concept_id = -1
            edge = []
            rel_mask = [x.startswith(REL) for x in predicted_concept]
            rel_mask = torch.tensor(rel_mask, dtype=torch.bool, device=device)
            for i, c in enumerate(predicted_concept):
                if i == 0:
                    if not c.startswith(REL):
                        last_concept_id = i
                    continue
                if c.startswith(REL):
                    if last_concept_id <= 0:
                        continue
                    arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]
                    arc[rel_mask[:i]] = 0
                    v = last_concept_id
                    p, u = arc[:v].max(0)
                    u = int(u)
                    edge.append((v, i, ''))  # v -> i
                    edge.append((i, u, ''))  # i -> u
                else:
                    last_concept_id = i
            c, e = unlinearize(predicted_concept, edge)
            # Prune unconnected concept
            # c, e = remove_unconnected_components(c, e)
            predicted_concept = c
            predicted_rel = e
        elif levi_graph == 'kahn':
            raise NotImplementedError('Naive BFS is theoretically unable to restore the AMR graph.')
            # last_concept_id = -1
            # edge = []
            # rel_mask = [x.startswith(REL) for x in predicted_concept]
            # rel_mask = torch.tensor(rel_mask, dtype=torch.bool, device=device)
            # for i, c in enumerate(predicted_concept):
            #     if i == 0:
            #         if not c.startswith(REL):
            #             last_concept_id = i
            #         continue
            #     if not c.startswith(REL):
            #         arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]
            #         arc[~rel_mask[:i]] = 0
            #         v = last_concept_id
            #         p, u = arc[:v + 1].max(0) # u is relation
            #         u = int(u)
            #         edge.append((v, i, ''))  #
            #         edge.append((i, u, ''))
            #         last_concept_id = i
            #
            # c, e = unlinearize(predicted_concept, edge)
            # # Prune unconnected concept
            # # c, e = remove_unconnected_components(c, e)
            # predicted_concept = c
            # predicted_rel = e
        else:
            for i in range(len(predicted_concept)):
                if i == 0:
                    continue
                # tgt_len x bsz x tgt_len -> head_len
                arc = best_hyp.state_dict['arc_ll%d' % i].squeeze_().exp_()[1:]  # head_len
                # dep_num x bsz x head_num x vocab_size -> head_len x vocab
                try:
                    _rel = best_hyp.state_dict['rel_ll%d' % i].clone()
                    rel = best_hyp.state_dict['rel_ll%d' % i].squeeze_().exp_()[1:, :]
                except IndexError as e:
                    raise e
                for head_id, (arc_prob, rel_prob) in enumerate(zip(arc.tolist(), rel.tolist())):
                    predicted_rel.append((i, head_id, arc_prob, rel_prob))
        concept_batch.append(predicted_concept)
        score_batch.append(best_hyp.score)
        relation_batch.append(predicted_rel)
    res['concept'] = concept_batch
    res['score'] = score_batch
    res['relation'] = relation_batch
    return res


def parse_data(model, pp: PostProcessor, data, input_file, output_file, beam_size=8, alpha=0.6, max_time_step=100,
               h=None):
    if not output_file:
        output_file = tempfile.NamedTemporaryFile().name
    tot = 0
    levi_graph = model.decoder.levi_graph if hasattr(model, 'decoder') else False
    with open(output_file, 'w') as fo:
        timer = CountdownTimer(len(data))
        order = []
        outputs = []
        for batch in data:
            order.extend(batch[IDX])
            res = parse_batch(model, batch, beam_size, alpha, max_time_step, h=h)
            outputs.extend(list(zip(res['concept'], res['relation'], res['score'])))
            timer.log('Parsing [blink][yellow]...[/yellow][/blink]', ratio_percentage=False)
        outputs = reorder(outputs, order)
        timer = CountdownTimer(len(data))
        for concept, relation, score in outputs:
            fo.write('# ::conc ' + ' '.join(concept) + '\n')
            fo.write('# ::score %.6f\n' % score)
            fo.write(pp.postprocess(concept, relation, check_connected=levi_graph) + '\n\n')
            tot += 1
            timer.log('Post-processing [blink][yellow]...[/yellow][/blink]', ratio_percentage=False)
    match(output_file, input_file)
    # print('write down %d amrs' % tot)


def parse_data_(model, pp: PostProcessor, data, beam_size=8, alpha=0.6, max_time_step=100, h=None):
    levi_graph = model.decoder.levi_graph if hasattr(model, 'decoder') else False
    if levi_graph:
        raise NotImplementedError('Only supports Graph Transducer')
    order = []
    outputs = []
    for batch in data:
        order.extend(batch[IDX])
        res = parse_batch(model, batch, beam_size, alpha, max_time_step, h=h)
        outputs.extend(list(zip(res['concept'], res['relation'], res['score'])))
    outputs = reorder(outputs, order)
    for concept, relation, score in outputs:
        yield pp.to_amr(concept, relation)


def match(output_file, input_file):
    block = []
    blocks = []
    for line in open(input_file, encoding='utf8').readlines():
        if line.startswith('#'):
            block.append(line)
        else:
            if block:
                blocks.append(block)
            block = []

    block1 = []
    blocks1 = []
    for line in open(output_file, encoding='utf8').readlines():
        if not line.startswith('#'):
            block1.append(line)
        else:
            if block1:
                blocks1.append(block1)
            block1 = []
    if block1:
        blocks1.append(block1)
    assert len(blocks) == len(blocks1), (len(blocks), len(blocks1))

    with open(output_file, 'w', encoding='utf8') as fo:
        for block, block1 in zip(blocks, blocks1):
            for line in block:
                fo.write(line)
            for line in block1:
                fo.write(line)

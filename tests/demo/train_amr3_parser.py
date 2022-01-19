# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-04-28 13:25
from elit.common.dataset import SortingSamplerBuilder
from elit.components.amr.seq2seq.seq2seq_amr_parser import Seq2seq_AMR_Parser
from elit.utils.log_util import cprint
from tests import cdroot

cdroot()

parser = Seq2seq_AMR_Parser()
save_dir = 'data/model/amr/amr3_bart_large'
cprint(f'Model will be saved in [cyan]{save_dir}[/cyan]')
parser.fit(
    'data/amr/amr_3.0/train.txt',
    'data/amr/amr_3.0/dev.txt',
    save_dir,
    epochs=30,
    eval_after=20,
    transformer='facebook/bart-large',
    sampler_builder=SortingSamplerBuilder(batch_max_tokens=6000),
    gradient_accumulation=10,
)
parser.load(save_dir)
test_score = parser.evaluate('data/amr/amr_3.0/test.txt', save_dir)[-1]
cprint(f'Official score on testset: [red]{test_score.score:.1%}[/red]')
finegrained = ['Smatch',
               'Unlabeled',
               'No WSD',
               'Concepts',
               'SRL',
               'Reentrancies',
               'Negations',
               'Named Ent.',
               'Wikification']
print('\t'.join(f'{test_score[k].score * 100:.1f}' for k in finegrained))
cprint(f'Model saved in [cyan]{save_dir}[/cyan]')
print(f'To perform wikification, refer to [BLINK](https://github.com/SapienzaNLP/spring/blob/8fad6a4ce59132b22d6bdb4d4eb3a9aa5223ead6/bin/blinkify.py) which will give you the extra `0.3` F1.')

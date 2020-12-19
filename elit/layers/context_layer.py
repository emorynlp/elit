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
from alnlp.modules.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from elit.common.structure import ConfigTracker


class _LSTMSeq2Seq(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            dropout: float = 0.0,
            bidirectional: bool = False,
    ):
        """
        Under construction, not ready for production
        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param bias:
        :param dropout:
        :param bidirectional:
        """
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, embed, lens, max_len):
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, True, total_length=max_len)
        return x


# We might update this to support yaml based configuration
class LSTMContextualEncoder(LstmSeq2SeqEncoder, ConfigTracker):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, dropout: float = 0.0,
                 bidirectional: bool = False, stateful: bool = False):
        super().__init__(input_size, hidden_size, num_layers, bias, dropout, bidirectional, stateful)
        ConfigTracker.__init__(self, locals())

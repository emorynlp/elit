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
# Author: Liyan Xu
import unittest


class TestOnlineCoref(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    @classmethod
    def convert_output_to_context(cls, coref_output):
        from elit.server.format import OnlineCorefContext
        if coref_output is None:
            return None
        return OnlineCorefContext(
            input_ids=coref_output.input_ids,
            sentence_map=coref_output.sentence_map,
            subtoken_map=coref_output.subtoken_map,
            mentions=coref_output.mentions,
            uttr_start_idx=coref_output.uttr_start_idx,
            speaker_ids=coref_output.speaker_ids
        )

    def test_online_coref(self):
        from elit.server.en import en_services
        from elit.server.format import Input
        from elit.components.coref.util import flatten
        utterances = [
            {'speaker_id': 1, 'text': 'I read an article today. It is about US politics.'},
            {'speaker_id': 2, 'text': 'What does it say about US politics?'},
            {'speaker_id': 1, 'text': 'It talks about the US presidential election.'},
            {'speaker_id': 2, 'text': 'I am interested to hear. Can you elaborate more?'},
            {'speaker_id': 1, 'text': 'Sure! The presidential election is indeed interesting.'}
        ]

        context = None
        tokens_to_date = []
        for turn, uttr in enumerate(utterances):
            input_doc = Input(text=uttr['text'], speaker_ids=uttr['speaker_id'],
                              coref_context=self.convert_output_to_context(context), models=['ocr'])
            output_doc = en_services.online_coref.predict_sequentially(input_doc, check_sanitization=True)
            print(output_doc)
            context = output_doc['ocr']

            # Print cluster text
            tokens_to_date += flatten(input_doc.tokens)
            for cluster in output_doc['ocr'].clusters:
                for i in range(len(cluster)):
                    m1, m2 = cluster[i]
                    cluster[i] = (m1, m2, ' '.join(tokens_to_date[m1:m2+1]))
            print(f'cluster text: {output_doc["ocr"].clusters}')
            print()


if __name__ == '__main__':
    unittest.main()

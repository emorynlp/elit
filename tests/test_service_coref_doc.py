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
import time


class TestDocCoref(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_doc_coref(self):
        from elit.server.en import en_services
        from elit.server.format import Input
        text = 'Pfizer said last week it may need the U.S. government to help it secure some components needed to ' \
               'make the vaccine. While the company halved its 2020 production target due to manufacturing issues, ' \
               'it said last week its manufacturing is running smoothly now. The government also has the option to ' \
               'acquire up to an additional 400 million doses of the vaccine.'

        batch_size = 32
        inputs = [Input(text=text[:], models=['dcr'])] * batch_size

        start_time = time.time()
        docs = en_services.doc_coref.predict(inputs)
        end_time = time.time()
        print(f'Doc coref time elapse for {batch_size} small documents: {end_time - start_time :.2f}s')

        assert len(docs) == len(inputs)
        print(docs[-1])


if __name__ == '__main__':
    unittest.main()

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
# Author: hankcs, Liyan Xu
import elit
from elit.pretrained.coref import DOC_COREF_SPANBERT_LARGE_EN, ONLINE_COREF_SPANBERT_LARGE_EN
from elit.server.en_parser import service_tokenizer, service_parser
from elit.server.format import Input
from elit.server.service_tokenizer import ServiceTokenizer
from elit.server.service_parser import ServiceParser
from elit.server.service_coref import ServiceCoreference


class BundledServices:
    def __init__(self,
                 tokenizer: ServiceTokenizer = None,
                 parser: ServiceParser = None,
                 doc_coref: ServiceCoreference = None,
                 online_coref: ServiceCoreference = None):
        self.tokenizer = tokenizer
        self.parser = parser
        self.doc_coref = doc_coref
        self.online_coref = online_coref
        self.emotion_detection = None


service_doc_coref = ServiceCoreference(
    service_tokenizer=service_tokenizer,
    models=elit.load(DOC_COREF_SPANBERT_LARGE_EN)
)
# service_doc_coref = ServiceCoreference(
#     service_tokenizer=service_tokenizer,
#     models=[elit.load(DOC_COREF_SPANBERT_LARGE_EN, devices=0),
#             elit.load(DOC_COREF_SPANBERT_LARGE_EN, devices=1)]
# )

service_online_coref = ServiceCoreference(
    service_tokenizer=service_tokenizer,
    models=elit.load(ONLINE_COREF_SPANBERT_LARGE_EN)
)

en_services = BundledServices(
    tokenizer=service_tokenizer,
    parser=service_parser,
    doc_coref=service_doc_coref,
    online_coref=service_online_coref
)


def main():
    text = [
        "Emory NLP is a research lab in Atlanta, GA. "
        "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
    ]
    input = Input(text=text)
    input.models = ['lem']
    docs = en_services.parser.parse([input])
    for doc in docs:
        print(doc)

    # See elit.client for coreference examples
    text = 'Pfizer said last week it may need the U.S. government to help it secure some components needed to ' \
           'make the vaccine. While the company halved its 2020 production target due to manufacturing issues, ' \
           'it said last week its manufacturing is running smoothly now. The government also has the option to ' \
           'acquire up to an additional 400 million doses of the vaccine.'
    input_doc = Input(text=text, models=['dcr'])
    doc = service_doc_coref.predict(input_doc)
    print(doc)


if __name__ == '__main__':
    main()

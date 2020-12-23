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
from typing import List, Callable, Union, Optional

from elit.components.coref.dto import CorefInput, CorefOutput
from elit.server.service_tokenizer import ServiceTokenizer
from elit.server.format import Input, OnlineCorefContext
from elit.server.en_util import eos, tokenize
from elit.common.document import Document


class ServiceCoreference:

    def __init__(self,
                 model: Callable[[CorefInput], CorefOutput],
                 service_tokenizer: ServiceTokenizer) -> None:
        self.model = model
        self.service_tokenizer = ServiceTokenizer(eos, tokenize) if service_tokenizer is None else service_tokenizer
        self.identifier = 'ocr' if self.model.config['online'] else 'dcr'

    def _translate_context(self, context: OnlineCorefContext) -> CorefOutput:
        return CorefOutput(
            input_ids=context.inputs_ids,
            sentence_map=context.sentence_map,
            subtoken_map=context.subtoken_map,
            mentions=context.mentions,
            uttr_start_idx=context.uttr_start_idx,
            speaker_ids=context.speaker_ids
        )

    def _translate_to_coref(self, input_doc: Input) -> Optional[CorefInput]:
        if self.identifier not in input_doc.models:
            return None
        return CorefInput(
            doc_or_uttr=input_doc.tokens,
            speaker_ids=input_doc.speaker_ids,
            genre=input_doc.genre,
            context=self._translate_context(input_doc.coref_context),
            return_prob=input_doc.return_coref_prob,
            language=input_doc.language,
            verbose=True if self.identifier == 'ocr' else input_doc.verbose
        )

    def _translate_from_coref(self, coref_output: Optional[CorefOutput], input_doc: Input) -> Document:
        if coref_output is None:
            return Document({'tokens': input_doc.tokens})
        else:
            return Document({
                'tokens': input_doc.tokens,
                self.identifier: coref_output
            })

    def _predict_single(self, coref_input: Optional[CorefInput]) -> Optional[CorefOutput]:
        if coref_input is None:
            return None
        return self.model(coref_input)

    def predict(self, inputs: Union[Input, List[Input]]) -> Union[Document, List[Document]]:
        """ Sequential prediction on multiple input docs. """
        self.service_tokenizer.tokenize_inputs(inputs)  # no effects (read-only) in server pipeline

        if isinstance(inputs, Input):
            return self._translate_from_coref(self._predict_single(self._translate_to_coref(inputs)), inputs)

        coref_inputs = [self._translate_to_coref(input_doc) for input_doc in inputs]
        output_docs = [self._translate_from_coref(self._predict_single(coref_input), inputs[i])
                       for i, coref_input in enumerate(coref_inputs)]
        return output_docs

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
from typing import List, Callable, Union, Optional, Any, Tuple
import asyncio
import random

from elit.components.coref.dto import CorefInput, CorefOutput
from elit.server.service_tokenizer import ServiceTokenizer
from elit.server.format import Input, OnlineCorefContext
from elit.server.en_util import eos, tokenize
from elit.common.document import Document

CorefCallable = Callable[[CorefInput], CorefOutput]


class ServiceCoreference:

    def __init__(self,
                 service_tokenizer: ServiceTokenizer,
                 models: Union[CorefCallable, List[CorefCallable]]) -> None:
        """
        Service for coreference resolution that supports async concurrent prediction on multi-GPUs.

        Models should either all be doc coref or all be online coref.
        The performance of the current model implementation is completely throttled by the mention extraction on CPU,
        therefore multi-GPUs do not help directly.
        Args:
            service_tokenizer ():
            models (): if more than one models, they should be placed on separate GPUs
        """
        self.service_tokenizer = ServiceTokenizer(eos, tokenize) if service_tokenizer is None else service_tokenizer
        if not isinstance(models, list):
            models = [models]
        self.models = models

        is_online = self.models[0].config['online']
        assert all(model.config['online'] == is_online for model in self.models), 'models should be of same type'

    @property
    def identifier(self) -> str:
        return 'ocr' if self.models[0].config['online'] else 'dcr'

    def _translate_context(self, context: OnlineCorefContext) -> CorefOutput:
        return CorefOutput(
            input_ids=context.input_ids,
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
            context=self._translate_context(input_doc.coref_context) if input_doc.coref_context else None,
            return_prob=input_doc.return_coref_prob,
            language=input_doc.language,
            verbose=False if self.identifier == 'ocr' else input_doc.verbose
        )

    def _translate_from_coref(self, coref_output: Optional[Union[CorefOutput, List[List[Tuple[Any]]]]],
                              input_doc: Input, return_tokens: bool = True) -> Document:
        if coref_output is None:
            return Document({'tokens': input_doc.tokens})
        elif return_tokens:
            return Document({
                'tokens': input_doc.tokens,
                self.identifier: vars(coref_output) if self.identifier == 'ocr' else coref_output
            })
        else:
            return Document({self.identifier: vars(coref_output) if self.identifier == 'ocr' else coref_output})

    def _predict_single(self, coref_input: Optional[CorefInput], model: CorefCallable,
                        **kwargs) -> Optional[CorefOutput]:
        if coref_input is None:
            return None
        return model(coref_input, **kwargs)

    def predict_sequentially(self, inputs: Union[Input, List[Input]], **kwargs) -> Union[Document, List[Document]]:
        """ Sequential prediction on multiple input docs; randomly pick a single model. """
        return_tokens = kwargs.get('return_tokens', True)
        single_input = False
        if isinstance(inputs, Input):
            single_input = True
            inputs = [inputs]

        self.service_tokenizer.tokenize_inputs(inputs)  # no effects (read-only) in server pipeline
        model = random.choice(self.models)

        coref_inputs = [self._translate_to_coref(input_doc) for input_doc in inputs]
        output_docs = [self._translate_from_coref(self._predict_single(coref_input, model, **kwargs), inputs[i],
                                                  return_tokens) for i, coref_input in enumerate(coref_inputs)]

        return output_docs[0] if single_input else output_docs

    async def _predict_single_routine(self, coref_input: Optional[CorefInput], model: CorefCallable,
                                      **kwargs) -> Optional[CorefOutput]:
        return self._predict_single(coref_input, model, **kwargs)

    async def _predict_routine(self, inputs: Union[Input, List[Input]], **kwargs) -> Union[Document, List[Document]]:
        return_tokens = kwargs.get('return_tokens', True)
        single_input = False
        if isinstance(inputs, Input):
            single_input = True
            inputs = [inputs]

        self.service_tokenizer.tokenize_inputs(inputs)  # no effects (read-only) in server pipeline
        model_picks = random.choices(self.models, k=len(inputs))

        coref_inputs = [self._translate_to_coref(input_doc) for input_doc in inputs]
        coref_outputs = asyncio.gather(*[self._predict_single_routine(coref_input, model, **kwargs)
                                         for coref_input, model in zip(coref_inputs, model_picks)])

        output_docs = [self._translate_from_coref(coref_output, inputs[i], return_tokens)
                       for i, coref_output in enumerate(await coref_outputs)]
        return output_docs[0] if single_input else output_docs

    def predict(self, inputs: Union[Input, List[Input]], **kwargs) -> Union[Document, List[Document]]:
        """ Concurrent prediction on multiple input docs; randomly distribute on all models. """
        output_docs = asyncio.run(self._predict_routine(inputs, **kwargs))
        return output_docs

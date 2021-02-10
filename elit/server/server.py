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
import asyncio
import functools
import traceback
from typing import List, Any

import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from elit.common.document import Document
from elit.server.en import en_services, BundledServices
from elit.server.format import Input
import sys

if sys.version_info[:2] < (3, 7):
    raise NotImplementedError('Server requires Python 3.7 or later.')

app = FastAPI()


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.code = code
        self.msg = msg


class ModelRunner(object):
    def __init__(self, services: BundledServices, max_queue_size=128, max_batch_size=32, max_wait=0.05):
        """

        Args:
            max_queue_size: we accept a backlog of MAX_QUEUE_SIZE before handing out "Too busy" errors
            max_batch_size: we put at most MAX_BATCH_SIZE things in a single batch
            max_wait: we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching
        """
        self.services = services
        self.max_wait = max_wait
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.queue = []
        self.queue_lock = None
        self.model = None
        self.needs_processing = None
        self.needs_processing_timer = None

    def schedule_processing_if_needed(self, loop):
        if len(self.queue) >= self.max_batch_size:
            logger.info("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.info("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = loop.call_at(self.queue[0]["time"] + self.max_wait, self.needs_processing.set)

    async def process_input(self, input):
        loop = asyncio.get_running_loop()
        our_task = {"done_event": asyncio.Event(loop=loop),
                    "input": input,
                    "time": loop.time()}
        async with self.queue_lock:
            if len(self.queue) >= self.max_queue_size:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task)
            logger.info("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed(loop)

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, batch: List[Input]) -> List[Any]:  # runs in other thread
        # After tokenization, we could run parsing and coreference concurrently
        # However, since GIL restricts only one process, the bottleneck is on CPU;
        # concurrent execution doesn't increase throughput in this case
        try:
            batch = self.services.tokenizer.tokenize_inputs(batch)
            docs = self.services.parser.parse(batch)
            docs_coref_doc = self.services.doc_coref.predict(batch, return_tokens=False)
            docs_coref_online = self.services.online_coref.predict(batch, return_tokens=False)
            for doc, doc_coref_doc, doc_coref_online in zip(docs, docs_coref_doc, docs_coref_online):
                doc.update(doc_coref_doc)
                doc.update(doc_coref_online)
            return docs
        except Exception as e:
            traceback.print_exc()
            return [e for _ in batch]

    async def model_runner(self):
        loop = asyncio.get_running_loop()
        self.queue_lock = asyncio.Lock(loop=loop)
        self.needs_processing = asyncio.Event(loop=loop)
        logger.info("started model runner")
        while True:
            logger.info('Waiting for needs_processing')
            await self.needs_processing.wait()
            self.needs_processing.clear()
            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            logger.info('Locking queue_lock')
            async with self.queue_lock:
                if self.queue:
                    longest_wait = loop.time() - self.queue[0]["time"]
                else:  # oops
                    longest_wait = None
                logger.info(
                    "launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))
                to_process = self.queue[:self.max_batch_size]
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed(loop)
            # so here we copy, it would be neater to avoid this
            batch = [t["input"] for t in to_process]
            result = await loop.run_in_executor(
                None, functools.partial(self.run_model, batch)
            )
            for t, r in zip(to_process, result):
                t["output"] = r
                t["done_event"].set()
            del to_process


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(runner.model_runner())


runner = ModelRunner(en_services)


# noinspection PyShadowingBuiltins
@app.post("/parse")
async def parse(input: Input):
    output: Document = await runner.process_input(input)
    if not isinstance(output, Document):
        raise HandlingError("Internal Server Error", code=500)
    return output.to_dict()


@app.get("/parse")
async def parse(text: str):
    input = Input(text=text)
    output: Document = await runner.process_input(input)
    if not isinstance(output, Document):
        raise HandlingError("Internal Server Error", code=500)
    return output.to_dict()


def run(host='0.0.0.0', port=8000, reload=False, workers=1):
    uvicorn.run('elit.server.server:app', host=host, port=port, reload=reload, workers=workers)


def main():
    run(reload=True)


if __name__ == '__main__':
    main()

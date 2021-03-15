# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-01-08 12:44
import elit
from elit.pretrained.mtl import LEM_POS_NER_DEP_ROBERTA_BASE_EN
from elit.server.en_util import eos, tokenize
from elit.server.service_parser import ServiceParser
from elit.server.service_tokenizer import ServiceTokenizer

service_tokenizer = ServiceTokenizer(eos, tokenize)
service_parser = ServiceParser(
    service_tokenizer=service_tokenizer,
    model=elit.load(LEM_POS_NER_DEP_ROBERTA_BASE_EN)
)

# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-18 20:15
from typing import Union, List

from pydantic import BaseModel


class Input(BaseModel):
    text: Union[str, List[str]] = None
    tokens: List[List[str]] = None
    models: List[str] = ["lem", "pos", "ner", "con", "dep", "srl", "amr"]
    language: str = 'en'
    verbose: bool = True

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

from os.path import abspath, join, dirname
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()
version = {}
with open(join(this_dir, "elit", "version.py")) as fp:
    exec(fp.read(), version)

setup(
    name='elit',
    version=version['__version__'],
    description='The Emory Language and Information Toolkit (ELIT)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/emorynlp/elit',
    author='Han He, Liyan Xu',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Development Status :: 3 - Alpha",
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Topic :: Text Processing :: Linguistic"
    ],
    keywords='corpus,machine-learning,NLU,NLP',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=[
        'sentencepiece==0.1.91',
        'termcolor',
        'phrasetree',
        'pynvml',
        'alnlp',
        'penman==0.6.2',
        'toposort==1.5',
        'unofficial_stog>=0.0.20',
        'uvicorn',
        'fastapi==0.65.2',
        'transformers',
        'scipy'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'elit=elit.main:main',
        ],
    },
)

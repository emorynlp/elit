# Getting Started

## Install

```
conda install pytorch cudatoolkit=10.2 -c pytorch      # [Optional] GPU support, adjust the cuda version 
git clone https://github.com/emorynlp/elit.git
cd elit
pip install -e .
```

## CLI

### Interactive 

Type the following command to launch an interactive parsing system. After the launch, type a sentence and hit enter to parse it.

```
$ elit parse
Emory NLP is a research lab in Atlanta.
{
  "lem": [
    ["emory", "nlp", "be", "a", "research", "lab", "in", "atlanta", "."]
  ],
  "pos": [
    ["NNP", "NNP", "VBZ", "DT", "NN", "NN", "IN", "NNP", "."]
  ],
  "ner": [
    [["ORG", 0, 2, "Emory NLP"], ["GPE", 7, 8, "Atlanta"]]
  ],
  "srl": [
    [[["ARG1", 0, 2, "Emory NLP"], ["PRED", 2, 3, "is"], ["ARG2", 3, 8, "a research lab in Atlanta"]]]
  ],
  "dep": [
    [[1, "com"], [5, "nsbj"], [5, "cop"], [5, "det"], [5, "com"], [-1, "root"], [7, "case"], [5, "ppmod"], [5, "p"]]
  ],
  "con": [
    ["TOP", [["S", [["NP", [["NNP", ["Emory"]], ["NNP", ["NLP"]]]], ["VP", [["VBZ", ["is"]], ["NP", [["NP", [["DT", ["a"]], ["NN", ["research"]], ["NN", ["lab"]]]], ["PP", [["IN", ["in"]], ["NP", [["NNP", ["Atlanta"]]]]]]]]]], [".", ["."]]]]]]
  ],
  "amr": [
    [["c0", "instance", "lab"], ["c1", "instance", "research-01"], ["c2", "instance", "atlanta"], ["c3", "instance", "atlanta"], ["c1", "domain", "c0"], ["c0", "location", "c2"], ["c0", "location", "c3"]]
  ],
  "tok": [
    ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", "."]
  ]
}
```

### Server

Launch the server with the following command and test it with any http request tool.

```
$ elit serve
$ curl http://0.0.0.0:8000/parse?text=Emory%20NLP%20is%20a%20research%20lab%20in%20Atlanta. | json_pp -json_opt pretty,canonical
```

You can also post json request to the server with fine-grained control specified in [Data Format](data_format.md). We also offer a client which implments ELIT protocol.

```python
from elit.client import Client

def _test_raw_text():
    text = "Emory NLP is a research lab in Atlanta, GA. " \
           "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(text))


def _test_sents():
    text = ["Emory NLP is a research lab in Atlanta, GA. ",
            "It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."]
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(text))


def _test_tokens():
    tokens = [
        ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
        ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", ".", "Dr.", "Choi", "is", "a", "professor",
         "at", "Emory", "University", "."]
    ]
    nlp = Client('http://0.0.0.0:8000')
    print(nlp.parse(tokens=tokens, models=['ner', 'srl', 'dep'], verbose=True))
```


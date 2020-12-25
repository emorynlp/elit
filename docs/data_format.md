# Data Format

## Input Format

ELIT requires the input to include either the `text` or `tokens` field in the JSON format.
The value of `text` can be either a string or a list of strings.
If `text` is a string, ELIT uses its [tokenizer](tokenization.md) to first segment the raw text into sentences then split each sentence into tokens.


```json
{
    "text": "Emory NLP is a research lab in Atlanta, GA. It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
}
```

If `text` is a list of strings, ELIT considers every string as a sentence and splits it into tokens.

```json
{
    "text": [
        "Emory NLP is a research lab in Atlanta, GA.",
        "It is founded by Jinho D. Choi in 2014.",
        "Dr. Choi is a professor at Emory University."
    ]
}
```

The value of `tokens` is a list of (list of strings), where the outer list represents sentences and each inter list represents tokens in the corresponding sentence.

```json
{
    "tokens": [
        ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
        ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", "."],
        ["Dr.", "Choi", "is", "a", "professor", "at", "Emory", "University", "."]
    ]
}
```

If both the `text` and `token` fields are provided, ELIT ignores `text` and processes only with the value in the `tokens` field.

The input JSON also needs to include the `models` and `language` fields.

```json
{
    "text": "Emory NLP is a research lab in Atlanta, GA. It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University.",
    "models": ["lem", "pos", "ner", "con", "dep", "srl", "amr", "coref"],
    "language": "en",
    "verbose": true
}
```

* `verbose`: if true, the output includes the word forms of all spans.


## Output Format

```json
{
    "text": "Emory NLP is a research lab in Atlanta, GA. It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University.",
    "tokens": [
        ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
        ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", "."],
        ["Dr.", "Choi", "is", "a", "professor", "at", "Emory", "University", "."]
    ],
    "lem": [
        ["emory", "nlp", "be", "a", "research", "lab", "in", "atlanta", ",", "ga", "."],
        ["it", "be", "found", "by", "jinho", "d.", "choi", "in", "2014", "."],
        ["dr.", "choi", "be", "a", "professor", "at", "emory", "university", "."]
    ],
    "pos": [
        ["NNP", "NNP", "VBZ", "DT", "NN", "NN", "IN", "NNP", ",", "NNP", "."],
        ["PRP", "VBZ", "VBN", "IN", "NNP", "NNP", "NNP", "IN", "CD", "."],
        ["NNP", "NNP", "VBZ", "DT", "NN", "IN", "NNP", "NNP", "."]
    ],
    "ner": [
        [[0, 2, "ORG", "Emory NLP"], [7, 10, "LOC", "Atlanta , GA"]],
        [[4, 7, "PERSON", "Jinho D. Choi"], [8, 9, "DATE", "2014"]],
        [[0, 2, "PERSON", "Dr. Choi"], [6, 8, "ORG", "Emory University"]]
    ],
    "con": [
        ["TOP", [["S", [["NP", [["NNP", ["Emory"]], ["NNP", ["NLP"]]]], ["VP", [["VBZ", ["is"]], ["NP", [["NP", [["DT", ["a"]], ["NN", ["research"]], ["NN", ["lab"]]]], ["PP", [["IN", ["in"]], ["NP", [["NP", [["NNP", ["Atlanta"]]]], [",", [","]], ["NP", [["NNP", ["GA"]]]]]]]]]]]], [".", ["."]]]]]],
        ["TOP", [["S", [["NP", [["PRP", ["It"]]]], ["VP", [["VBZ", ["is"]], ["VP", [["VBN", ["founded"]], ["PP", [["IN", ["by"]], ["NP", [["NNP", ["Jinho"]], ["NNP", ["D."]], ["NNP", ["Choi"]]]]]], ["PP", [["IN", ["in"]], ["NP", [["CD", ["2014"]]]]]]]]]], [".", ["."]]]]]],
        ["TOP", [["S", [["NP", [["NNP", ["Dr."]], ["NNP", ["Choi"]]]], ["VP", [["VBZ", ["is"]], ["NP", [["NP", [["DT", ["a"]], ["NN", ["professor"]]]], ["PP", [["IN", ["at"]], ["NP", [["NNP", ["Emory"]], ["NNP", ["University"]]]]]]]]]], [".", ["."]]]]]]
    ],
    "dep": [
        [[1, "com"], [5, "nsbj"], [5, "cop"], [5, "det"], [5, "com"], [-1, "root"], [7, "case"], [5, "ppmod"], [7, "p"], [7, "appo"], [5, "p"]],
        [[2, "obj"], [2, "aux"], [-1, "root"], [6, "case"], [6, "com"], [6, "com"], [2, "nsbj"], [8, "case"], [2, "ppmod"], [2, "p"]],
        [[1, "com"], [4, "nsbj"], [4, "cop"], [4, "det"], [-1, "root"], [7, "case"],  [7, "com"], [4, "ppmod"], [4, "p"]]
    ],
    "srl": [
        [
            [[2, 3 "PRED", "is"],  [0, 2, "ARG1", "Emory NLP"],  [3, 6, "ARG2", "a research lab"],  [6, 10, "ARGM-LOC", "in Atlanta , GA"]]
        ],
        [
            [[2, 3, "PRED", "founded"], [3, 7, "ARG0", "by Jinho D. Choi"], [0, 1, "ARG1", "It"], [7, 9, "ARGM-TMP", "in 2014"]]
        ],
        [
            [[2, 3, "PRED", "is"], [0, 2, "ARG1", "Dr. Choi"], [3, 5, "ARG2", "a professor"], [5, 8, "PRED", "at Emory University"]]
        ]
    ],
    "amr": [
        [["c0", "ARG1", "c1"], ["c0", "ARG2", "c2"], ["c6", "domain", "c2"], ["c0", "instance", "have-mod-91"], ["c1", "instance", "name"], ["c2", "instance", "lab"], ["c5", "instance", "city"], ["c6", "instance", "research-01"], ["c7", "instance", "state"], ["c2", "location", "c5"], ["c5", "location", "c7"], ["c1", "op1", "\"emory\"@attr3@"], ["c1", "op2", "\"atlanta\"@attr4@"]],
        [["c0", "ARG0", "c2"], ["c4", "ARG0", "c2"], ["c0", "ARG1", "c1"], ["c4", "ARG2", "c5"], ["c6", "domain", "c5"], ["c7", "domain", "c5"], ["c0", "instance", "found-01"], ["c1", "instance", "it"], ["c2", "instance", "person"], ["c4", "instance", "have-org-role-91"], ["c5", "instance", "officer"], ["c6", "instance", "executive"], ["c7", "instance", "chief"], ["c0", "time", "2014@attr3@"]],
        [["c0", "ARG0", "c2"], ["c0", "ARG1", "c1"], ["c0", "ARG1", "c3"], ["c0", "ARG1", "c4"], ["c0", "instance", "have-org-role-91"], ["c1", "instance", "professor"], ["c2", "instance", "doctor"], ["c3", "instance", "doctor"], ["c4", "instance", "university"], ["c5", "instance", "emory"], ["c4", "name", "c5"]]
    ],
    "coref": [
        [[0, 0, 2, "Emory NLP"], [1, 0, 1, "It"]],
        [[1, 4, 7, "Jinho D. Choi"], [2, 0, 2, "Dr. Choi"]]
    ]
}
```

If  `verbose` is `false`, the following changes are made:

```json
{
    "ner": [
        [[0, 2, "ORG"], [7, 10, "LOC"]],
        [[4, 7, "PERSON"], [8, 9]],
        [[0, 2, "PERSON"], [6, 8, "ORG"]]
    ],
    "srl": [
        [
            [[2, 3 "PRED"],  [0, 2, "ARG1"],  [3, 6, "ARG2"],  [6, 10, "ARGM-LOC"]]
        ],
        [
            [[2, 3, "PRED"], [3, 7, "ARG0"], [0, 1, "ARG1"], [7, 9, "ARGM-TMP"]]
        ],
        [
            [[2, 3, "PRED"], [0, 2, "ARG1"], [3, 5, "ARG2"], [5, 8, "PRED"]]
        ]
    ],
    "coref": [
        [[0, 0, 2], [1, 0, 1]],
        [[1, 4, 7], [2, 0, 2]]
    ]
}
```

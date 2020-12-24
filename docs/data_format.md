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
    "models": ["lem", "pos", "ner", "con", "dep", "srl", "amr", "dcr", "ocr"],
    "language": "en",
    "verbose": true
}
```

* `verbose`: if true, the output includes the word forms of all spans.

For coreference resolution models, `dcr` resolves the traditional document coreference, and `ocr` resolves the online coreference for dialogues. The following fields can also be included for `dcr` or `ocr`: `speaker_ids`, `genre`, `coref_context`, `return_coref_prob` (see [format.py](elit/server/format.py)). 

In addition, `coref_context` is required for `ocr` with 1+ turns. It can be obtained from the online output of the previous utterance. See [client.py](elit/client.py) for examples.

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
        [],
        [],
        []
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
        [],
        [],
        []
    ],
    "dcr": [
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
    "dcr": [
        [[0, 0, 2], [1, 0, 1]],
        [[1, 4, 7], [2, 0, 2]]
    ]
}
```

It should be noted that for `ocr`, the cluster mentions are indexed by the global token indices (starting from the beginning of the dialogue) so that it can refer to mentions in the past context. The model itself doesn't know about the sentence id in the past context; it only knows the global token indices upon each input.

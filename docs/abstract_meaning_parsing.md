# Abstract Meaning Parsing

Abstract Meaning Representation parsing is the task of parsing a sentence to a representation of the abstract meaning in a sentence as a rooted, directed, acyclic graph.

ELIT provides two APIs for parsing a raw sentence into an AMR graph.

## State-of-The-Art AMR Parser

We recommend to use the SOTA AMR parser with `83.3` Smatch score trained on the AMR 3.0 dataset:

```python
import elit
parser = elit.load(elit.pretrained.amr.AMR3_BART_LARGE_EN)
amr = parser('The boy wants the girl to believe him.')
print(amr)
```

Outputs:

```text
(z0 / want-01
    :ARG0 (z1 / boy)
    :ARG1 (z2 / believe-01
              :ARG0 (z3 / girl)
              :ARG1 z1))
```

Feel free to parse multiple sentences in one batch to benefit from parallelization.

### Evaluation

Its detailed performance is listed as follows:

```bash
$ cat ~/.elit/amr3_bart_large_elit_20211030_173300/test.log
{Smatch P: 84.00% R: 82.60% F1: 83.30%}{Unlabeled P: 86.40% R: 84.90% F1: 85.70%}{No WSD P: 84.50% R: 83.10% F1: 83.80%}{Non_sense_frames P: 91.90% R: 91.30% F1: 91.60%}{Wikification P: 81.70% R: 80.80% F1: 81.20%}{Named Ent. P: 89.20% R: 87.00% F1: 88.10%}{Negations P: 71.70% R: 70.90% F1: 71.30%}{IgnoreVars P: 73.80% R: 73.10% F1: 73.50%}{Concepts P: 90.70% R: 89.60% F1: 90.10%}{Frames P: 88.50% R: 87.90% F1: 88.20%}{Reentrancies P: 70.40% R: 71.80% F1: 71.10%}{SRL P: 79.00% R: 79.60% F1: 79.30%}
```

To re-produce this evaluation, run the followling code:

```python
import elit
parser = elit.load(elit.pretrained.amr.AMR3_BART_LARGE_EN)
parser.evaluate('data/amr/amr_3.0/test.txt', '/tmp')
```

which will generate the Smatch score `/tmp/test.log` **before** wikification:

```
81/81 {Smatch P: 83.70% R: 82.40% F1: 83.00%}{Unlabeled P: 86.10% R: 84.60% F1: 85.40%}{No WSD P: 84.20% R: 82.90% F1: 83.50%}{Non_sense_frames P: 91.90% R: 91.30% F1: 91.60%}{Wikification P: 75.00% R: 74.20% F1: 74.60%}{Named Ent. P: 89.20% R: 87.00% F1: 88.10%}{Negations P: 71.70% R: 70.90% F1: 71.30%}{IgnoreVars P: 73.50% R: 72.80% F1: 73.10%}{Concepts P: 90.70% R: 89.60% F1: 90.10%}{Frames P: 88.50% R: 87.90% F1: 88.20%}{Reentrancies P: 70.40% R: 71.80% F1: 71.10%}{SRL P: 79.00% R: 79.60% F1: 79.30%} ET: 8 m 1 s
```

The predictions are saved in `/tmp/test.pred.txt`. To perform wikification, refer to [BLINK](https://github.com/SapienzaNLP/spring/blob/8fad6a4ce59132b22d6bdb4d4eb3a9aa5223ead6/bin/blinkify.py) which will give you the extra `0.3` F1.

## AMR Decoder in MTL

Users can also load a MTL component and specify `tasks=['lem', 'amr'])` to parse a sentence into an AMR graph. E.g.,

```python
import elit
nlp = elit.load('LEM_POS_NER_DEP_SDP_CON_AMR_ROBERTA_BASE_EN')
print(nlp(['Emory','NLP','is','in','Atlanta'], tasks=['lem', 'amr']))
```

Outputs:

```python
{
  "tok": [
    ["Emory", "NLP", "is", "in", "Atlanta"]
  ],
"lem": [
    ["emory", "nlp", "be", "in", "atlanta"]
  ],
  "amr": [
    [["c0", "ARG1", "c1"], ["c0", "ARG2", "c2"], ["c0", "instance", "be-located-at-91"], ["c1", "instance", "emory nlp"], ["c2", "instance", "atlanta"]]
  ],
}
```

`amr` stores the logical triples of Abstract Meaning Representation in the format of `(source, relation, target)`. Note that the `Document` class will convert it to Penman format when being accessed through code. Thus, you will get Penman `Graph`s by accessing `doc['amr']`.


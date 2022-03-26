# Dependency Parsing

Constituency parsing is the task of parsing a sentence to its dependency grammar which represents its syntax in the form of head/dependent word pairs. 

Users can load a MTL component and specify `tasks='dep'` to have sentences parsed by ELIT. E.g.,

```python
import elit
nlp = elit.load('LEM_POS_NER_DEP_SDP_CON_AMR_ROBERTA_BASE_EN')
print(nlp(['Emory','NLP','is','in','Atlanta']
, tasks='dep'))
```

Outputs:

```python
{
  "tok": [
    ["Emory", "NLP", "is", "in", "Atlanta"]
  ],
  "dep": [
    [[1, "com"], [3, "nsbj"], [3, "cop"], [-1, "root"], [3, "obj"]]
  ]
}
```

`dep` stores the `(head, relation)` of each token, with the offset starting from `-1` (ROOT). In this component, the primary dependency of Deep Dependency Graph Representation is used. The full representation with secondary dependencies are provided in `LEM_POS_NER_DEP_ROBERTA_BASE_EN`.

### Korean Model

We have also released a single task dependency parsing model for Korea trained on the Korean Treebank, which are demonstrated using the following codes.

```python
import elit
parser = elit.load('KOREAN_TREEBANK_BIAFFINE_DEP')
tree = parser('그는 르노가 3 월말까지 인수제의 시한을 갖고 있다고 덧붙였다 .'.split())
print(tree)
```

The input is a tokenized sentence or a list of multiple tokenized sentences. The type of the output is [`CoNLLSentence` ](https://github.com/emorynlp/elit/blob/main/elit/components/parsers/conll.py#L179) which is a list of [`CoNLLUWord`](https://github.com/emorynlp/elit/blob/main/elit/components/parsers/conll.py#L95). When the output is fed into `print` or `str`, a CoNLLU format will be returned:

```
1	그는	_	_	_	_	9	nsubj	_	_
2	르노가	_	_	_	_	7	csubj	_	_
3	3	_	_	_	_	4	nummod	_	_
4	월말까지	_	_	_	_	7	obl	_	_
5	인수제의	_	_	_	_	6	compound	_	_
6	시한을	_	_	_	_	7	obj	_	_
7	갖고	_	_	_	_	9	advcl	_	_
8	있다고	_	_	_	_	7	aux	_	_
9	덧붙였다	_	_	_	_	0	root	_	_
10	.	_	_	_	_	9	punct	_	_
```


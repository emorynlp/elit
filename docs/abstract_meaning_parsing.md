# Abstract Meaning Parsing

Abstract Meaning Representation parsing is the task of parsing a sentence to a representation of the abstract meaning in a sentence as a rooted, directed, acyclic graph.

Users can load a MTL component and specify `tasks=['lem', 'amr'])` to have sentences parsed by ELIT. E.g.,

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

`amr` stores the logical triples of Abstract Meaning Representation in the format of `(source, relation, target)`. Note that the `Document` class will convert it to Penman format when being accessed through code.


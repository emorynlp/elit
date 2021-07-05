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


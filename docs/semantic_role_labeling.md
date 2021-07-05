# Semantic Role Labeling

Semantic role labeling parses the predicate-argument structure of a sentence often answers “who did what to whom”. Currently, ELIT supports span-based SRL.

Users can load a MTL component and specify `tasks='srl'` to have sentences parsed by ELIT. E.g.,

```python
import elit
nlp = elit.load('LEM_POS_NER_DEP_SDP_CON_AMR_ROBERTA_BASE_EN')
print(nlp([['Emory','NLP','is','in','Atlanta']], tasks='srl'))
```

Outputs:

```python
{
  "tok": [
    ["Emory", "NLP", "is", "in", "Atlanta"]
  ],
  "srl": [
    [[["ARG1", 0, 2, "Emory NLP"], ["PRED", 2, 3, "is"], ["ARG2", 3, 5, "in Atlanta"]]]
  ],
}
```

`srl` stores the `(role, start, end, form)` of the predicates and arguments corresponding to each flattened predicate-argument structure. In this component, the OntoNotes 5 SRL annotations are used with an additional role `PRED` indicating the predicate.


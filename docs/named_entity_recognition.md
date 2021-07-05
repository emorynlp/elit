# Named Entity Recognition

Named Entity Recognition (NER) is the task of recognizing the span and type of each entity in a sentence. It is also a built-in decoder of the MTL framework. Users can load a MTL component and specify `tasks='ner'` to have the NER annotated by ELIT. E.g.,

```python
import elit
nlp = elit.load('LEM_POS_NER_DEP_ROBERTA_BASE_EN')
print(nlp([['Emory', 'NLP', 'is', 'a', 'research', 'lab', 'in', 'Atlanta', ',', 'GA', '.'], 
           ['It', 'is', 'founded', 'by', 'Jinho', 'D.', 'Choi', 'in', '2014', '.'], 
           ['Dr.', 'Choi', 'is', 'a', 'professor', 'at', 'Emory', 'University', '.']]
, tasks='ner'))
```

Outputs:

```python
{
  "ner": [
    [["ORG", 0, 2, "Emory NLP"], ["GPE", 7, 8, "Atlanta"], ["GPE", 9, 10, "GA"]],
    [["PERSON", 4, 7, "Jinho D. Choi"], ["DATE", 8, 9, "2014"]],
    [["PERSON", 1, 2, "Choi"], ["ORG", 6, 8, "Emory University"]]
  ],
  "tok": [
    ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
    ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", "."],
    ["Dr.", "Choi", "is", "a", "professor", "at", "Emory", "University", "."]
  ]
}
```

In `ner` field, each entity is represented as a tupe of `(type, start, end, form)`.


# Named Entity Recognition

Named Entity Recognition (NER) is also a built-in decoder of the MTL framework. Users can load a MTL component and specify `tasks='ner'` to have the NER annotated by ELIT. E.g.,

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
    [["Emory NLP", "ORG", 0, 2], ["Atlanta", "GPE", 7, 8], ["GA", "GPE", 9, 10]],
    [["Jinho D. Choi", "PERSON", 4, 7], ["2014", "DATE", 8, 9]],
    [["Choi", "PERSON", 1, 2], ["Emory University", "ORG", 6, 8]]
  ],
  "tok": [
    ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
    ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", "."],
    ["Dr.", "Choi", "is", "a", "professor", "at", "Emory", "University", "."]
  ]
}
```

In `ner` field, each entity is represented 
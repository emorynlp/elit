# Tokenization

ELIT features with a rule-based English tokenizer which offers both tokenization and sentence segmentation.

### Tokenization

The `EnglishTokenizer` class handles common abbreviations, apostrophes, concatenation words, hyphens, network protocols, emojis, emails, html entities, list item units with expert-crafted rules. E.g.:

```python
from elit.components.tokenizer import EnglishTokenizer
tokenizer = EnglishTokenizer()
text = "Emory NLP is a research lab in Atlanta, GA. It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
print(tokenizer.tokenize(text))
```

Output:

```python
['Emory', 'NLP', 'is', 'a', 'research', 'lab', 'in', 'Atlanta', ',', 'GA', '.', 'It', 'is', 'founded', 'by', 'Jinho', 'D.', 'Choi', 'in', '2014', '.', 'Dr.', 'Choi', 'is', 'a', 'professor', 'at', 'Emory', 'University', '.']
```

### Sentence Segmentation

The tokenized tokens can be fed into `tokenizer.segment` for sentence segmentation. E.g.:

```python
from elit.components.tokenizer import EnglishTokenizer
tokenizer = EnglishTokenizer()
print(tokenizer.segment(
  ['Emory', 'NLP', 'is', 'a', 'research', 'lab', 'in', 'Atlanta', ',', 'GA', '.', 'It', 'is', 'founded', 'by',
   'Jinho', 'D.', 'Choi', 'in', '2014', '.', 'Dr.', 'Choi', 'is', 'a', 'professor', 'at', 'Emory',
   'University', '.']))
```

Output:

```python
[['Emory', 'NLP', 'is', 'a', 'research', 'lab', 'in', 'Atlanta', ',', 'GA', '.'], ['It', 'is', 'founded', 'by', 'Jinho', 'D.', 'Choi', 'in', '2014', '.'], ['Dr.', 'Choi', 'is', 'a', 'professor', 'at', 'Emory', 'University', '.']]
```


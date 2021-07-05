# Constituency Parsing

Constituency parsing is the task of parsing a sentence to its [phrase structure grammar](https://en.wikipedia.org/wiki/Phrase_structure_grammar) which represents its syntax in the form of nested constituent structure. 

Users can load a MTL component and specify `tasks='con'` to have sentences parsed by ELIT. E.g.,

```python
import elit
nlp = elit.load('LEM_POS_NER_DEP_SDP_CON_AMR_ROBERTA_BASE_EN')
print(nlp([['Emory', 'NLP', 'is', 'a', 'research', 'lab', 'in', 'Atlanta', ',', 'GA', '.'], 
           ['It', 'is', 'founded', 'by', 'Jinho', 'D.', 'Choi', 'in', '2014', '.'], 
           ['Dr.', 'Choi', 'is', 'a', 'professor', 'at', 'Emory', 'University', '.']]
, tasks='con'))
```

Outputs:

```python
{
  "con": [
    ["TOP", [["S", [["NP", [["_", ["Emory"]], ["_", ["NLP"]]]], ["VP", [["_", ["is"]], ["NP", [["NP", [["_", ["a"]], ["_", ["research"]], ["_", ["lab"]]]], ["PP", [["_", ["in"]], ["NP", [["NP", [["_", ["Atlanta"]]]], ["_", [","]], ["NP", [["_", ["GA"]]]]]]]]]]]], ["_", ["."]]]]]],
    ["TOP", [["S", [["NP", [["_", ["It"]]]], ["VP", [["_", ["is"]], ["VP", [["_", ["founded"]], ["PP", [["_", ["by"]], ["NP", [["_", ["Jinho"]], ["_", ["D."]], ["_", ["Choi"]]]]]], ["PP", [["_", ["in"]], ["NP", [["_", ["2014"]]]]]]]]]], ["_", ["."]]]]]],
    ["TOP", [["S", [["NP", [["_", ["Dr."]], ["_", ["Choi"]]]], ["VP", [["_", ["is"]], ["NP", [["NP", [["_", ["a"]], ["_", ["professor"]]]], ["PP", [["_", ["at"]], ["NP", [["_", ["Emory"]], ["_", ["University"]]]]]]]]]], ["_", ["."]]]]]]
  ],
  "tok": [
    ["Emory", "NLP", "is", "a", "research", "lab", "in", "Atlanta", ",", "GA", "."],
    ["It", "is", "founded", "by", "Jinho", "D.", "Choi", "in", "2014", "."],
    ["Dr.", "Choi", "is", "a", "professor", "at", "Emory", "University", "."]
  ]
}
```

`con` stores the constituency trees, specifically `(label, child-constituents)` for the non-terminal constituents and `form` for the terminals. Note that we designed a nested list representation with the round brackets replaced by square brackets to avoid ambiguity and make it compatible with JSON. When not being printed out, the `Document` class will convert our nested list structure to the conventional bracketed tree.

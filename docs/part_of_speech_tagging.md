# Part-of-Speech Tagging

## API

The part-of-speech (POS) tagger is built into the Multi-Task Learning framework of ELIT. To use the POS tagger, a MTL component is required. One can list the pre-trained MTL components via the following code snippets.

```python
import elit
print(elit.pretrained.mtl.ALL)
```

Output:

```python
{
  'LEM_POS_NER_DEP_SDP_CON_AMR_ELECTRA_BASE_EN': 'https://elit-models.s3-us-west-2.amazonaws.com/v2/en_pos_ner_srl_dep_con_amr_electra_base_20201222.zip',
  'LEM_POS_NER_DEP_SDP_CON_AMR_ROBERTA_BASE_EN': 'https://elit-models.s3-us-west-2.amazonaws.com/v2/en_pos_ner_srl_dep_con_amr_roberta_base_20210402_152521.zip',
  'LEM_POS_NER_DEP_ROBERTA_BASE_EN': 'https://elit-models.s3-us-west-2.amazonaws.com/v2/en_lem_pos_ner_ddr_roberta_base_20210325_121606.zip'
}
```

Each of them is a pair of an identifier and the URL to it. The identifier indicates the task it supports and the backbone transformer it was trained on. If you are only interested in POS, it's recommended to use `LEM_POS_NER_DEP_ROBERTA_BASE_EN` as it involves less tasks so its size is smaller. 

Once you've decided which model to use, you can pass its identifier to `elit.load`. This method will download the model and load it to the least occupied GPU if any. E.g.,

```python
nlp = elit.load('LEM_POS_NER_DEP_ROBERTA_BASE_EN')
print(nlp('I banked 1 dollar in a bank .'.split(), tasks='pos'))
```

Output:

```python
{'pos': ['PRP', 'VBD', 'CD', 'NN', 'IN', 'DT', 'NN', '.'], 
 'tok': ['I', 'banked', '1', 'dollar', 'in', 'a', 'bank', '.']}
```

The output is a dict mapping the task names to corresponding annotations. It always contains the tokens so that each tag can be easily aligned with its actual token.

### Batched Prediction

Note that batched prediction is usually faster than sequential prediction even on CPUs. So it's always recommended to gather the input sentences into one list and make only 1 function call. E.g.,

```python
nlp([["The", "first", "sentence", "."],
     ["The", "second", "sentence", "."]])
```

is faster than call `nlp` twice separately:

```python
nlp(["The", "first", "sentence", "."])
nlp(["The", "second", "sentence", "."])
```

## Annotation Guideline

Please refer to [Part-of-Speech Tagging Guidelines for the Penn Treebank](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1603&context=cis_reports).

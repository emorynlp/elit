# Data Format for Online Coreference Resolution (OCR)

The main difference between document coreference resolution (`dcr`) and online coreference resolution (`ocr`) is that `ocr` requires context information for dialogue history and previously extracted mentions, while `dcr` is stateless and does not require any context.

## Input Format (1st Turn)

For the first utterance in a dialogue, there is no context available; therefore, the input for `ocr` is similar to `dcr`:

```json
{
    "tokens": [["I", "read", "an", "article", "today", "."]],
    "models": ["ocr"],
    "speaker_ids": 1,
    "language": "en",
    "verbose": false
}
```

## Output Format

* The cluster prediction is the `clusters` field under `ocr` in the output. Different from other models, the output tokens are indexed by the global token offset without sentence indication, because in `ocr` we also need to refer to past mentions in the context, and the raw context sentences are not part of the input anymore; only the latest utterance is in each input.
* Different from `dcr`, the clusters can include singletons since they could become coreferent mentions in the future.
* The following fields capture the current **state** of the dialogue, and they are intended to be consumed by the model in the next round prediction to quickly pick up the context. They are not directly human-readable because we don't want to re-tensorize the readable raw context each time. From the user perspective, the user just needs to include these fields in the next input as an encapsulated context without worrying about their content. 
    * `input_ids`, `sentence_map`, `subtoken_map`, `mentions`, `uttr_start_idx`, `speaker_ids`
* Depending on the model setting, the speaker IDs can be directly encoded in the `input_ids`; in that case, `speaker_ids` itself will be empty.
* `verbose` will always be set to `false` in `ocr`, because the model does not have access to the actual text of previous mentions in the context.

```json
{
    "tokens": [["I", "read", "an", "article", "today", "."]],
    "ocr": {
        "clusters": [
            [[0, 1]], 
            [[2, 4]]
        ], 
        "input_ids": [101, 102, 28997, 146, 2373, 1126, 3342, 2052, 119, 102], 
        "sentence_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        "subtoken_map": [0, 0, 0, 0, 1, 2, 3, 4, 5, 5], 
        "mentions": [[3, 3], [5, 6]], 
        "uttr_start_idx": [2], 
        "speaker_ids": [], 
        "linking_prob": null, 
        "error_msg": null
    }
}
```

## Input Format (1+ Turn)

* The prediction for the current utterance (after 1st turn) will always need to include the context in the `coref_context` field.
    * User just need to include the same `input_ids`, `sentence_map`, `subtoken_map`, `mentions`, `uttr_start_idx`, `speaker_ids` from previous output, providing a snapshot of the previous dialogue state for the model.

```json
{
    "tokens": [["Can", "you", "tell", "me", "what", "it", "is", "about", "?"]],
    "models": ["ocr"],
    "speaker_ids": 2,
    "coref_context": {
        "input_ids": [101, 102, 28997, 146, 2373, 1126, 3342, 2052, 119, 102],
        "sentence_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        "subtoken_map": [0, 0, 0, 0, 1, 2, 3, 4, 5, 5], 
        "mentions": [[3, 3], [5, 6]],
        "uttr_start_idx": [2], 
        "speaker_ids": []
    }
    "language": "en",
    "verbose": false
}
```

The output would be similar to:

```json
{
    "tokens": [["Can", "you", "tell", "me", "what", "it", "is", "about", "?"]], 
    "ocr": {
        "clusters": [
            [[0, 1], [7, 8]], [[9, 10]], 
            [[2, 4], [11, 12]]
        ], 
        "input_ids": [101, 28997, 146, 2373, 1126, 3342, 2052, 119, 102, 28998, 2825, 1128, 1587, 1143, 1184, 1122, 1110, 1164, 136, 102], 
        "sentence_map": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 
        "subtoken_map": [0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14], 
        "mentions": [[2, 2], [4, 5], [11, 11], [13, 13], [15, 15]], 
        "uttr_start_idx": [1, 9], 
        "speaker_ids": [], 
        "linking_prob": null, 
        "error_msg": null
    }
}
```

* Note that all mentions in the clusters are indexed by the global token offset. In this case, the global tokens would be: `["I", "read", "an", "article", "today", ".", "Can", "you", "tell", "me", "what", "it", "is", "about", "?"]`

Data pipeline simplifications:
1. Separate out the types of tasks and keep their pipelines separate.
  1. Pre-training (map-style)
  2. Pre-training (iterable-style)
  3. Left-conditional generation (seq2seq)
  4. Arbirary infilling generation (infill)

2. Do not add any special tokens while tokenizing. 
    * Add them during collation:
        1. ILM pretraining, 
    * We could have added them during on_the_fly_processing but that would not be "on the fly" for map-style datasets. One issue with this is that we will not be able to create packed sequences because they require special tokens between the sequences. For the case of packed sequences, we will have to use on_the_fly_processing to add special tokens.

|when| model | task |
|---|---|---|
| during collation | ILM | pre-training |
| during collation | ILM | left-conditional generation training |
| during collation | ILM | left-conditional generation inference |
| during collation | ILM | arbitrary infilling generation training|
| during collation | ILM | arbitrary infilling generation inference |
| during collation | ARLM | unpacked pre-training |
| during on-the-fly| ARLM | packed pre-training | --> Dump a map-style dataset of packed sequences with special tokens or use iterable dataset. In either case.
| during collation | ARLM | left-conditional generation training |
| during collation| ARLM | left-conditional generation inference |
|during collation | MDLM | unpacked pre-training |
|during on-the-fly | MDLM | packed pre-training |
|during on-the-fly | IT | pre-training |


# Special tokens
For ILM we will need three special tokens:
CLS for classification, BOS for target starting. Placing the BOS after prefix signals the model that prefix is immutable. This is something we never do in pre-training. Perhaps we should consider doing this in pre-training and make pre-training also a seq2seq task. Which makes me think that we should have another special token say `[FIXED]` to signal to the model that some parts of the input are immutable and we can have these parts anywhere in the sequence. We can use the SEP token for this.

For a general model, we will need to support seq2seq case.

* Case 1: Should we place the CLS before the prefix?
  * (Con) This will break caching if we ever were to cache. But we can't cache right now anyway so we will ignore this case for now.
  * (Con) In seq2seq setting, because of the left-padding the positin of the CLS token will get staggered and we'll have to keep track of its exact position in a separate tensor.
  * (Pro) The position of the CLS token is fixed, which will be a good thing.

* Case 2: Place the CLS before the target.
  * (Con) The position is not fixed. We don't know how important that is.
  * (Con) We will need to modify the model to read-out this position dynamically.
  * (Pro) Can be generalized to per-gap CLS token.

For now, we will place the CLS token before the prefix.

Ex for case 1:
```
[CLS][SEP]p1 p2 ... pN [SEP] [BOS] t1 t2 ... tM [SEP] p1 p2 ... pO [SEP] [BOS] t1 t2 ....
```

Ex for case 2:
```
[SEP]p1 p2 ... pN [SEP] [CLS] [BOS] t1 t2 ... tM [SEP] p1 p2 ... pO [SEP] [CLS] [BOS] t1 t2 ....
```










Handling packed sequences:
1. (Option 1) 
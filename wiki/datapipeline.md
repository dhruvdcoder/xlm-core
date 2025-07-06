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


# General Flow

Generally, for each stage, train, val, test, predict, we can have multiple datasets. Therefore, we create a dataset level abstraction called `DatasetManager` that performs the following:

1. Prepare
 - prepare data on rank 0 only
 - things like download and tokenize.

2. Setup
 - Performed on all ranks.
 - loads the data
 - If using an iterable dataset, it tells the dataset about the rank and world size. For example, for huggingface iterable dataset, we will need to call `.split_dataset_by_node()` to split the dataset across nodes.


3. Create dataloaders
  - We create one dataloader per dataset.
  - Depending on the case: DDP/non-DDP | map-style/iterable-style | setup the right dataloader and sampler.


## Case 1: DDP | Iterable Dataset

1. Prepare
  - prepare data on rank 0 and save it to the disk using `.save_to_disk(num_shards)`

2. Setup
  - `load_from_disk()` followed by `.to_iterable_dataset(num_shards)`
  - `.shuffle(buffer_size)`
  - `.split_dataset_by_node()`

3. Create dataloaders
  - Use `StatefulDataloader` without a sampler.

### What happens internally?

- `.to_iterable_dataset(num_shards)` splits the dataset into `num_shards`. It does not write to disk. Simply creates logical shards which is useful for splitting the dataset across nodes as well as for shuffling. For example, if we have 1000 examples and 64 shards, then 64\*15 = 960, therefore each shard will have at least 15 examples. The first 40 shards will have 16 examples and the last 24 shards will have 15 examples. Now let's say we perform DDP training with 4 GPUs and 4 dataloader workers per GPU, first the dataset shards will be assigned to the GPUs:
 - GPU 0: shards 1-16 : 16\*16          = 256 examples
 - GPU 1: shards 17-32 : 16\*16         = 256 examples
 - GPU 2: shards 33-48 : 8\*16 + 8\*15  = 248 examples (8 shards with 16 examples and 8 shards with 15 examples)
 - GPU 3: shards 49-64 : 16\*15         = 240 examples (16 shards with 15 examples)
                                          ------------
                                          Total: 1000 examples

- Once the shards are assigned to the nodes/GPUs, then the examples will be assigned to the dataloader workers. For iterable datasets, pytorch let's the dataset decide how to slice the examples. According to [pytorch docs](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), the iterable dataset is responsible for slicing the dataset by checking the `torch.utils.data.get_worker_info().num_workers` and `torch.utils.data.get_worker_info().id`. HF dataset assigns entire shards to each worker. This is for efficiency reasons. Now each worker can read chunks of contiguous examples from the shard. But when unevenly sized shards exist, each worker can produce partial batch at the end of the epoch. So there can be upto num_workers partial batches on each node. In the worst case, all the workers on one node produce partial batches on one node and no partial batches on the other node. Making the difference in the number of batches seen per node `num_workers`. Even if the num of batches seen per node is different  by 1, the training hangs. Setting `drop_last=True` will also not help here. Following are some minor points regarding how the shards are assigned to the workers:
  - The shards are assigned to workers in round robin fashion.
  - If num_shards is not divisible by num_workers, due to the round robin assignment, the initial workers will get more shards than the later workers. 


### Best practices
- Reduce the number of uneven shards. The goal is to make sure that no worker can produce a full micro batch using the extra single examples of the shards assigned to it. So choose the num_shards such that the remainder (N % num_shards) < shards_per_worker < micro_batch_size. The second condition ensures that no worker can produce a full micro batch using the extra examples of the shards assigned to it. The first condition ensures that the workers belonging to one node cannot gang up to produce too many partial batches.
If we set `drop_last=True` all the partial batches from all the workers will be dropped. So in this case, only the second condition is important. This condition was violated when I used `num_shards=1024` `num_nodes=4` `num_workers=5` `micro_batch_size=32` because each worker got 51.2 shards.

 - shards_per_worker = num_shards / (num_nodes * num_workers). So it is best to reduce the number of shards to a reasonable value.

Even the above did not work. But the following super conservative condition worked:
 - Make sure that no node (not workers) can accumulate more than 1 micro batch worth of extra examples.






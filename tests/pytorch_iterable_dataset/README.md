Issue: It is unclear how pytorch dataloader splits iterable dataset amongst workers, i.e., whether contiguous chunks of batch size are assigned to each worker or the iterable is sliced to perform round robin.

Observations:

- Another interesting observation is that somehow the hf iterable dataset is getting to know how many workers the dataloader is using! Now that I think more about it, I think this is the expected behaviour for an iterable dataset. According to [pytorch docs](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), the iterable dataset is responsible for slicing the dataset by checking the `torch.utils.data.get_worker_info().num_workers` and `torch.utils.data.get_worker_info().id`. HF dataset assigns entire shards to each worker. This is for efficiency reasons. Now each worker can read chunks of contiguous examples from the shard. But when unevenly sized shards exist, each worker can produce partial batch at the end of the epoch. So there can be upto num_workers partial batches on each node. In the worst case, all the workers on one node produce partial batches on one node and no partial batches on the other node. Making the difference in the number of batches seen per node `num_workers`. Even if the num of batches seen per node is different  by 1, the training hangs. Setting `drop_last=True` will also not help here.


# Runs

```
DS_SIZE = 60
BATCH_SIZE = 2
NUM_WORKERS = 2
NUM_SHARDS = 8
SEED = 42
```

Console output:
```
Shard 0: [0, 1, 2, 3, 4, 5, 6, 7]        length: 8
Shard 1: [8, 9, 10, 11, 12, 13, 14, 15]  length: 8
Shard 2: [16, 17, 18, 19, 20, 21, 22, 23]        length: 8
Shard 3: [24, 25, 26, 27, 28, 29, 30, 31]        length: 8
Shard 4: [32, 33, 34, 35, 36, 37, 38]    length: 7
Shard 5: [39, 40, 41, 42, 43, 44, 45]    length: 7
Shard 6: [46, 47, 48, 49, 50, 51, 52]    length: 7
Shard 7: [53, 54, 55, 56, 57, 58, 59]    length: 7
dataloader worker#0, ': Starting to iterate over 4/8 shards.    <------- 4 shards assigned to worker 0
[worker 0] yielding 0
[worker 0] yielding 1
[worker 0] yielding 2
[worker 0] yielding 3
[worker 0] yielding 4
[worker 0] yielding 5
dataloader worker#1, ': Starting to iterate over 4/8 shards.    <------- 4 shards assigned to worker 1
[worker 1] yielding 8
[worker 1] yielding 9
[worker 1] yielding 10
[worker 1] yielding 11
[worker 0] yielding 6
[worker 0] yielding 7
[worker 1] yielding 12
[worker 1] yielding 13
[worker 1] yielding 14
[worker 1] yielding 15
[worker 0] yielding 16
[worker 0] yielding 17
[worker 0] yielding 18
[worker 0] yielding 19
[worker 1] yielding 24
[worker 1] yielding 25
[worker 1] yielding 26
[worker 0] yielding 20[worker 1] yielding 27

[worker 0] yielding 21
[worker 1] yielding 28
[worker 0] yielding 22
[worker 1] yielding 29[worker 0] yielding 23

[worker 1] yielding 30
[worker 1] yielding 31
[worker 0] yielding 32
[worker 0] yielding 33
[worker 0] yielding 34
[worker 0] yielding 35
[worker 1] yielding 39
[worker 1] yielding 40
[worker 1] yielding 41
[worker 1] yielding 42
[worker 0] yielding 36
[worker 0] yielding 37
[worker 1] yielding 43
[worker 0] yielding 38
[worker 1] yielding 44
[worker 0] yielding 46
[worker 1] yielding 45
[worker 0] yielding 47
[worker 0] yielding 48
[worker 1] yielding 53
[worker 1] yielding 54
[worker 1] yielding 55
[worker 0] yielding 49
[worker 0] yielding 50
[worker 1] yielding 56
[worker 1] yielding 57
[worker 0] yielding 51
[worker 0] yielding 52
[worker 1] yielding 58
[worker 1] yielding 59
dataloader worker#0, ': Finished iterating over 4/4 shards.
dataloader worker#1, ': Finished iterating over 4/4 shards.
```

Since we don't shuffle following will be the shard assignments:

```
Shard 0: [0, 1, 2, 3, 4, 5, 6, 7]        length: 8 <------- worker 0
Shard 1: [8, 9, 10, 11, 12, 13, 14, 15]  length: 8 <------- worker 1
Shard 2: [16, 17, 18, 19, 20, 21, 22, 23]        length: 8 <------- worker 0
Shard 3: [24, 25, 26, 27, 28, 29, 30, 31]        length: 8 <------- worker 1
Shard 4: [32, 33, 34, 35, 36, 37, 38]    length: 7 <------- worker 0
Shard 5: [39, 40, 41, 42, 43, 44, 45]    length: 7 <------- worker 1
Shard 6: [46, 47, 48, 49, 50, 51, 52]    length: 7 <------- worker 0
Shard 7: [53, 54, 55, 56, 57, 58, 59]    length: 7 <------- worker 1
```

Batches seen: 
```
Batch 0: [0, 1] <------- worker: 0, global_shard: 0, local_shard: 0
Batch 1: [8, 9] <------- worker: 1, global_shard: 1, local_shard: 0
Batch 2: [2, 3] <------- worker: 0, global_shard: 0, local_shard: 0
Batch 3: [10, 11] <------- worker: 1, global_shard: 1, local_shard: 0
Batch 4: [4, 5] 
Batch 5: [12, 13]
Batch 6: [6, 7]
Batch 7: [14, 15] 
Batch 8: [16, 17] <------- worker: 0, global_shard: 2, local_shard: 1
Batch 9: [24, 25] <------- worker: 1, global_shard: 3, local_shard: 1
Batch 10: [18, 19]
Batch 11: [26, 27]
Batch 12: [20, 21]
Batch 13: [28, 29]
Batch 14: [22, 23]
Batch 15: [30, 31] <--- larger shards end here on both workers.
Batch 16: [32, 33]
Batch 17: [39, 40]
Batch 18: [34, 35]
Batch 19: [41, 42]
Batch 20: [36, 37]
Batch 21: [43, 44]
Batch 22: [38, 46] <---- Global shards 4 and 6 shards got smoothly stitched here for worker 0
Batch 23: [45, 53] <--- Global shards 5 and 7 got smoothly stitched here for worker 1
Batch 24: [47, 48]
Batch 25: [54, 55]
Batch 26: [49, 50]
Batch 27: [56, 57]
Batch 28: [51, 52]
Batch 29: [58, 59]
```


# Observations

- The shards are assigned to workers in round robin fashion.
- If num_shards is not divisible by num_workers, due to the round robin assignment, the initial workers will get more shards than the later workers.
This script is to test IterableDataset pipeline with DDP+num_workers>0.


See `wiki/datapipeline.md` for the analysis.


# Observations
The nodes are getting the same number of examples `30` each in each epoch. But the number of batches each node sees varies amongst the nodes as well as between epochs on the same node.
Sometimes the last two batches have one example each and sometimes all batches have exactly 2 examples. When in the same epoch, the two nodes see different number of batches, the training does hang.

- Another interesting observation is that somehow the hf iterable dataset is getting to know how many workers the dataloader is using! Now that I think more about it, I think this is the expected behaviour for an iterable dataset. According to [pytorch docs](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), the iterable dataset is responsible for slicing the dataset by checking the `torch.utils.data.get_worker_info().num_workers` and `torch.utils.data.get_worker_info().id`. HF dataset assigns entire shards to each worker. This is for efficiency reasons. Now each worker can read chunks of contiguous examples from the shard. But when unevenly sized shards exist, each worker can produce partial batch at the end of the epoch. So there can be upto num_workers partial batches on each node. In the worst case, all the workers on one node produce partial batches on one node and no partial batches on the other node. Making the difference in the number of batches seen per node `num_workers`. Even if the num of batches seen per node is different  by 1, the training hangs. Setting `drop_last=True` will also not help here.

# Best practices
- Reduce the number of uneven shards. The goal is to make sure that no worker can produce a full micro batch using the extra single examples of the shards assigned to it. So choose the num_shards such that the remainder (N % num_shards) < shards_per_worker < micro_batch_size. The second condition ensures that no worker can produce a full micro batch using the extra examples of the shards assigned to it. The first condition ensures that the workers belonging to one node cannot gang up to produce too many partial batches.
If we set `drop_last=True` all the partial batches from all the workers will be dropped. So in this case, only the second condition is important. This condition was violated when I used `num_shards=1024` `num_nodes=4` `num_workers=5` `micro_batch_size=32` because each worker got 51.2 shards.

shards_per_worker = num_shards / (num_nodes * num_workers)


# Edge cases

- What happens when shards_per_node is not divisible by num_workers?
  - In this situation HF iterable dataset assigns more shards to some workers in round robin fashion.
- What happends when I have different number of shards on the disk when compared to the input to `.to_iterable_dataset(num_shards)`?
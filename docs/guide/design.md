# Design

This document describes the data pipeline and design decisions in XLM.

## Data Pipeline Simplifications

1. **Separate task types** – Keep their pipelines separate:
   - Pre-training (map-style)
   - Pre-training (iterable-style)
   - Left-conditional generation (seq2seq)
   - Arbitrary infilling generation (infill)

2. **Special tokens** – Do not add special tokens while tokenizing. Add them during collation (or on-the-fly for packed sequences).

| When | Model | Task |
|------|-------|------|
| during collation | ILM | pre-training |
| during collation | ILM | left-conditional generation training |
| during collation | ILM | left-conditional generation inference |
| during collation | ILM | arbitrary infilling generation training |
| during collation | ILM | arbitrary infilling generation inference |
| during collation | ARLM | unpacked pre-training |
| during on-the-fly | ARLM | packed pre-training |
| during collation | ARLM | left-conditional generation training |
| during collation | ARLM | left-conditional generation inference |
| during collation | MDLM | unpacked pre-training |
| during on-the-fly | MDLM | packed pre-training |
| during on-the-fly | IT | pre-training |

## Special Tokens

For ILM we need three special tokens: CLS for classification, BOS for target starting. Placing the BOS after prefix signals to the model that the prefix is immutable.

* **Case 1: Place CLS before the prefix**
  - (Pro) The position of the CLS token is fixed
  - (Con) In seq2seq setting, because of left-padding the position of the CLS token can get staggered

* **Case 2: Place CLS before the target**
  - (Pro) Can be generalized to per-gap CLS token
  - (Con) The position is not fixed; we need to modify the model to read-out this position dynamically

For now, we place the CLS token before the prefix.

## General Flow

For each stage (train, val, test, predict), we can have multiple datasets. The `DatasetManager` abstraction performs:

1. **Prepare** – On rank 0 only: download and tokenize
2. **Setup** – On all ranks: load data; for iterable datasets, split by rank/world size
3. **Create dataloaders** – One per dataset, with appropriate sampler for DDP/non-DDP and map-style/iterable-style

## DDP with Iterable Dataset

1. **Prepare** – On rank 0, save to disk using `.save_to_disk(num_shards)`
2. **Setup** – `load_from_disk()` followed by `.to_iterable_dataset(num_shards)`, `.shuffle(buffer_size)`, `.split_dataset_by_node()`
3. **Create dataloaders** – Use `StatefulDataLoader` without a sampler

### Best Practices

- Reduce the number of uneven shards to avoid partial batches across workers
- Choose `num_shards` such that remainder conditions are satisfied
- Ensure no node can accumulate more than 1 micro batch worth of extra examples

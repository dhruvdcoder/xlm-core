# MLM — masked language model

## UniRef50 (protein, standard)

Train a from-scratch rotary Transformer MLM on `airkingbd/uniref50` with the ESM2
tokenizer vocabulary (`facebook/esm2_t30_150M_UR50D` — used only for token ids, not
weights).  Each sequence is independently truncated to `block_size` and padded to form
a fixed-length batch.

```bash
xlm job_type=train job_name=uniref50_mlm_run experiment=uniref50_mlm
```

Core pieces: `datamodule/uniref50` + dataset configs `uniref50_train` / `uniref50_val`,
and `experiment=uniref50_mlm`.

---

## UniRef50 (protein, packed — recommended for GPU efficiency)

Trains the same model but packs multiple protein sequences per block instead of padding.
Key differences from the standard variant:

| | Standard | Packed |
|---|---|---|
| Packing | one protein per slot, padded | multiple proteins per block, no padding |
| Cropping | first `block_size` tokens | random window of `block_size` (DPLM-style) |
| Attention | full 2-D mask | **block-diagonal** — each protein only attends to itself; 3-D boolean mask (default) or `BlockMask` via FlexAttention (`model.use_flex_attn=true`) |
| Positions | monotonic 0…`block_size-1` | **reset to 0** at the start of each protein |
| Collator | `DefaultMLMCollator` | `PackedMLMCollator` |

### How it works

1. `preprocess_fn` tokenises each `seq` string into `token_ids` (no truncation, full
   sequence cached).
2. `pack_sequences_fn` (used as `on_the_fly_group_processor`) randomly crops sequences
   longer than `block_size` to a random window of that length, then concatenates them
   with EOS separators and chunks into exactly-`block_size` blocks.
3. `PackedMLMCollator` receives a pre-packed block and computes:
   - a **3-D block-diagonal attention mask** `(batch, seq_len, seq_len)` by detecting
     segment boundaries at each EOS position,
   - **per-sequence reset positions** that restart at 0 after every EOS,
   - standard random MLM masking.
4. `MLMLoss.loss_fn` branches on `attention_mask.ndim`: when 3-D it uses the precomputed
   `positions` from the batch directly; when 2-D it falls back to the existing cumsum
   logic for the standard padded case.

### Training

Standard (3-D boolean mask, SDPA fallback):
```bash
xlm job_type=train job_name=uniref50_packed_mlm_run experiment=uniref50_packed_mlm
```

With FlexAttention (recommended — fused Triton kernel, no O(seq²) mask materialisation):
```bash
xlm job_type=train job_name=uniref50_packed_mlm_run experiment=uniref50_packed_mlm \
    model.use_flex_attn=true
```

### Debug / inspect sequence packing (`debug=overfit`, batch_size=2)

Use `per_device_batch_size=2` to see two packed blocks side-by-side, making it easy to
inspect that EOS separators land at the right places and that the block-diagonal mask
and reset positions are correct.

```bash
xlm job_type=train job_name=uniref50_packed_mlm_debug experiment=uniref50_packed_mlm \
    debug=overfit \
    global_batch_size=2 \
    per_device_batch_size=2 \
    num_dataloader_workers=2 \
    datamodule.dataset_managers.train.lm.dataloader_kwargs.drop_last=false
```

The `print_batch_fn` (`mlm.datamodule_mlm.print_batch_mlm`) will print the decoded
tokens and mask for the first example in each batch so you can verify proteins are
separated by `<eos>` and that each protein's residues appear as a contiguous block.

---

## OpenWebText (text, packed)

```bash
xlm job_type=train job_name=owt_packed_mlm_run experiment=owt_packed_mlm
```

Uses naive packing (full cross-block attention, monotonic positions) — appropriate for
text where cross-document leakage is tolerable.

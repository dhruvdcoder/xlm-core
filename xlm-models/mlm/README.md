# MLM — masked language model

## UniRef50 (protein)

Train a from-scratch rotary Transformer MLM on `airkingbd/uniref50` with the ESM2 tokenizer vocabulary (`facebook/esm2_t30_150M_UR50D` — used only for token ids, not weights):

```bash
xlm job_type=train job_name=uniref50_mlm_run experiment=uniref50_mlm
```

Core pieces: `datamodule/uniref50` + dataset configs `uniref50_train` / `uniref50_val`, and `experiment=uniref50_mlm` (mirrors `owt_mlm` scheduling defaults).

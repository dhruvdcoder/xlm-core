# iwslt14_joint_char_bpe_v1

Vendored into `xlm.tasks.iwslt14_de_en.tokenizer_v1` (default for Hydra group
`tokenizer/iwslt14_joint_char_bpe_v1`). Rebuild source of truth:
`iwslt14-rdm-prep/artifacts/iwslt14_joint_char_bpe_v1`.

Joint German–English character BPE trained on `iwslt14_de_en_rdm_text_v1` training text only
(all DE `source_text` lines, then all EN `target_text`).

- Class: Hugging Face `CharBPETokenizer` (whitespace-only pre-tokenization)
- Requested vocab size: 10152 (includes specials)
- Specials: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`, `<mask>=4`
- Load: `AutoTokenizer.from_pretrained(this_directory)`
- Post-load tokens (e.g. EILM `[DELETE]` / `[EXPAND_*]`) can be appended via
  `tokenizer.add_tokens(..., special_tokens=True)` without changing base IDs.

See `manifest.json` and `stats.json` for hashes and length/unk statistics.

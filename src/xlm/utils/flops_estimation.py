"""Generic FLOP estimates (Dream modeling helpers; see also gpt2_transformer MFU)."""


def attention_flops_per_token(
    n_layers: int, seq_len: int, dim: int, causal: bool
) -> float:
    # Formula from flash-attention benchmarks
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + int(
        attention_flops_per_token(n_layers, seq_len, dim, False)
    )

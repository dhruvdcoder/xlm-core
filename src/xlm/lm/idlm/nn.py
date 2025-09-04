"""Neural network utilities for IDLM model.

This file contains utility functions for IDLM computations.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT


def hyp1f1_1_nplus1_vec(
    x: Float[TT, " batch"], n: Integer[TT, " batch"], K: int = 500
) -> Float[TT, " batch"]:
    """
    Compute $S(n, x) = \sum_{k=0}^\infty \frac{n! x^k}{(n+k)!}$ using recurrence relation.
    $T_0 = 1$
    $T_{k+1} = T_k * \frac{x}{n+1+k}$
    $S(n, x) = \sum_{k=0}^K T_k$ for large K.

    Args:
        x: scalar tensor, n: (batch,) tensor
        n: (batch,) tensor
        K: int, number of terms to sum
    Returns:
        S: (batch,) tensor
    """
    # x: scalar tensor, n: (batch,) tensor
    device = x.device
    n = n.unsqueeze(1)  # shape (batch, 1)
    x = x.unsqueeze(1)  # shape (batch, 1)
    # n = n.to(torch.float64)
    # x = x.to(torch.float64)

    # create matrix of denominators of shape (*batch, K), where *batch is the leading dimensions of x
    # ks = torch.arange(K, dtype=torch.float64, device=device).unsqueeze(1)  # k=0..K-1
    ks = torch.arange(K, dtype=x.dtype, device=device).unsqueeze(
        0
    )  # k=0..K-1, shape (1, K)
    den = n + 1 + ks  # shape (batch, K)

    # factors = x / (n+1+k)
    factors = x / den  # shape (batch, K)

    # compute cumulative product along k to get T_k/T_0
    cumfac = torch.cumprod(factors, dim=-1)  # shape (batch, K)

    # prepend T_0=1 to align
    T = torch.cat([torch.ones_like(n), cumfac], dim=-1)  # (batch, K+1)

    # sum over k
    return T.sum(dim=-1)  # shape (batch,)


def incomplete_gamma_factor_using_series(
    n: Integer[TT, " batch"], dot_sigma: Float[TT, " batch"], K: int = 20
) -> Float[TT, " batch"]:
    """Compute incomplete gamma factor using series approximation.

    This function computes the incomplete gamma factor used in IDLM diffusion
    for correcting the first step prediction probability.

    Args:
        n: Number of drops per example in the batch
        dot_sigma: Total noise values
        K: Number of terms to use in the series approximation

    Returns:
        Incomplete gamma factor for each example in the batch
    """
    return hyp1f1_1_nplus1_vec(dot_sigma, n, K=K)


def remove_tokens(
    token_ids: Integer[TT, "batch seq_len"],
    ids_to_remove: Union[Integer[TT, " n"], int],
    pad_token_id: int,
) -> Integer[TT, "batch seq_len"]:
    """
    Remove all ids_to_remove (e.g. mask tokens) from token_ids
    and shift the non-mask tokens to fill the gap.
    The resulting tensor has the same shape as token_ids.
    The extra (empty) slots at the end of
    each row are filled with pad_token_id.

    Args:
        token_ids (torch.Tensor): Tensor of shape (batch, seq_len) containing token ids.
        ids_to_remove (int): The id of the mask token that should be removed or a tensor of shape (n,) containing the ids to remove.
        pad_token_id (int): The id to use for padding the empty positions.
        hold_mask (bool): For the positions where this is true, we will consider them as tokens even if they are in ids_to_remove.

    Returns:
        torch.Tensor: A tensor of the same shape as token_ids with ids_to_remove removed.
    """
    # Create a boolean mask where True indicates positions that are not mask tokens.
    if isinstance(ids_to_remove, int):
        non_mask = token_ids != ids_to_remove  # shape: (batch, seq_len)
    else:
        non_mask = torch.isin(
            token_ids, ids_to_remove, invert=True
        )  # shape: (batch, seq_len)

    output = _remove_tokens(token_ids, non_mask, pad_token_id)
    return output


def _remove_tokens(
    token_ids: Integer[TT, "batch seq_len"],
    non_mask: Bool[TT, "batch seq_len"],
    pad_token_id: int,
):
    # Compute the new positions for each non-mask token. For every row, this tells us where
    # (i.e. to which index) a valid token should be placed if we compress out the mask tokens.
    valid_indices = non_mask.cumsum(dim=1) - 1  # shape: (batch, seq_len)

    # Create an output tensor filled with pad_token_id.
    output = torch.full_like(token_ids, pad_token_id)

    # Create a batch index tensor for advanced indexing.
    batch_indices = (
        torch.arange(token_ids.size(0), device=token_ids.device)
        .unsqueeze(1)
        .expand_as(token_ids)
    )

    # For all positions where the token is not a mask, scatter them into the output
    # at the computed target indices.
    output[batch_indices[non_mask], valid_indices[non_mask]] = token_ids[
        non_mask
    ]

    return output


def log_softmax_last_two_dims(
    x: Float[TT, " *batch seq_len vocab_size"],
) -> Float[TT, " *batch seq_len vocab_size"]:
    # return x # DEBUG_SPARSE
    y = x - torch.amax(
        x, dim=(1, 2), keepdim=True
    )  # shape (*batch, seq_len, vocab_size)
    return y - torch.logsumexp(
        y, dim=(1, 2), keepdim=True
    )  # shape (*batch, seq_len, vocab_size)


def masked_ce_last_two_dims(
    logits: Float[TT, "batch seq vocab"],
    target: Float[TT, "batch seq vocab"],
    mask: Bool[TT, "batch seq vocab"],
    min_value: float,
    inplace: bool = False,
) -> Float[TT, "batch"]:
    """Computes cross entropy using `logits` and `target` probabilities.
    The `mask` entries of `target` are ignored by setting them of -infty (effectively).
    Ideally, pytorch should handle -infy in the logits values that represent 0 predicted probability,
    but it currenlty does not: https://github.com/pytorch/pytorch/issues/49844
    The `mask` entries are ignored by setting corresponding logits to min_value.
    Handles edge cases:
    - When all positions are masked: returns 0 loss
    - When targets are all zeros: returns 0 loss (no information to learn)

    Note: If `inplace` is True, the logits will not be usable after this call.

    Args:
        logits: Unnormalized logits of shape (*batch, seq, vocab).
        target: Target probabilities of shape (*batch, seq, vocab).
        mask: Mask of shape (*batch, seq, vocab). True means position is masked.
        min_value: The minimum value to use for the logits.
        inplace: If True, the logits will be modified in place.
    """
    # raise NotImplementedError(
    #    "TODO: Check the case when everything is masked before using."
    # )
    leading_dims = logits.size()[:-2]
    if inplace:
        logits.masked_fill_(mask, min_value)
    else:
        logits = logits.masked_fill(mask, min_value)

    return torch.nn.functional.cross_entropy(
        logits.reshape(*leading_dims, -1),
        target.reshape(*leading_dims, -1),
        reduction="none",
    )


def topk_over_last_two_dims(
    tensor: Float[TT, "* batch d1 d2"], k: int
) -> Tuple[Float[TT, "*batch k"], Float[TT, "*batch k 2"]]:
    """
    Compute top-k values and their indices over dimensions 1 and 2 of a 3D tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, dim1, dim2).
        k (int): Number of top elements to retrieve.

    Returns:
        torch.Tensor: Top-k values, shape (batch_size, k).
        torch.Tensor: Unraveled indices of top-k values, shape (batch_size, k, 2).
    """
    # Ensure the tensor is 3D
    assert tensor.dim() == 3, "Input tensor must be 3D"

    # Flatten dimensions 1 and 2
    batch_size, dim1, dim2 = tensor.size()
    flattened = tensor.view(batch_size, -1)  # Shape (batch_size, dim1 * dim2)

    # Apply topk along the flattened dimension
    top_values, top_indices = torch.topk(flattened, k, dim=1)

    # Unravel the flattened indices back to dimensions 1 and 2
    unraveled_indices = torch.stack(
        torch.unravel_index(top_indices, (dim1, dim2)), dim=-1
    )  # Shape (batch_size, k, 2)

    return top_values, unraveled_indices


def max_over_last_two_dims(
    x: Float[TT, "* batch d1 d2"],
) -> Tuple[
    Float[TT, "*batch"], Tuple[Integer[TT, "*batch"], Integer[TT, "*batch"]]
]:
    """
    Compute the maximum values and their indices over dimensions 1 and 2 of a 3D tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, dim1, dim2).

    Returns:
        torch.Tensor: Maximum values, shape (batch_size,).
        torch.Tensor: Unraveled indices of maximum values, shape (batch_size, 2).
    """
    # Ensure the tensor is 3D
    assert x.dim() == 3, "Input tensor must be 3D"

    # Flatten dimensions 1 and 2
    sizes = x.size()
    leading_dims = sizes[:-2]
    dim1, dim2 = sizes[-2:]
    flattened = x.view(*leading_dims, -1)  # Shape (*leading_dims, dim1 * dim2)

    # Find the maximum values and their indices along the flattened dimension
    max_values, max_indices = torch.max(flattened, dim=-1)  # shape (*batch)

    # Unravel the flattened indices back to dimensions 1 and 2
    index_1, index_2 = torch.unravel_index(
        max_indices, (dim1, dim2)
    )  # shape 2 * (*batch)

    return max_values, (index_1, index_2)


def sample_over_last_two_dims(
    logits: Float[TT, "*batch d1 d2"],
    sampling_function: Callable[
        [Float[TT, "*batch cat"]], Integer[TT, "*batch"]
    ],
) -> Tuple[Integer[TT, " *batch"], Integer[TT, " *batch"]]:
    """
    Sample values and their indices over dimensions 1 and 2 of a 3D tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, dim1, dim2).
            It can represent probabilities or unnormalized logits.

    Returns:
        Tuple[Integer[TT, "*batch"], Integer[TT, "*batch"]]:
            - Sampled values, shape (batch_size,).
            - Unraveled indices of sampled values, shape (2, batch_size).
    """
    # Ensure the tensor is 3D
    assert logits.dim() == 3, "Input tensor must be 3D"
    leading_dims = logits.size()[:-2]
    dim1, dim2 = logits.size()[-2:]
    flattened = logits.view(
        *leading_dims, -1
    )  # Shape (*leading_dims, dim1 * dim2)
    # Sample indices using the Gumbel-Max trick via sample_categorical
    # sampled_indices = sample_from_logits(flattened)  # Shape (batch_size,)
    sampled_indices = sampling_function(flattened)  # Shape (batch_size,)

    # Gather sampled values
    # sampled_values = flattened.gather(1, sampled_indices.unsqueeze(1)).squeeze(
    #    1
    # )  # Shape (batch_size,)

    # Unravel the indices back to dimensions 1 and 2
    index_1, index_2 = torch.unravel_index(
        sampled_indices, (dim1, dim2)
    )  # Each shape (batch_size,)

    # Convert indices to float for consistency with type hints
    return index_1, index_2


def general_sample_over_last_two_dims(
    logits: Float[TT, "*batch seq vocab"],
    sampling_function: Callable[
        [Float[TT, "*batch cat"]], Integer[TT, "*batch"]
    ],
    second_sampling_function: Optional[
        Callable[[Float[TT, "*batch cat"]], Integer[TT, "*batch"]]
    ],
) -> Tuple[Integer[TT, " *batch"], Integer[TT, " *batch"]]:
    """

    Args:
        logits: Joint logits of shape (*batch, seq, vocab).
        sampling_function: If second_sampling_fuction is None, this will be used to jointly sample
            the sequence and vocabulary dimensions.
            If second_sampling_function is not None, this will be used to sample from the vocab dimension.
        second_sampling_function: If given, it will be use for the sequence dimension.
    Returns:
        sequence_indices, vocabulary_indices
    """
    if second_sampling_function is None:
        return sample_over_last_two_dims(logits, sampling_function)
    # Sequence
    seq_logits = torch.logsumexp(logits, dim=-1)  # shape (*batch, seq)
    sequence_indices = second_sampling_function(seq_logits)  # shape (*batch)
    # Vocabulary
    # Gather the logits for the selected sequence indices
    batch_indices = torch.arange(
        logits.size(0), device=logits.device
    )  # shape (batch,)
    vocab_logits = logits[
        batch_indices, sequence_indices, :
    ]  # shape (*batch, vocab)
    vocabulary_indices = sampling_function(vocab_logits)  # shape (*batch)
    return sequence_indices, vocabulary_indices


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 3
    vocab_size = 4
    dtype = torch.bfloat16
    device = torch.device("cuda")
    logits = torch.randn(batch_size, seq_len, vocab_size).to(
        dtype=dtype, device=device
    )
    target = torch.zeros(batch_size, seq_len, vocab_size).to(
        dtype=logits.dtype, device=device
    )
    all_mask = torch.ones(
        batch_size, seq_len, vocab_size, dtype=torch.bool, device=device
    )
    loss = masked_ce_last_two_dims(
        logits, target, all_mask, min_value=torch.finfo(logits.dtype).min
    )
    print(loss)

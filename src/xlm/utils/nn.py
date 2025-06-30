from contextlib import contextmanager
from typing import Any, List
import torch


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_mean(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
        It must be 1 for non-masked values and 0 for masked values.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    dtype = vector.dtype
    value_count = (
        torch.sum(mask, dim=dim, keepdim=keepdim).to(dtype=dtype) + 1e-9
    )
    # tiny_value_of_dtype(dtype)
    return value_sum / value_count


def masked_sum(
    vector: torch.Tensor,
    mask: torch.BoolTensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    To calculate sum along certain dimensions on masked values

    # Parameters

    vector : `torch.Tensor`
        The vector to calculate sum.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate sum
    keepdim : `bool`
        Whether to keep dimension

    # Returns

    `torch.Tensor`
        A `torch.Tensor` of including the sum values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    return torch.sum(replaced_vector, dim=dim, keepdim=keepdim)


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


dtype_map = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def dtype(string: str) -> torch.dtype:
    """
    Convert a string to a PyTorch data type.

    # Parameters

    string : `str`
        The string to convert.

    # Returns

    `torch.dtype`
        The PyTorch data type.
    """
    if string in dtype_map:
        return dtype_map[string]
    else:
        raise ValueError(f"Unknown dtype: {string}")


# Define a function to select the appropriate dtype
def get_autocast_dtype():
    if (
        torch.cuda.get_device_properties(0).major >= 8
    ):  # Ampere and later architectures
        return torch.bfloat16
    else:
        return torch.float16


@contextmanager
def no_grad(no_grad_: bool = True):
    if no_grad_:
        with torch.no_grad():
            yield
    else:
        yield


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """Need this since torch.multinomial does not accept 3D input and cannot handle unnormalized probabilities.

    So we implement the "exponential race method" manually which can handle any number of leading dimensions
    and can handle unnormalized probabilities (not logits, )

    Note: This is not differentiable.. Use gumbel softmax for it.
    Args:
        probs: (*batch, seq_len, vocab_size) can have any number of leading dimensions.
            Note: probs should be positive, can be unnormalized.
    Returns:
        (*batch, seq_len)
    """
    exp_1_samples = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / exp_1_samples).argmax(dim=-1)


def sample_from_logits(
    logits: torch.Tensor, temperature: float = 1.0, noise_scale: float = 1.0
) -> torch.Tensor:
    """
    Sample from logits using the Gumbel-Max trick. Similar to sample_categorical, but works with logits (real valued).
    Args:
        logits: (*batch, seq_len, vocab_size) can have any number of leading dimensions.
    Returns:
        (*batch, seq_len)
    """
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
    )
    perturbed_logits = (logits / temperature) + gumbel_noise * noise_scale
    return perturbed_logits.argmax(dim=-1)


def sample_from_top_k(k: int, logits: torch.Tensor) -> torch.Tensor:
    """
    Sample from the top-k logits using the Gumbel-Max trick.
    Args:
        logits: (*batch, seq_len, vocab_size) can have any number of leading dimensions.
        k: The number of top logits to consider for sampling.
    Returns:
        (*batch, seq_len)
    """
    # Get the top-k logits and their indices
    if logits.shape[-1] < k:
        k = logits.shape[-1]
    top_k_logits, top_k_indices = torch.topk(
        logits, k, dim=-1
    )  # (*batch, seq_len, k)

    # Add Gumbel noise to the top-k logits
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(top_k_logits) + 1e-10) + 1e-10
    )
    perturbed_top_k_logits = (
        top_k_logits + gumbel_noise
    )  # (*batch, seq_len, k)

    # Sample from the perturbed top-k logits
    sampled_indices = perturbed_top_k_logits.argmax(
        dim=-1
    )  # (*batch, seq_len)

    # Map back to the original indices
    sampled_indices_full = top_k_indices.gather(
        -1, sampled_indices.unsqueeze(-1)
    ).squeeze(-1)

    return sampled_indices_full  # (*batch, seq_len)


def sample_from_top_p(p: float, logits: torch.Tensor) -> torch.Tensor:
    """
    Sample from the top-p logits using the Gumbel-Max trick.

    Args:
        p (float): The cumulative probability threshold. Must be between 0 and 1.
        logits (torch.Tensor): A tensor of shape (*batch, seq_len, vocab_size) representing
                               the unnormalized log probabilities for each token.

    Returns:
        torch.Tensor: A tensor of shape (*batch, seq_len) containing the sampled token indices.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"Parameter p must be in (0, 1], got {p}")

    # Compute softmax probabilities
    probs = torch.softmax(
        logits, dim=-1
    )  # Shape: (*batch, seq_len, vocab_size)

    # Sort the probabilities and corresponding logits in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    sorted_logits = torch.gather(logits, dim=-1, index=sorted_indices)

    # Compute the cumulative sum of the sorted probabilities
    cumulative_probs = torch.cumsum(
        sorted_probs, dim=-1
    )  # Shape: (*batch, seq_len, vocab_size)

    # Create a mask for tokens to keep: cumulative_probs >= p
    sorted_indices_to_keep = cumulative_probs >= p

    # Ensure that at least one token is kept
    sorted_indices_to_keep[..., 0] = True

    # Initialize a mask for the original logits
    mask = torch.zeros_like(
        logits, dtype=torch.bool
    )  # Shape: (*batch, seq_len, vocab_size)

    # Scatter the sorted mask back to the original logits' positions
    mask.scatter_(-1, sorted_indices, sorted_indices_to_keep)

    # Mask out logits that are not in the top-p subset by setting them to -inf
    masked_logits = logits.masked_fill(~mask, float("-inf"))

    # Add Gumbel noise to the masked logits
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(masked_logits) + 1e-10) + 1e-10
    )
    perturbed_logits = masked_logits + gumbel_noise

    # Sample the token with the highest perturbed logit
    sampled_indices = perturbed_logits.argmax(
        dim=-1
    )  # Shape: (*batch, seq_len)

    return sampled_indices  # (*batch, seq_len)


def hyp1f1_1_nplus1_vec(x, n, K=500):
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


def pad_truncate_list(
    ids: List[Any],
    max_len: int,
    pad_token: Any,
    pad_left: bool = False,
) -> List[Any]:
    if not pad_left:
        return ids[:max_len] + [pad_token] * (max_len - len(ids))
    else:
        return [pad_token] * (max_len - len(ids)) + ids[
            -max_len:
        ]  # when padding left, truncate left side

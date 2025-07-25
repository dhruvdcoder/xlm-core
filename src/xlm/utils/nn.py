from contextlib import contextmanager
from typing import Any, List, Literal, Optional, Tuple, Union, overload
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


# overload for tensor for typing
@overload
def pad_truncate_list(
    ids: List[Any],
    max_len: int,
    pad_token: Any,
    pad_left: bool = False,
    return_num_padded: bool = False,
) -> List[Any]: ...


@overload
def pad_truncate_list(
    ids: List[Any],
    max_len: int,
    pad_token: Any,
    pad_left: bool = False,
    return_num_padded: bool = True,
) -> Tuple[List[Any], int]: ...


def pad_truncate_list(
    ids: List[Any],
    max_len: int,
    pad_token: Any,
    pad_left: bool = False,
    return_num_padded: bool = False,
) -> Union[List[Any], Tuple[List[Any], int]]:
    num_padded = max_len - len(ids)
    if not pad_left:
        padded = ids[:max_len] + [pad_token] * num_padded
    else:
        padded = [pad_token] * num_padded + ids[
            -max_len:
        ]  # when padding left, truncate left side
    if return_num_padded:
        return padded, num_padded
    else:
        return padded


def add_gumbel_noise(
    logits: torch.Tensor, temperature: float = 1.0, noise_scale: float = 1.0
) -> torch.Tensor:
    """
    Add gumbel noise to logits which will result in samples from the distribtution if argmaxed.
    Args:
        logits: (*batch, seq_len, vocab_size) can have any number of leading dimensions. Assumed to be log of exponentiated scores.
            That is, we assume logits are $l_i$ in $p_i = exp(l_i) / \sum_i \exp(l_i)$.
    Returns:
        (*batch, seq_len)
    """
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
    )
    perturbed_logits = (logits / temperature) + gumbel_noise * noise_scale
    return perturbed_logits


def add_exp_1_noise(
    probs: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from unnormalized probability using the exponential race method.
    Similar to using gumbel noise, trick but we require the probs to be positive (can be unnormalized).
    You can generate samples without repeatations from the output by taking argmax or topk, etc.
    Args:
        probs: (*batch, seq_len, vocab_size) can have any number of leading dimensions.
    Returns:
        (*batch, seq_len)
    """
    exp_1_samples = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    perturbed = probs / exp_1_samples
    return perturbed


def select_random_indices(
    inp_shape: torch.Size,
    num_unmask: torch.Tensor,
    select_from_mask: Optional[torch.Tensor] = None,
    selection_score: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    selection_mode: Literal["greedy", "sample"] = "greedy",
    score_mode: Literal["logits", "uprobs"] = "logits",
) -> torch.Tensor:
    """
    Select random indices from the last dimension using the selection_score.
    1. If selection score is None then it is assumed to be uniform.
    2. If score_mode = logits and selection_mode=sample, then temperature can be used to control the temperature of the distribution.
    3. If select_from_mask is provided, indices only from these positions are sampled.
    Args:
        inp_shape: torch.Size, typeically (batch, d)
        num_unmask: (batch,) int tensor
        select_from_mask: (batch, d) tensor, if provided, we only sample from the selected
        selection_score: logit-like score for selection (can be negative). Should match inp_shape, so typically (batch, d)
        score_mode:
            "logits" => p_i = \exp(s_i)/\sum_j \exp(s_j)
            "uprobs" => p_i = s_i / \sum_j s_j
    """
    if select_from_mask is not None:
        mask = select_from_mask
    else:
        # mask = torch.ones_like(inp, dtype=torch.bool)
        mask = torch.ones(
            *inp_shape, dtype=torch.bool, device=num_unmask.device
        )
    # create a tensor of shape (bs, seq) with
    # first num_unmask[i] elements of row i are 1 rest are 0
    # print(f"\nnum_unmask={num_unmask}")
    # temp = torch.nn.functional.one_hot(num_unmask, num_classes=inp.shape[-1])
    temp = torch.nn.functional.one_hot(num_unmask, num_classes=inp_shape[-1])
    # print(f"\ntemp=\n{temp}")
    temp2 = (1 - temp.cumsum(dim=-1)).to(dtype=torch.bool)
    # print(f"\ntemp2=\n{temp2.int()}")

    # now we shuffle all indices but pinning the indices for non-mask to the right end
    # rand = torch.rand(bs, seq)
    # rand = torch.rand_like(inp, dtype=torch.float32)
    if selection_score is None:  # uniform
        rand = torch.rand(
            *inp_shape, dtype=torch.float32, device=num_unmask.device
        )
    else:
        if selection_mode == "greedy":
            rand = selection_score.clone()
        elif selection_mode == "sample":
            if score_mode == "logits":
                rand = add_gumbel_noise(
                    selection_score, temperature=temperature
                )
            elif score_mode == "uprobs":
                rand = add_exp_1_noise(selection_score)
            else:
                raise ValueError(f"score_mode={score_mode} not supported")

        else:
            raise ValueError(f"selection_mode={selection_mode} not supported")
    rand[~mask] = float(
        "-inf"
    )  # to keep non-masked positions always on the right
    rand_perm = torch.argsort(rand, dim=-1, descending=True)
    # print(f"\nrand_perm=\n{rand_perm}")
    # rand_perm and temp2 have the information to index into a
    # unmask_out = torch.zeros_like(inp, dtype=torch.bool).scatter_(
    #    -1, rand_perm, temp2
    # )
    unmask_out = torch.zeros(
        *inp_shape, dtype=torch.bool, device=num_unmask.device
    ).scatter_(-1, rand_perm, temp2)
    # print(f"unmask=\n{unmask_out.int()}")
    # check if all unmask_out are mask
    assert mask[unmask_out].all()
    # check that the number of unmask matches the target
    assert (unmask_out.sum(-1) == num_unmask).all()
    return unmask_out

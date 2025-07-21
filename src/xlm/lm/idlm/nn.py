"""Neural network utilities for IDLM model.

This file contains utility functions for IDLM computations.
"""

import torch
from torch import Tensor as TT
from jaxtyping import Integer, Float


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

from typing import Optional, Tuple
from jaxtyping import Integer, Float, Bool
from torch import Tensor as TT
import torch


def get_tertiary_relative_position_matrix(
    pi: Integer[TT, "batch_size seq_len"],
) -> Integer[TT, "batch_size seq_len seq_len"]:
    """
    Get the tertiary relative position matrix for a given permutation with entries -1, 0, 1.
    r_ij = 1 if pi_i > pi_j,
           -1 if pi_i < pi_j,
           0 if pi_i == pi_j.
    """
    pi1 = pi.unsqueeze(-1)  # (batch_size, seq_len, 1)
    pi2 = pi.unsqueeze(-2)  # (batch_size, 1, seq_len)
    r = (pi1 > pi2).long() - (
        pi1 < pi2
    ).long()  # (batch_size, seq_len, seq_len) with entries -1, 0, 1
    return r


def is_valid_pi(pi: Integer[TT, "batch_size seq_len"]) -> bool:
    """
    Check if the permutation is valid.
        1. The first position should be min (corresponding to BOS) and the second position should be max (corresponding to EOS)
    """
    min_pos = torch.min(pi, dim=-1).values
    max_pos = torch.max(pi, dim=-1).values
    check_0 = pi[:, 0] == min_pos
    if not check_0.all():
        raise ValueError(
            "The first position should be min (corresponding to BOS)"
        )
    check_1 = pi[:, 1] == max_pos
    if not check_1.all():
        raise ValueError(
            "The second position should be max (corresponding to EOS)"
        )
    return True


def get_absolute_position_matrix(
    r: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len seq_len"]:
    """
    Get the absolute position matrix M, with M[i, t] being the absolute position of token i at generation step t.
    """
    num_elements_on_left = torch.max(
        r, torch.zeros(1, dtype=r.dtype, device=r.device).expand_as(r)
    )

    m = torch.cumsum(num_elements_on_left, dim=-1).triu(
        0
    )  # fill the lower triangle with 0
    return m


def _get_right_pointer_position(
    m: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len"]:
    """
    Get the position of the closest token on the right of the token being inserted.

    Args:
        m: The absolute position matrix containing absolute position for each step. The output of `get_absolute_position_matrix` above.
    Returns:
      p: Where p[*, t] is the position w.r.t input order (z) of the closest token on the right of the token being inserted (t-th token).
    """
    # diag[*,t]=m[*,t,t]: The absolute position of t-th token (the one being inserted at step t) in the visible sequence at step t.
    diag = torch.diagonal(m, dim1=-2, dim2=-1)  # (batch_size, seq_len)
    # mask[*,j,t]=1 if j<t and m[*,j,t]<m[*,t,t]
    mask = (diag.unsqueeze(-2) < m).triu(
        0
    )  # (batch_size, seq_len, seq_len) # lower triangle to 0
    temp = m.masked_fill(
        ~mask, 100000
    )  # Fill the lower triangle and any position not on the right by a large number
    right_pointer_position = torch.argmin(temp, dim=-2)  # minimum over column
    return right_pointer_position


def _get_left_pointer_position(
    m: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len"]:
    """
    Get the position of the closest token on the left of the token being inserted.
    """
    diag = torch.diagonal(m, dim1=-2, dim2=-1)  # (batch_size, seq_len)
    mask = (diag.unsqueeze(-2) > m).triu(0)  # (batch_size, seq_len, seq_len)
    temp = m.masked_fill(
        ~mask, -100000
    )  # Fill the lower triangle and any position not on the left by a large negative number
    left_pointer_position = torch.argmax(temp, dim=-2)  # maximum over column
    return left_pointer_position


def get_left_right_pointer_position(
    pi: Integer[TT, "batch_size seq_len"],
    roll_over_fill_value: Optional[int] = None,
) -> Tuple[
    Integer[TT, "batch_size seq_len"], Integer[TT, "batch_size seq_len"]
]:
    """
    Get the position of the closest token on the left and right of the token being inserted.

    Args:
        pi: The permutation of the input sequence.
        roll_over_fill_value: If provided, move the pointer matrix by one to left and fill the rightmost column with fill_value.
    Returns:
        lp: The position of the closest token on the left of the token being inserted.
        rp: The position of the closest token on the right of the token being inserted.
    """
    m = get_absolute_position_matrix(get_tertiary_relative_position_matrix(pi))
    lp = _get_left_pointer_position(m)
    rp = _get_right_pointer_position(m)
    if roll_over_fill_value is not None:
        lp = lp.roll(shifts=-1, dims=-1)
        rp = rp.roll(shifts=-1, dims=-1)
        lp[..., -1] = roll_over_fill_value
        rp[..., -1] = roll_over_fill_value
    return lp, rp


def masked_logsumexp(
    logits: Float[TT, "*batch num_classes"],
    mask: Bool[TT, "*batch num_classes"],
    min_value: float,
) -> Float[TT, "batch seq_len"]:
    """
    Compute the logsumexp of the logits, ignoring the masked positions.
    """
    # if all items are masked, we don't manually place a fill-value.
    # the later computation can generate NaNs in that case.
    logits_masked = logits.masked_fill(mask, min_value)
    lse = torch.logsumexp(logits_masked, dim=-1)
    return lse


def get_closest_right_neighbor(
    pi: Integer[TT, "batch_size seq_len"]
) -> Integer[TT, "batch_size seq_len"]:
    """
    Get the index of the closest right neightbor.
    """
    is_on_right = pi.unsqueeze(-1) < pi.unsqueeze(-2)
    # is_on_right[*, i, j] = 1 if pi[i] < pi[j]
    r = get_tertiary_relative_position_matrix(pi)
    m = get_absolute_position_matrix(r)
    rp = _get_right_pointer_position(m)
    return rp


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 3

    # generate a valid pi
    # pi = torch.rand((batch_size, seq_len)).argsort(dim=-1) + 1
    # pi = torch.cat(
    #    [
    #        torch.zeros(batch_size, 1, dtype=pi.dtype, device=pi.device),
    #        torch.ones(batch_size, 1, dtype=pi.dtype, device=pi.device)
    #        * (seq_len + 1),
    #        pi,
    #    ],
    #    dim=-1,
    # )
    pi = torch.tensor(
        [
            [0, 4, 3, 1, 2, 5, 6],
            [0, 6, 3, 4, 2, 1, 5],
        ]
    )
    # is_valid_pi(pi)
    print(f"pi:\n {pi}")

    r = get_tertiary_relative_position_matrix(pi)
    print(f"r:\n {r}")
    m = get_absolute_position_matrix(r)
    print(f"m:\n {m}")
    rp = _get_right_pointer_position(m)
    print(f"rp:\n {rp}")

    lp = _get_left_pointer_position(m)
    print(f"lp:\n {lp}")

    """
    Expected output:
    tokens: [
            [BOS,EOS, a, b, c, EOD, PAD]
            [BOS,EOS, e, f, g, h, EOD]
    ]
    pi:
    tensor([[0, 4, 3, 1, 2, 5, 6],
            [0, 5, 3, 4, 2, 1, 6]])
    r:
    tensor([[[ 0, -1, -1, -1, -1, -1, -1],
            [ 1,  0,  1,  1,  1, -1, -1],
            [ 1, -1,  0,  1,  1, -1, -1],
            [ 1, -1, -1,  0, -1, -1, -1],
            [ 1, -1, -1,  1,  0, -1, -1],
            [ 1,  1,  1,  1,  1,  0, -1],
            [ 1,  1,  1,  1,  1,  1,  0]],

            [[ 0, -1, -1, -1, -1, -1, -1],
            [ 1,  0,  1,  1,  1,  1,  1],
            [ 1, -1,  0, -1,  1,  1, -1],
            [ 1, -1,  1,  0,  1,  1, -1],
            [ 1, -1, -1, -1,  0,  1, -1],
            [ 1, -1, -1, -1, -1,  0, -1],
            [ 1, -1,  1,  1,  1,  1,  0]]])
    m:
    tensor([[[0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 4, 4],
            [0, 0, 1, 2, 3, 3, 3],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 5, 5],
            [0, 0, 0, 0, 0, 0, 6]],

            [[0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 0, 1, 1, 2, 3, 3],
            [0, 0, 0, 2, 3, 4, 4],
            [0, 0, 0, 0, 1, 2, 2],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 5]]])
    rp:
    tensor([[0, 0, 1, 2, 2, 0, 0],
            [0, 0, 1, 1, 2, 4, 1]])
    lp:
    tensor([[0, 0, 0, 0, 3, 1, 5],
            [0, 0, 0, 2, 0, 0, 3]])
    """

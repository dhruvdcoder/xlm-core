from jaxtyping import Integer
from torch import Tensor as TT
import torch


def get_tertiary_relative_position_matrix(
    pi: Integer[TT, "batch_size seq_len"],
) -> Integer[TT, "batch_size seq_len seq_len"]:
    """
    Get the tertiary relative position matrix for a given permutation with entries -1, 0, 1.
    """
    pi1 = pi.unsqueeze(-1)  # (batch_size, seq_len, 1)
    pi2 = pi.unsqueeze(-2)  # (batch_size, 1, seq_len)
    r = (pi1 > pi2).long() - (
        pi1 < pi2
    ).long()  # (batch_size, seq_len, seq_len) with entries -1, 0, 1
    return r


def get_absolute_position_matrix(
    r: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len seq_len"]:
    """
    Get the absolute position matrix M, with M[i, t] being the absolute position of token i at generation step t.
    """
    num_elements_on_left = torch.max(
        r, torch.zeros(1, dtype=r.dtype, device=r.device).expand_as(r)
    )

    m = torch.cumsum(num_elements_on_left, dim=-1).triu(0)
    return m


def get_right_pointer_position(
    m: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len"]:
    """
    Get the right pointer for each generation step.

    Args:
        m: The absolute position matrix containing absolute position for each step. The output of `get_absolute_position_matrix` above.
    """
    diag = torch.diagonal(m, dim1=-2, dim2=-1)  # (batch_size, seq_len)
    diag = diag - 1
    temp = (diag.unsqueeze(-2) == m).to(torch.int64) # (batch, seq_len, seq_len)
    # Currently triu does not support fill_value (https://github.com/pytorch/pytorch/issues/97892). So we use masked_scatter
    temp.masked_fill_(torch.tril(torch.ones_like(temp).bool(), diagonal=-1), -1)
    right_pointer = temp.argmax(dim=1) # (batch, seq_len, seq_len)
    mask = diag == -1
    right_pointer[mask] = -1
    return right_pointer

def get_left_pointer_position(
    m: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len"]:
    """
    Get the left pointer for each generation step.

    Args:
        m: The absolute position matrix containing absolute position for each step. The output of `get_absolute_position_matrix` above.
    """
    diag = torch.diagonal(m, dim1=-2, dim2=-1)  # (batch_size, seq_len)
    _, seq_len = diag.shape
    diag = diag + 1
    temp = (diag.unsqueeze(-2) == m).to(torch.int64) # (batch, seq_len, seq_len)
    # Currently triu does not support fill_value (https://github.com/pytorch/pytorch/issues/97892). So we use masked_scatter
    temp.masked_fill_(torch.tril(torch.ones_like(temp).bool(), diagonal=-1), -1)
    left_pointer = temp.argmax(dim=1) # (batch, seq_len, seq_len)
    mask = (diag > torch.arange(seq_len, device=diag.device))
    left_pointer[mask] = -1
    return left_pointer

def get_left_right_pointer_position(
    m: Integer[TT, "batch_size seq_len seq_len"],
) -> Integer[TT, "batch_size seq_len"]:
    """
    Get the left and right pointer for each generation step.

    Args:
        m: The absolute position matrix containing absolute position for each step. The output of `get_absolute_position_matrix` above.
    """
    diag = torch.diagonal(m, dim1=-2, dim2=-1)  # (batch_size, seq_len)
    left_pointer = (diag.unsqueeze(-2) > m).masked_fill(torch.triu(m).bool(), -100000).argmax(dim=-1)
    right_pointer = (diag.unsqueeze(-2) < m).masked_fill(torch.triu(m).bool(), 100000).argmin(dim=-1)
    return left_pointer, right_pointer



if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5

    pi = torch.tensor([[0,1,2,3,4],[4,3,2,1,0],[0,2,1,4,3],[1,0,4,3,2]])
    print(f"pi:\n {pi}")

    r = get_tertiary_relative_position_matrix(pi)
    print(f"r:\n {r}")
    m = get_absolute_position_matrix(r)
    print(f"m:\n {m}")
    lp = get_left_pointer_position(m)
    rp = get_right_pointer_position(m)
    print(f"left_pointer:\n {lp}")
    print(f"right_pointer:\n {rp}")
    

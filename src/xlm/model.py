from typing import Protocol
from typing import Iterator, Tuple
import torch


class Model(Protocol):
    def get_named_params_for_weight_decay(
        self,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]: ...

    def get_named_params_for_no_weight_decay(
        self,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]: ...

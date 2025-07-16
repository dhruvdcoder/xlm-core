from typing import Protocol
from typing import Iterator, Tuple
import torch
from huggingface_hub import PyTorchModelHubMixin


class Model(PyTorchModelHubMixin):
    def get_named_params_for_weight_decay(
        self,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        raise NotImplementedError

    def get_named_params_for_no_weight_decay(
        self,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        raise NotImplementedError

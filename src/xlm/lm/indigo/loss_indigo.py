from typing import (
    Any,
    Dict,
    Optional,
)


from xlm.harness import (
    Harness,
    LossFunction,
)
from xlm.utils.rank_zero import RankedLogger
from .types_indigo import IndigoBatch, IndigoLossDict

logger = RankedLogger(__name__, rank_zero_only=True)


###############################################################
# region: Loss functions


class IndigoLoss(LossFunction[IndigoBatch, IndigoLossDict]):
    def __init__(
        self,
        model=None,
        tokenizer=None,
    ):
        self.model = model
        self.tokenizer = tokenizer  # type: ignore

    def configure(self, pl_module: Harness):
        pass  # nothing to do here for now

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def loss_fn(
        self,
        batch: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # TODO (URV): Implement the loss function.
        pass

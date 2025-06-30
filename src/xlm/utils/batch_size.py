from typing import Optional
import lightning as L
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def is_ddp(trainer: Optional[L.Trainer]) -> bool:
    if trainer is not None:
        strategy = trainer.strategy
        if isinstance(strategy, DDPStrategy):
            return True
        elif isinstance(strategy, SingleDeviceStrategy):
            return False
        else:
            raise ValueError(
                f"Dataloader does not support {type(strategy)} strategy"
            )
    else:
        logger.warning(
            "Tried to detect DDP strategy before trainer was set."
            "Are you calling `LightningDataModule.*_dataloader()` methods manually?"
        )
        return False


def determine_per_device_batch_size(
    trainer: Optional[L.Trainer], batch_size: int
) -> int:
    if trainer is None:
        raise ValueError(
            "Trainer is not setup. Cannot determine the number of devices."
        )
    if is_ddp(trainer):
        num_nodes = trainer.num_nodes
        num_gpus_per_node = trainer.num_devices
        accum_steps = trainer.accumulate_grad_batches
        per_device_batch_size = batch_size // (
            num_nodes * num_gpus_per_node * accum_steps
        )
        remainder = batch_size % (num_nodes * num_gpus_per_node * accum_steps)
        if remainder != 0:
            raise ValueError(
                "Batch size is not divisible by the number of nodes, GPUs per node and accum_steps."
            )
        return per_device_batch_size
    else:
        return batch_size

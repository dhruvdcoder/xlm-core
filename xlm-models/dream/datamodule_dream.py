"""Dream-specific data utilities for xlm-core (batch printing, etc.)."""

from typing import Any, Dict

from xlm.datamodule import Tokenizer
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def print_batch_dream(
    batch: Dict[str, Any],
    split: str,
    tokenizer: Tokenizer,
    dataloader_name: str = "",
) -> None:
    """Debug helper that logs the first example of a Dream / MLM prediction batch."""
    ids = batch["input_ids"][0]
    mask = batch["attention_mask"][0]
    text = tokenizer.decode(ids[mask.bool()], skip_special_tokens=False)
    logger.info(
        f"[Dream] split={split}  dl_name={dataloader_name}  "
        f"shape={tuple(batch['input_ids'].shape)}  "
        f"prompt_preview={text}"
    )

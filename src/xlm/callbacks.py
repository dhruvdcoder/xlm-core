import json
from pathlib import Path
from typing import (
    Optional,
)


import torch
from torch import Tensor as TT
from xlm.datamodule import Tokenizer
from .generative_perplexity import (
    GenerativePerplexityEvaluator,
    GenerativePerplexityEvaluatorResult,
)
from torchmetrics import Perplexity
import lightning as L

############################################################
# region: Callbacks


class GenerativePerplexityCallbackBase:
    def __init__(
        self,
        evaluator: GenerativePerplexityEvaluator,
        predictions_dir: Path,
    ):
        """
        Args:
            evaluator: Any generic evaluator that implements the protocol.
        """
        self.evaluator = evaluator
        ignore_index = evaluator.ignore_index
        assert ignore_index is not None
        logger.info(
            f"Using ignore_index: {ignore_index} for generative perplexity with "
            f"Evaluator: {evaluator}"
        )
        self.predictions_dir = predictions_dir

    def _get_unconditional_samples_file_path(
        self,
        split: str,  # Literal["train", "val", "test"],
        dataloader_name: str,
        epoch: int,
        step: int,
    ) -> Path:

        return (
            self.predictions_dir
            / f"{split}/{dataloader_name}/{epoch=}_{step=}.jsonl"
        )

    @torch.inference_mode()
    def _on_validation_epoch_end(
        self,
        split: str,  # Literal["train", "val", "test"],
        dataloader_name: str,
        epoch: int,
        step: int,
        device: torch.device,
        tokenizer: Tokenizer,
    ) -> Optional[TT]:
        file_path = self._get_unconditional_samples_file_path(
            split,
            dataloader_name,
            epoch,
            step,
        )

        perplexity = self.compute_generative_perplexity(
            self.evaluator,
            device,
            tokenizer,
            file_path,
        )
        return perplexity

    @staticmethod
    def compute_generative_perplexity(
        evaluator: GenerativePerplexityEvaluator,
        device: torch.device,
        tokenizer: Tokenizer,
        file_path: Path,
    ) -> Optional[TT]:
        if not file_path.exists():
            logger.error(
                f"No samples found at {file_path}. "
                "If you are using a callback like UnconditionalSampleGenerator, to generate samples, "
                "make sure that it is placed before GenerativePerplexity in the callbacks list."
            )
            return None

        # Load samples from file and evaluate in batches
        samples = []
        generative_perplexity_metric = Perplexity(
            ignore_index=evaluator.ignore_index
        ).to(device)
        generative_perplexity_metric.reset()
        with evaluator.loaded(
            tokenizer,  # type: ignore
            device,
        ):  # load it on the same device for now
            with open(file_path) as f:
                for line in f:
                    sample = json.loads(line)
                    text = sample["text"]
                    samples.append(text)

                    if len(samples) == evaluator.batch_size:
                        result: Optional[
                            GenerativePerplexityEvaluatorResult
                        ] = evaluator(samples)
                        # reset the samples list for the next batch
                        samples = []
                        if result is None:
                            continue
                        logits = result["logits"]
                        target = result["target"]
                        generative_perplexity_metric.update(logits, target)

                # Handle remaining samples in last batch if any
                if samples:
                    result = evaluator(samples)
                    if result is not None:
                        logits = result["logits"]
                        target = result["target"]
                        generative_perplexity_metric.update(logits, target)

        perplexity = generative_perplexity_metric.compute()
        return perplexity


class GenerativePerplexityCallback(
    GenerativePerplexityCallbackBase, L.Callback
):
    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningDataModule
    ) -> None:
        # just handshake
        self._on_validation_epoch_end(
            split="val",
            dataloader_name="prediction",
            epoch=trainer.current_epoch,
            step=trainer.global_step,
            device=pl_module.device,
            tokenizer=pl_module.tokenizer,
        )


# endregion: Callbacks
############################################################

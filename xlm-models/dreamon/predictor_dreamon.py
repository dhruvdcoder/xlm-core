from typing import Any, Dict, List, Optional

import torch
from xlm.datamodule import Tokenizer
from xlm.harness import Predictor

from .dreamon_model import DreamOnModel


class DreamOnPredictor(torch.nn.Module, Predictor[Any, Dict[str, Any]]):
    """Harness wrapper around `DreamOnModel.diffusion_generate`."""

    def __init__(
        self,
        model: DreamOnModel,
        tokenizer: Tokenizer,
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.diffusion_kwargs = diffusion_kwargs or {}

    def predict(
        self,
        batch: Any,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_ids = batch["input_ids"]
        out = self.model.diffusion_generate(
            input_ids,
            **self.diffusion_kwargs,
        )
        seq = out.sequences if hasattr(out, "sequences") else out
        texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        return {"text": texts, "ids": seq}

    def to_dict(
        self,
        batch: Any,
        preds: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [{"text": t} for t in preds["text"]]

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError(
            "DreamOnPredictor.generate is not implemented; use predict()."
        )

from typing import Optional, cast

import torch
from .types_mlm import MLMBatch, MLMLossDict, MLMModel
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from xlm.utils.nn import masked_mean


class MLMLoss(LossFunction[MLMBatch, MLMLossDict]):
    def __init__(
        self,
        loss_on_padding: bool = False,
        loss_on_visible_tokens: bool = False,
        model: Optional[MLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_num_masked_factor: bool = False,
    ):
        self.loss_on_padding = loss_on_padding
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None
        self.use_num_masked_factor = use_num_masked_factor
        self.flash_attn = None

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        # Move FlexAttention BlockMask to the correct device here, before the
        # compiled loss_fn.  BlockMask is a custom PyTorch object so Lightning's
        # auto-transfer does not handle it; moving it in __call__ (which is NOT
        # compiled) keeps the compiled region graph-break-free.
        block_mask = batch.get("block_mask")
        if block_mask is not None:
            target_device = batch["input_ids"].device
            if block_mask.kv_indices.device != target_device:
                block_mask = block_mask.to(target_device)
            # The collator builds the BlockMask on CPU; the mask_mod closure it
            # stores captures a CPU `seg` tensor and uses 2-D indexing
            # (seg[b, q_idx]).  Inductor cannot lower multi-index gathers as
            # pointwise subgraphs, and the CPU tensor causes a device mismatch.
            # Replace mask_mod here (before any compiled region) with a fresh
            # closure that (1) holds the already-GPU segment_ids and (2) uses
            # flattened 1-D indexing — a single-index load, which is lowerable.
            segment_ids = batch.get("segment_ids")
            if segment_ids is not None:
                seg_flat = segment_ids.to(target_device).reshape(-1)
                seq_len_int: int = segment_ids.shape[1]
                def _patched_mask_mod(b, h, q_idx, kv_idx,
                                      _sf=seg_flat, _sl=seq_len_int):
                    return _sf[b * _sl + q_idx] == _sf[b * _sl + kv_idx]
                block_mask.mask_mod = _patched_mask_mod
            batch = {**batch, "block_mask": block_mask}
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        if self.flash_attn is None:
            self.flash_attn = getattr(self.model, "force_flash_attn", False)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(dtype=torch.bool)
        targets = batch["target_ids"]
        assert targets is not None

        model = cast(MLMModel, self.model)

        if attention_mask.ndim == 3:
            # Packed sequences: collator already computed per-sequence positions,
            # a block-diagonal 3D attention mask, and per-token segment ids.
            # When the model supports FlexAttention (use_flex_attn=True), it will
            # build a BlockMask from segment_ids and ignore attention_mask.
            positions = batch.get("positions")
            segment_ids = batch.get("segment_ids")
            block_mask = batch.get("block_mask")
            logits = model(input_ids, attention_mask, positions, segment_ids=segment_ids, block_mask=block_mask)
        else:
            # Standard padded sequences: derive monotonic positions from the
            # 1-D padding mask (cumulative count of non-padding tokens).
            positions = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
            positions *= attention_mask  # technically not needed
            logits = model(input_ids, attention_mask if not self.flash_attn else None, positions)

        ignore = torch.zeros_like(input_ids, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(input_ids != self.mask_token_id_tensor)  # type: ignore
        targets[ignore] = -100
        if ignore.all():
            # Need to do this manually because pytorch doesn't do the logical thing. See https://github.com/pytorch/pytorch/issues/70348
            return {
                "loss": torch.tensor(
                    0.0,
                    device=logits.device,
                    dtype=logits.dtype,
                    requires_grad=True,
                )
            }

        logits_T = logits.transpose(1, 2)

        ce = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="none", ignore_index=-100
        ) # shape (batch, seq_len)
        if self.use_num_masked_factor:
            num_masked = (input_ids == self.mask).sum(dim=-1)
            factor = 1 / (num_masked + 1)
            ce = ce * factor
        ce = masked_mean(ce.flatten(), ~ignore.flatten(), dim=-1)
        # we can compute nlls by indexing using non-ignored tokens, but right now we will just use reduction=mean
        return {"loss": ce}

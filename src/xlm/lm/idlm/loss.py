"""Loss function implementation for Idlm model.

This file implements the IDLM training loss computation.
Based on IDLM v2 implementation but adapted for XLM framework.
"""

from typing import Optional, List, Callable, Literal, cast
import torch
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from xlm.lm.ilm.nn import masked_ce_last_two_dims
from .nn import incomplete_gamma_factor_using_series
from .types import IdlmBatch, IdlmLossDict, IdlmModel


class IdlmLoss(LossFunction[IdlmBatch, IdlmLossDict]):
    """Loss function for IDLM model with masked cross-entropy and length prediction.

    This implements the IDLM diffusion loss with:
    - Masked cross-entropy loss for token prediction
    - Length prediction loss (cross-entropy or diffusion-based)
    - Optional diffusion weighting
    - Support for constraints during generation
    """

    def __init__(
        self,
        model: Optional[IdlmModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        loss_on_padding: bool = False,
        use_constraint: bool = False,
        use_diffusion_weight_for_ce: bool = False,
        use_diffusion_weight_for_length_loss: bool = False,
        use_n_drops_for_ce: bool = False,
        length_loss: Literal["diffusion", "cross_entropy"] = "cross_entropy",
        ce_weight: float = 1.0,
        length_loss_weight: float = 1.0,
        length_class_weights: Optional[Callable[[int], torch.Tensor]] = None,
        max_diffusion_weight: float = 100.0,
        use_incomplete_gamma_factor: bool = False,
        send_t_to_model: bool = False,
    ):
        """Initialize the IDLM loss function.

        Args:
            model: The IDLM model instance
            tokenizer: Tokenizer for processing tokens
            loss_on_padding: Whether to compute loss on padding tokens
            use_constraint: Whether to use constraint masking
            use_diffusion_weight_for_ce: Whether to apply diffusion weighting to CE loss
            use_diffusion_weight_for_length_loss: Whether to apply diffusion weighting to length loss
            use_n_drops_for_ce: Whether to scale CE diffusion weight by number of drops
            length_loss: Type of length loss ("cross_entropy" or "diffusion")
            ce_weight: Weight for cross-entropy loss
            length_loss_weight: Weight for length loss
            length_class_weights: Function to generate class weights for length prediction
            max_diffusion_weight: Maximum value for diffusion weights
            use_incomplete_gamma_factor: Whether to use incomplete gamma factor correction
            send_t_to_model: Whether to send time step t to model (vs total_noise)
        """
        self.model = model
        self.tokenizer = tokenizer  # type: ignore
        self.loss_on_padding = loss_on_padding
        self.use_constraint = use_constraint
        self.use_diffusion_weight_for_ce = use_diffusion_weight_for_ce
        self.use_diffusion_weight_for_length_loss = (
            use_diffusion_weight_for_length_loss
        )
        self.use_n_drops_for_ce = use_n_drops_for_ce
        self.length_loss = length_loss
        self.ce_weight = ce_weight
        self.length_loss_weight = length_loss_weight
        self.length_class_weights_fn = length_class_weights
        self.max_diffusion_weight = max_diffusion_weight
        self.use_incomplete_gamma_factor = use_incomplete_gamma_factor
        self.send_t_to_model = send_t_to_model

        # Will be set during configure()
        self.mask_token_id_tensor: Optional[torch.Tensor] = None
        self._min_value: Optional[float] = None
        self.length_class_weights: Optional[torch.Tensor] = None
        self.w_len: Optional[torch.Tensor] = None
        self.w_ce: Optional[torch.Tensor] = None
        self.factorials: Optional[torch.Tensor] = None

        # Validation
        if (
            self.length_loss == "diffusion"
            and self.length_class_weights_fn is not None
        ):
            raise ValueError(
                "length_class_weights_fn is not supported for diffusion length loss"
            )

    def min_value(self, logits: torch.Tensor) -> float:
        """Get the minimum value for the given logits dtype."""
        if self._min_value is None:
            self._min_value = torch.finfo(logits.dtype).min
        return self._min_value

    def configure(self, pl_module: Harness) -> None:
        """Configure the loss function with the lightning module."""
        assert self.tokenizer is not None
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,  # type: ignore
            dtype=torch.long,
            device=pl_module.device,
        )

        # Set up length class weights if provided
        if self.length_class_weights_fn is not None:
            num_classes = cast(IdlmModel, self.model).num_classes
            weights = self.length_class_weights_fn(num_classes).to(
                device=pl_module.device
            )
            self.length_class_weights = weights

        # Set up loss weights as tensors
        self.w_len = torch.tensor(
            self.length_loss_weight, device=pl_module.device
        )
        self.w_ce = torch.tensor(self.ce_weight, device=pl_module.device)

        # Set up factorials for diffusion length loss
        if self.length_loss == "diffusion":
            num_classes = cast(IdlmModel, self.model).num_classes
            self.factorials = torch.lgamma(
                torch.arange(num_classes, device=pl_module.device) + 1
            )

    def create_mask(
        self,
        non_pad: torch.Tensor,
        constraint: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Create mask for loss computation."""
        if constraint is None or not self.use_constraint:
            return (non_pad.logical_not_()[:, 1:]).unsqueeze(-1)
        elif self.use_constraint and constraint is not None:
            return (
                non_pad.logical_not_().logical_or_(constraint)[:, 1:]
            ).unsqueeze(-1)
        else:
            raise ValueError("Invalid constraint")

    def get_compilable_functions(self) -> List[Callable]:
        """Get functions that can be compiled for efficiency."""
        return [self.loss_fn]

    def __call__(
        self,
        batch: IdlmBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> IdlmLossDict:
        """Compute the IDLM loss."""
        # Handle sparse/dense target conversion if needed
        target_ids = batch["target_ids"]
        if hasattr(target_ids, "to_dense"):
            # Convert sparse to dense for computation
            batch_copy = batch.copy()
            batch_copy["target_ids"] = target_ids.to_dense()
            loss_dict = self.loss_fn(
                batch_copy, batch_idx, dataloader_idx, dataloader_name
            )
        else:
            loss_dict = self.loss_fn(
                batch, batch_idx, dataloader_idx, dataloader_name
            )

        return loss_dict

    def loss_fn(
        self,
        batch: IdlmBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> IdlmLossDict:
        """Core loss computation for IDLM.

        Args:
            batch: The input batch
            batch_idx: Batch index
            dataloader_idx: Dataloader index
            dataloader_name: Dataloader name

        Returns:
            Dictionary containing loss components
        """
        # Extract batch components
        x_t = batch["input_ids"]  # shape (batch, seq_len)
        target_ids = batch["target_ids"]  # shape (batch, seq_len, vocab_size)
        noise_rate = batch["noise_rate"]  # shape (batch,)
        total_noise = batch["total_noise"]  # shape (batch,)
        t = batch["t"]  # shape (batch,)
        constraint = batch["constraint"]  # shape (batch, seq_len) or None
        cls_position = batch["cls_position"]  # shape (batch,)

        assert self.model is not None

        # Handle attention mask - since dropped tokens are physically removed,
        # we only need to consider padding
        non_pad = (
            batch["attention_mask"].to(dtype=torch.bool)
            if not self.loss_on_padding
            else torch.ones_like(batch["attention_mask"], dtype=torch.bool)
        )  # shape (batch, seq_len)

        # Create positions (cumulative sum of non-padded positions)
        positions = (torch.cumsum(non_pad, dim=-1) - 1).clamp_min(0).long()

        # Forward pass through model
        model = cast(IdlmModel, self.model)
        logits, length_logits = model(
            x_t,
            t if self.send_t_to_model else total_noise,
            non_pad,  # Use non_pad since no dropped tokens in the new approach
            positions=positions,
            cls_position=cls_position,
        )

        # Get number of drops from the batch (more efficient than computing from target_ids)
        n_drops_tensor = batch["n_drops"]
        if hasattr(n_drops_tensor, "to_dense"):
            # Convert sparse tensor to dense and sum to get total drops per example
            n_drops = n_drops_tensor.to_dense().sum(dim=1)
        else:
            # If already dense, sum along sequence dimension
            n_drops = n_drops_tensor.sum(dim=1)

        # Normalize target probabilities
        p_target = target_ids / (n_drops[:, None, None] + 1e-9)

        # Create loss mask (skip first position reserved for length prediction)
        loss_mask = self.create_mask(non_pad, constraint)

        # Compute cross-entropy loss using masked CE
        ce = masked_ce_last_two_dims(
            logits[:, 1:, :],  # Skip first position
            p_target[:, 1:, :],  # Skip first position
            loss_mask,
            min_value=self.min_value(logits),
            inplace=False,
        )

        # Compute length loss
        if self.length_loss == "cross_entropy":
            length_loss = torch.nn.functional.cross_entropy(
                length_logits,
                n_drops.long(),
                reduction="none",
                weight=self.length_class_weights,
            )  # shape (batch,)
        elif self.length_loss == "diffusion":
            pred_expected_delta_l = model.get_mean_delta_l(
                length_logits, batch["attention_mask"]
            )  # shape (batch,)
            # Diffusion-based length loss with factorial term
            assert (
                self.factorials is not None
            ), "Factorials must be initialized for diffusion length loss"
            length_loss = (
                pred_expected_delta_l
                - n_drops * torch.log(pred_expected_delta_l + 1e-8)
                + self.factorials[
                    n_drops.long().clamp(0, len(self.factorials) - 1)
                ]
            )  # shape (batch,)
        else:
            raise ValueError(f"Unknown length_loss type: {self.length_loss}")

        # Apply loss weights
        assert self.w_len is not None and self.w_ce is not None
        w_len = self.w_len.unsqueeze(0)  # shape (1,)
        w_ce = self.w_ce.unsqueeze(0)  # shape (1,)

        # Apply diffusion weighting if requested
        if (
            self.use_diffusion_weight_for_length_loss
            or self.use_diffusion_weight_for_ce
        ):
            diffusion_weight_len = (
                noise_rate / (total_noise + 1e-6)
            ).clamp_max(
                self.max_diffusion_weight
            )  # shape (batch,)

            if self.use_incomplete_gamma_factor:
                inp_len = non_pad.sum(dim=-1)
                factor = torch.where(
                    inp_len - n_drops > 0,
                    torch.ones_like(diffusion_weight_len),
                    incomplete_gamma_factor_using_series(n_drops, total_noise),
                )
                diffusion_weight_len = diffusion_weight_len / factor

            if self.use_n_drops_for_ce:
                diffusion_weight_ce = diffusion_weight_len * n_drops
            else:
                diffusion_weight_ce = diffusion_weight_len

            if (
                self.use_diffusion_weight_for_length_loss
                and self.use_diffusion_weight_for_ce
            ):
                w_len = w_len * diffusion_weight_len
                w_ce = w_ce * diffusion_weight_ce
            elif self.use_diffusion_weight_for_length_loss:
                w_len = w_len * diffusion_weight_len
            else:  # use_diffusion_weight_for_ce
                w_ce = w_ce * diffusion_weight_ce

        # Apply weights and combine losses
        ce = ce * w_ce
        length_loss = length_loss * w_len
        example_loss = ce + length_loss
        loss = example_loss.mean()

        return {
            "loss": loss,
            "batch_loss": example_loss.detach(),
            "per_example_length_loss": length_loss.detach(),
            "per_example_ce": ce.detach(),
            "length_logits": length_logits.detach(),
            "n_drops": n_drops.detach(),
        }

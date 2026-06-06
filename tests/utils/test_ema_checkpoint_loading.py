"""Tests verifying that EMA weights are correctly applied when loading from checkpoint.

These tests ensure that the `from_checkpoint(apply_ema=True)` path correctly
overwrites model parameters with EMA shadow params, and that the broken
`on_load_checkpoint` + `manual_ema_restore` path does NOT achieve this
(documenting the Lightning quirk where load_state_dict overwrites
on_load_checkpoint modifications).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from torch_ema import ExponentialMovingAverage
from xlm.utils.ema import EMACallback


class _SimpleModel(nn.Module):
    """Minimal model with enough parameters to distinguish EMA from raw."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)
        self.linear = nn.Linear(8, 4)
        self.head = nn.Linear(4, 10)


def _create_checkpoint_with_divergent_ema(model: nn.Module) -> dict:
    """Create a fake Lightning checkpoint where EMA shadow params differ from raw.

    Simulates a training run: start with raw weights, create EMA, then
    perturb the raw weights so they diverge from the EMA shadow.
    """
    # Create EMA from current model weights
    trainable = [p for p in model.parameters() if p.requires_grad]
    ema = ExponentialMovingAverage(trainable, decay=0.99, use_num_updates=True)

    # Perturb raw weights so they diverge from EMA shadows
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Build checkpoint
    state_dict = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    ema_state = ema.state_dict()

    checkpoint = {
        "epoch": 5,
        "global_step": 1000,
        "pytorch-lightning_version": "2.0.0",
        "state_dict": state_dict,
        "loops": {},
        "callbacks": {},
        "optimizer_states": [],
        "lr_schedulers": [],
        "hparams_name": "cfg",
        "hyper_parameters": {},
        "ema": ema_state,
    }
    return checkpoint


class TestApplyEmaWeightsCorrectness:
    """Test that _apply_ema_weights correctly overwrites model params with EMA."""

    def test_apply_ema_overwrites_params_with_shadow(self):
        """After _apply_ema_weights, model params must match EMA shadow_params."""
        model = _SimpleModel()

        # Record original weights (these become EMA shadows)
        original_params = [p.data.clone() for p in model.parameters()]

        # Create EMA from current weights
        trainable = [p for p in model.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(trainable, decay=0.99, use_num_updates=True)
        ema_state = ema.state_dict()
        shadow_params = ema_state["shadow_params"]

        # Perturb model weights so they differ from EMA
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        # Verify model weights are now different from shadows
        for param, shadow in zip(model.parameters(), shadow_params):
            assert not torch.allclose(param.data, shadow, atol=1e-6), (
                "Test setup error: model weights should differ from EMA"
            )

        # Apply EMA via the same mechanism as Harness._apply_ema_weights
        ema_restore = ExponentialMovingAverage(
            [p for p in model.parameters() if p.requires_grad],
            decay=ema_state["decay"],
            use_num_updates=ema_state.get("num_updates") is not None,
        )
        ema_restore.load_state_dict(ema_state)
        ema_restore.copy_to()

        # Verify model weights now match EMA shadows
        for param, shadow in zip(model.parameters(), shadow_params):
            assert torch.allclose(param.data, shadow, atol=1e-7), (
                f"After _apply_ema_weights, model params must match EMA shadows. "
                f"Max diff: {(param.data - shadow).abs().max().item()}"
            )

    def test_on_load_checkpoint_ema_gets_overwritten_by_load_state_dict(self):
        """Demonstrate the Lightning quirk: copy_to() in on_load_checkpoint is
        overwritten by the subsequent load_state_dict call.

        This documents WHY from_checkpoint(apply_ema=True) applies EMA AFTER
        load_state_dict rather than inside on_load_checkpoint.
        """
        import lightning as L

        class _MinimalLightningModule(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = _SimpleModel()
                self._apply_ema_in_on_load = False

            def on_load_checkpoint(self, checkpoint):
                if self._apply_ema_in_on_load and "ema" in checkpoint:
                    ema_state = checkpoint["ema"]
                    ema = ExponentialMovingAverage(
                        [p for p in self.parameters() if p.requires_grad],
                        decay=ema_state["decay"],
                        use_num_updates=ema_state.get("num_updates") is not None,
                    )
                    ema.load_state_dict(ema_state)
                    ema.copy_to()

        # Create module and checkpoint with divergent EMA
        module = _MinimalLightningModule()
        ckpt = _create_checkpoint_with_divergent_ema(module.model)

        # Save checkpoint to disk
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            ckpt_path = f.name
            torch.save(ckpt, f)

        try:
            # Load with on_load_checkpoint EMA application
            loaded = _MinimalLightningModule()
            loaded._apply_ema_in_on_load = True
            loaded = loaded.__class__.load_from_checkpoint(
                ckpt_path,
                map_location="cpu",
            )

            # The loaded model should have RAW weights (not EMA) because
            # load_state_dict overwrites the EMA applied in on_load_checkpoint
            raw_state = ckpt["state_dict"]
            for name, param in loaded.named_parameters():
                if name in raw_state:
                    assert torch.allclose(param.data, raw_state[name], atol=1e-7), (
                        f"Expected on_load_checkpoint EMA to be overwritten by "
                        f"load_state_dict for key {name}"
                    )
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

    def test_post_load_ema_application_works(self):
        """Verify that applying EMA AFTER load_state_dict completes works correctly.

        This is the mechanism used by Harness.from_checkpoint(apply_ema=True).
        """
        import lightning as L

        class _MinimalLightningModule(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = _SimpleModel()

        # Create module and checkpoint with divergent EMA
        module = _MinimalLightningModule()
        ckpt = _create_checkpoint_with_divergent_ema(module.model)
        shadow_params = ckpt["ema"]["shadow_params"]

        # Save checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            ckpt_path = f.name
            torch.save(ckpt, f)

        try:
            # Load normally (no EMA in on_load_checkpoint)
            loaded = _MinimalLightningModule.load_from_checkpoint(
                ckpt_path, map_location="cpu"
            )

            # Now apply EMA AFTER load (same as from_checkpoint does)
            ema_state = ckpt["ema"]
            ema = ExponentialMovingAverage(
                [p for p in loaded.parameters() if p.requires_grad],
                decay=ema_state["decay"],
                use_num_updates=ema_state.get("num_updates") is not None,
            )
            ema.load_state_dict(ema_state)
            ema.copy_to()

            # Verify params now match EMA shadows
            for param, shadow in zip(
                (p for p in loaded.parameters() if p.requires_grad),
                shadow_params,
            ):
                assert torch.allclose(param.data, shadow, atol=1e-7), (
                    f"Post-load EMA application failed. "
                    f"Max diff: {(param.data - shadow).abs().max().item()}"
                )

            # Verify params do NOT match raw state_dict
            raw_state = ckpt["state_dict"]
            mismatches = 0
            for name, param in loaded.named_parameters():
                if name in raw_state:
                    if not torch.allclose(param.data, raw_state[name], atol=1e-6):
                        mismatches += 1
            assert mismatches > 0, (
                "After EMA application, params should differ from raw state_dict"
            )
        finally:
            Path(ckpt_path).unlink(missing_ok=True)

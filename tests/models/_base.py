"""Base test mixin classes for model families.

These mixins define **shared test logic** that every model family must
satisfy.  A concrete test class inherits from one or more mixins and
provides a small set of fixtures that wire the mixin to the specific
model, loss, or predictor under test.

Example -- adding tests for a new model called ``foo``::

    # tests/models/foo/test_model_foo.py
    from tests.models._base import BaseModelTests

    class TestFooModel(BaseModelTests):
        @pytest.fixture()
        def model(self, tiny_foo_model):
            return tiny_foo_model

        @pytest.fixture()
        def run_forward(self, model, simple_tokenizer):
            def _run(batch_size=2, seq_len=16, partial_mask=False):
                x = torch.randint(0, simple_tokenizer.vocab_size, (batch_size, seq_len))
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
                if partial_mask:
                    mask[:, -4:] = False
                positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                return model(x, attention_mask=mask, positions=positions)
            return _run

See ``wiki/running_tests.md`` for the full guide.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Constants shared across all mixin tests
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
SEQ_LEN = 16


# ===================================================================
# region: BaseModelTests
# ===================================================================


class BaseModelTests:
    """Mixin for model (``nn.Module``) tests.

    **Required fixtures** (must be provided by the subclass):

    ``model``
        The ``nn.Module`` under test.

    ``run_forward``
        A callable ``(batch_size=2, seq_len=16, partial_mask=False) -> Tensor``
        that builds input tensors, calls ``model.forward()``, and returns the
        *vocab-logits* tensor of shape ``(batch, seq, vocab)``.

        This is where each model encodes its unique forward signature
        (e.g. MDLM passes ``t``, ARLM uses a 3-D causal mask, ILM
        unpacks a tuple).

    **Provided fixtures** (available automatically via ``conftest``):

    ``simple_tokenizer``
        Used to determine ``vocab_size`` for shape assertions.
    """

    def test_forward_output_shape(self, run_forward, simple_tokenizer):
        logits = run_forward(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
        assert logits.shape == (
            BATCH_SIZE,
            SEQ_LEN,
            simple_tokenizer.vocab_size,
        )

    def test_forward_with_partial_mask(self, run_forward, simple_tokenizer):
        """Model handles an attention mask with trailing padding."""
        logits = run_forward(
            batch_size=BATCH_SIZE, seq_len=SEQ_LEN, partial_mask=True
        )
        assert logits.shape == (
            BATCH_SIZE,
            SEQ_LEN,
            simple_tokenizer.vocab_size,
        )

    def test_gradient_flows(self, run_forward, model):
        """At least one model parameter receives a non-None gradient.

        Previously this only asserted ``logits.requires_grad`` — that is
        trivially true after any non-detached forward, regardless of whether
        gradients actually reach the model. The captured ``model`` fixture
        lets us assert the real invariant.
        """
        # Make sure no stale grads bias the check.
        model.zero_grad(set_to_none=True)
        logits = run_forward(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
        assert logits.requires_grad
        logits.sum().backward()
        grads = [
            p.grad
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads, "no model parameter received a gradient"
        assert any(torch.any(g != 0) for g in grads), (
            "all model parameter gradients are exactly zero"
        )

    def test_weight_decay_param_split(self, model):
        """Weight-decay and no-weight-decay sets are disjoint and exhaustive."""
        wd = {n for n, _ in model.get_named_params_for_weight_decay()}
        no_wd = {n for n, _ in model.get_named_params_for_no_weight_decay()}
        assert wd.isdisjoint(no_wd)
        assert wd | no_wd == {n for n, _ in model.named_parameters()}


# endregion
# ===================================================================


# ===================================================================
# region: BaseLossTests
# ===================================================================


class BaseLossTests:
    """Mixin for loss-function tests.

    **Required fixtures** (must be provided by the subclass):

    ``loss_fn``
        The loss callable (satisfies the ``LossFunction`` protocol).

    ``batch``
        A model-specific batch dict that ``loss_fn`` can consume.
    """

    def test_returns_loss_key(self, loss_fn, batch):
        result = loss_fn(batch)
        assert "loss" in result

    def test_loss_is_scalar(self, loss_fn, batch):
        result = loss_fn(batch)
        assert result["loss"].dim() == 0

    def test_loss_is_finite(self, loss_fn, batch):
        result = loss_fn(batch)
        assert torch.isfinite(result["loss"])

    def test_loss_requires_grad(self, loss_fn, batch):
        result = loss_fn(batch)
        assert result["loss"].requires_grad

    def test_gradients_reach_model(self, loss_fn, batch):
        result = loss_fn(batch)
        result["loss"].backward()
        grads = [
            p.grad
            for p in loss_fn.model.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0


# endregion
# ===================================================================


# ===================================================================
# region: BaseCollatorTests
# ===================================================================


class BaseCollatorTests:
    """Mixin for collator tests.

    **Required fixtures** (must be provided by the subclass):

    ``collator``
        The collator callable.

    ``raw_examples``
        A list of raw ``BaseCollatorInput``-style dicts ready to be
        passed to ``collator()``.
    """

    def test_output_has_target_ids(self, collator, raw_examples):
        batch = collator(raw_examples)
        assert "target_ids" in batch

    def test_output_shapes_consistent(self, collator, raw_examples):
        batch = collator(raw_examples)
        n = len(raw_examples)
        assert batch["input_ids"].shape[0] == n
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        # If the collator declares a ``block_size`` (as all standard
        # train collators do), enforce the upper bound so that
        # block-size truncation behavior is actually exercised.
        block_size = getattr(collator, "block_size", None)
        if block_size is not None:
            assert batch["input_ids"].shape[1] <= block_size


# endregion
# ===================================================================

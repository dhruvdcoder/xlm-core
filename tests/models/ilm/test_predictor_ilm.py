"""Unit tests for the ILM predictor."""

import pytest

from ilm.predictor_ilm import ILMPredictor


class TestILMPredictorConstruction:
    """Construction / argument validation for :class:`ILMPredictor`.

    Full prediction requires a wired model + collated sparse batch +
    an iterative insertion decoding loop; those are covered by the
    integration suite. These tests verify the constructor wiring that
    every other code path depends on.
    """

    def test_construction_with_top_k(
        self, simple_tokenizer, tiny_ilm_model, dummy_noise_schedule
    ):
        pred = ILMPredictor(
            max_steps=2,
            max_length=8,
            tokenizer=simple_tokenizer,
            noise_schedule=dummy_noise_schedule,
            sampling_method="sample_top_k",
            top=5,
            model=tiny_ilm_model,
        )
        assert pred.max_steps == 2
        assert pred.max_length == 8
        assert pred.model is tiny_ilm_model
        assert pred.tokenizer is simple_tokenizer

    def test_construction_with_top_p(
        self, simple_tokenizer, tiny_ilm_model, dummy_noise_schedule
    ):
        pred = ILMPredictor(
            max_steps=2,
            max_length=8,
            tokenizer=simple_tokenizer,
            noise_schedule=dummy_noise_schedule,
            sampling_method="sample_top_p",
            p=0.9,
            model=tiny_ilm_model,
        )
        assert pred.max_steps == 2

    def test_tokenizer_required(self, dummy_noise_schedule):
        with pytest.raises(ValueError, match="tokenizer is required"):
            ILMPredictor(
                max_steps=2,
                max_length=8,
                tokenizer=None,
                noise_schedule=dummy_noise_schedule,
            )

    def test_invalid_sampling_method_rejected(
        self, simple_tokenizer, dummy_noise_schedule
    ):
        with pytest.raises(ValueError, match="Invalid sampling method"):
            ILMPredictor(
                max_steps=2,
                max_length=8,
                tokenizer=simple_tokenizer,
                noise_schedule=dummy_noise_schedule,
                sampling_method="bogus",  # type: ignore[arg-type]
            )

    def test_invalid_second_sampling_method_rejected(
        self, simple_tokenizer, dummy_noise_schedule
    ):
        with pytest.raises(
            ValueError, match="Invalid second sampling method"
        ):
            ILMPredictor(
                max_steps=2,
                max_length=8,
                tokenizer=simple_tokenizer,
                noise_schedule=dummy_noise_schedule,
                second_sampling_method="bogus",  # type: ignore[arg-type]
            )

    def test_token_ids_to_suppress_default(
        self, simple_tokenizer, tiny_ilm_model, dummy_noise_schedule
    ):
        """By default mask, eos, cls, bos token ids are suppressed."""
        pred = ILMPredictor(
            max_steps=2,
            max_length=8,
            tokenizer=simple_tokenizer,
            noise_schedule=dummy_noise_schedule,
            model=tiny_ilm_model,
        )
        suppressed = set(pred.token_ids_to_suppress.tolist())
        expected = {
            simple_tokenizer.mask_token_id,
            simple_tokenizer.eos_token_id,
            simple_tokenizer.cls_token_id,
            simple_tokenizer.bos_token_id,
        }
        assert expected.issubset(suppressed)

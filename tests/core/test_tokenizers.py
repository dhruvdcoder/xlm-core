"""Unit tests for tokenizer implementations in ``xlm.datamodule``."""

import pytest

from xlm.datamodule import SimpleSpaceTokenizer


class TestSimpleSpaceTokenizer:
    """Tests for :class:`SimpleSpaceTokenizer`."""

    def test_vocab_size(self, simple_tokenizer, small_vocab):
        # 7 special tokens + len(small_vocab) content tokens
        assert simple_tokenizer.vocab_size == 7 + len(small_vocab)

    def test_special_token_ids(self, simple_tokenizer):
        assert simple_tokenizer.pad_token_id == 0
        assert simple_tokenizer.cls_token_id == 1
        assert simple_tokenizer.mask_token_id == 2
        assert simple_tokenizer.eos_token_id == 3
        assert simple_tokenizer.bos_token_id == 4

    def test_encode_decode_roundtrip(self, simple_tokenizer):
        text = "0 1 2 3"
        token_ids = simple_tokenizer.encode(text, add_special_tokens=False)
        decoded = simple_tokenizer.decode(token_ids, skip_special_tokens=False)
        assert decoded.strip() == text

    def test_tokenize_splits_on_space(self, simple_tokenizer):
        tokens = simple_tokenizer.tokenize("10 20 30")
        assert tokens == ["10", "20", "30"]

    def test_unknown_token_raises(self, simple_tokenizer):
        with pytest.raises(ValueError, match="not in vocab"):
            simple_tokenizer.tokenize("NOT_IN_VOCAB")

    def test_for_numbers_factory(self):
        tok = SimpleSpaceTokenizer.for_numbers(10)
        # 7 special + 10 number tokens
        assert tok.vocab_size == 17


class TestSimpleSpaceTokenizerHypothesis:
    """Property-based tests (requires ``hypothesis``)."""

    @pytest.mark.slow
    def test_encode_decode_roundtrip_property(self, simple_tokenizer):
        """Encode-decode round-trip holds for arbitrary in-vocab sequences."""
        from hypothesis import given, settings
        import hypothesis.strategies as st

        vocab_tokens = [str(i) for i in range(50)]
        token_seq = st.lists(
            st.sampled_from(vocab_tokens), min_size=1, max_size=20
        )

        @given(tokens=token_seq)
        @settings(max_examples=50)
        def _check(tokens):
            text = " ".join(tokens)
            ids = simple_tokenizer.encode(text, add_special_tokens=False)
            roundtrip = simple_tokenizer.decode(
                ids, skip_special_tokens=False
            )
            assert roundtrip.strip() == text

        _check()

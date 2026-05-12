"""Unit tests for :mod:`xlm.utils.text`."""

from xlm.utils.text import remove_trailing_pads


class TestRemoveTrailingPads:
    def test_no_trailing_pad_is_passthrough(self, simple_tokenizer):
        text = "hello world"
        assert remove_trailing_pads(text, simple_tokenizer) == "hello world"

    def test_strips_single_trailing_pad_with_space(self, simple_tokenizer):
        pad = simple_tokenizer.pad_token  # "[PAD]"
        text = f"hello {pad}"
        assert remove_trailing_pads(text, simple_tokenizer) == "hello"

    def test_strips_multiple_trailing_pads_with_space(self, simple_tokenizer):
        pad = simple_tokenizer.pad_token
        text = f"hello {pad} {pad} {pad}"
        assert remove_trailing_pads(text, simple_tokenizer) == "hello"

    def test_strips_pads_when_no_leading_space(self, simple_tokenizer):
        pad = simple_tokenizer.pad_token
        # First check the falls-back-to-no-space branch.
        text = f"hi{pad}{pad}"
        assert remove_trailing_pads(text, simple_tokenizer) == "hi"

    def test_empty_string_passthrough(self, simple_tokenizer):
        assert remove_trailing_pads("", simple_tokenizer) == ""

    def test_only_pads(self, simple_tokenizer):
        pad = simple_tokenizer.pad_token
        text = f"{pad}{pad}{pad}"
        assert remove_trailing_pads(text, simple_tokenizer) == ""

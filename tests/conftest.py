"""Root conftest.py -- shared fixtures for the entire test suite."""

import pytest
import torch

from xlm.datamodule import SimpleSpaceTokenizer
from xlm.noise import DummyNoiseSchedule


# ---------------------------------------------------------------------------
# Tokenizer fixtures
# ---------------------------------------------------------------------------

# A small vocabulary that is reused across many tests.
_SMALL_VOCAB = [str(i) for i in range(50)]


@pytest.fixture()
def small_vocab():
    """A list of 50 token strings (\"0\" .. \"49\")."""
    return list(_SMALL_VOCAB)


@pytest.fixture()
def simple_tokenizer(small_vocab):
    """A :class:`SimpleSpaceTokenizer` built from *small_vocab*.

    Special-token layout (see ``SimpleSpaceTokenizer.__init__``):

    * ``[PAD]`` = 0, ``[CLS]`` = 1, ``[MASK]`` = 2, ``[EOS]`` = 3,
      ``[BOS]`` = 4, ``[SEP]`` = 5, ``[UNK]`` = 6
    * Vocab tokens start at id 7.

    Total vocabulary size = 7 (special) + 50 (vocab) = 57.
    """
    return SimpleSpaceTokenizer(vocab=small_vocab)


# ---------------------------------------------------------------------------
# Noise schedule fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_noise_schedule():
    """A :class:`DummyNoiseSchedule` (raises on every method)."""
    return DummyNoiseSchedule()


# ---------------------------------------------------------------------------
# Batch / tensor helpers
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_SIZE = 4


@pytest.fixture(params=[16, 32], ids=["block16", "block32"])
def block_size(request):
    """Parametrised block sizes useful for collator / model tests."""
    return request.param


@pytest.fixture()
def batch_size():
    return _DEFAULT_BATCH_SIZE


@pytest.fixture()
def random_input_ids(simple_tokenizer, batch_size, block_size):
    """Random ``input_ids`` tensor of shape ``(batch_size, block_size)``."""
    vocab_size = simple_tokenizer.vocab_size
    return torch.randint(0, vocab_size, (batch_size, block_size))


@pytest.fixture()
def ones_attention_mask(batch_size, block_size):
    """All-ones attention mask (no padding)."""
    return torch.ones(batch_size, block_size, dtype=torch.long)


@pytest.fixture()
def zeros_token_type_ids(batch_size, block_size):
    """All-zeros token_type_ids."""
    return torch.zeros(batch_size, block_size, dtype=torch.long)


@pytest.fixture()
def base_batch(random_input_ids, ones_attention_mask, zeros_token_type_ids):
    """A minimal ``BaseBatch``-compatible dict."""
    return {
        "input_ids": random_input_ids,
        "attention_mask": ones_attention_mask,
        "token_type_ids": zeros_token_type_ids,
    }


# ---------------------------------------------------------------------------
# Tiny model hyper-parameter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_model_kwargs(simple_tokenizer):
    """Minimal kwargs shared by RotaryTransformer*Model constructors.

    Creates a model with ~10 k parameters -- fast enough for CI.
    """
    return dict(
        num_embeddings=simple_tokenizer.vocab_size,
        d_model=32,
        num_layers=1,
        nhead=2,
        dropout=0.0,
        rotary_emb_dim=16,
        max_length=64,
    )

"""Root conftest.py -- shared fixtures for the entire test suite."""

from pathlib import Path
from typing import Any, Callable, Dict

import datasets
import pytest
import torch

from tests.datamodule_helpers import (
    IdTrackingCollator,
    build_inmem_datasets,
    make_patched_download,
)
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


@pytest.fixture()
def real_loglinear_schedule():
    """A real :class:`ContinuousTimeLogLinearSchedule` for tests.

    Many collators and predictors (``DefaultMDLMCollator``, ``MDLMPredictor``,
    ``DefaultILMCollator``, etc.) call ``noise_schedule.sample_t`` and
    ``noise_schedule(t)`` during collation/prediction, and the catch-all
    ``DummyNoiseSchedule`` raises on every call. This fixture builds a tiny,
    deterministic-shaped, real schedule that any of those tests can use.

    Note: ``ContinuousTimeLogLinearSchedule.__init__`` raises
    ``NotImplementedError`` if ``sigma_min > 0`` (verified by
    ``tests/models/mdlm/test_noise_mdlm.py::TestContinuousTimeLogLinearSchedule
    ::test_sigma_min_positive_not_implemented``), so we use ``sigma_min=0.0``.
    """
    from mdlm.noise_mdlm import ContinuousTimeLogLinearSchedule

    return ContinuousTimeLogLinearSchedule(
        sigma_min=0.0,
        sigma_max=4.0,
        antithetic_sampling=True,
        eps=1e-3,
    )


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


# ---------------------------------------------------------------------------
# DatasetManager fixtures
# ---------------------------------------------------------------------------
#
# Used by the fast single-method tests in ``tests/core/test_datamodule.py``
# *and* by the multi-process / SLURM tests in ``tests/integration/``.
# All datasets are in-memory; the patched ``_download`` makes sure no
# network I/O ever happens during testing.  See
# :mod:`tests.datamodule_helpers` for the registry contents and the
# ``id`` numbering scheme.


@pytest.fixture(scope="session")
def inmem_datasets() -> Dict[str, datasets.Dataset]:
    """Canonical in-memory datasets keyed by ``"<repo>/<ds>/<split>"``."""
    return build_inmem_datasets()


@pytest.fixture()
def patched_download(
    monkeypatch: pytest.MonkeyPatch,
    inmem_datasets: Dict[str, datasets.Dataset],
) -> Dict[str, datasets.Dataset]:
    """Monkey-patch ``DatasetManager._download`` to read from memory.

    The returned dict is a per-test copy of :func:`inmem_datasets`; tests
    may register additional entries on it before constructing a
    ``DatasetManager``.
    """
    registry: Dict[str, datasets.Dataset] = dict(inmem_datasets)
    monkeypatch.setattr(
        "xlm.datamodule.DatasetManager._download",
        make_patched_download(registry),
        raising=True,
    )
    return registry


@pytest.fixture()
def manual_cache_dir(tmp_path: Path) -> Path:
    """A clean per-test directory for the DatasetManager manual cache."""
    cache = tmp_path / "manual_cache"
    cache.mkdir()
    return cache


@pytest.fixture()
def result_dir(tmp_path: Path) -> Path:
    """A clean per-test directory for distributed-runner result files."""
    out = tmp_path / "results"
    out.mkdir()
    return out


@pytest.fixture()
def simple_collator(simple_tokenizer) -> IdTrackingCollator:
    """A collator that stacks fixed-length tensors and tracks example ids."""
    return IdTrackingCollator(tokenizer=simple_tokenizer)


_PREPROCESS_FN = "tests.datamodule_helpers.example_to_input_ids"


@pytest.fixture()
def dataset_manager_factory(
    simple_tokenizer,
    simple_collator: IdTrackingCollator,
    patched_download: Dict[str, datasets.Dataset],
    manual_cache_dir: Path,
) -> Callable[..., Any]:
    """Return a callable building a ``DatasetManager`` with sensible defaults.

    Default config:

    * reads the in-memory dataset at ``mem/raw/train`` via the patched
      ``_download`` (no network);
    * runs :func:`tests.datamodule_helpers.example_to_input_ids` as the
      preprocess function with ``columns_to_keep=["id"]`` so the unique
      ``id`` survives the ``ds.map(remove_columns=...)`` step;
    * uses ``batch_size=2``, ``num_workers=0`` (no multiprocessing in the
      dataloader -- the multi-proc *runner* tests override this);
    * disables shuffle in dataloader_kwargs (the train-DDP map-style path
      attaches its own sampler and rejects ``shuffle=True``).

    Pass any kwarg accepted by :class:`xlm.datamodule.DatasetManager` to
    override.  ``manual_cache_dir`` is **not** a constructor kwarg; tests
    receive it via the :func:`manual_cache_dir` fixture and pass it to
    ``prepare_data``/``setup`` themselves.
    """
    from xlm.datamodule import DatasetManager

    def _make(
        full_name: str = "mem/raw/train",
        **overrides: Any,
    ):
        defaults: Dict[str, Any] = dict(
            collator=simple_collator,
            full_name=full_name,
            full_name_debug=full_name,
            dataloader_kwargs={
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
            preprocess_function=_PREPROCESS_FN,
            columns_to_keep=["id"],
        )
        defaults.update(overrides)
        return DatasetManager(**defaults)

    return _make

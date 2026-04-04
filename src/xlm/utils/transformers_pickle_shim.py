"""Map legacy pickle paths used in older HuggingFace ``transformers`` checkpoints.

Some Lightning checkpoints reference ``transformers.tokenization_python.Trie``;
current ``transformers`` defines ``Trie`` in ``tokenization_utils`` instead.
Without this shim, ``torch.load`` / ``load_from_checkpoint`` can raise
``ModuleNotFoundError: No module named 'transformers.tokenization_python'``.
"""

from __future__ import annotations

import sys
import types


def install() -> None:
    """Register ``transformers.tokenization_python`` if missing (for unpickling)."""
    if "transformers.tokenization_python" in sys.modules:
        return
    try:
        from transformers.tokenization_utils import Trie
    except ImportError:
        return
    mod = types.ModuleType("transformers.tokenization_python")
    mod.Trie = Trie
    sys.modules["transformers.tokenization_python"] = mod

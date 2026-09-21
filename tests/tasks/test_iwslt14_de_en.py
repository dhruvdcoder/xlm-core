"""Unit tests for IWSLT14 DE→EN preprocess (raw BPE ids, no specials)."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from xlm.tasks.iwslt14_de_en import (
    TOKENIZER_V1_DIR,
    iwslt14_mt_preprocess_fn,
    joint_char_bpe_v1_dir,
)

_EXPECTED_TOKENIZER_JSON_SHA256 = (
    "975aedcb168bc1010ee3c67e28e8bf6242f2891e4a080c618782170ffe8f5e78"
)


class _MockTokenizer:
    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3
    mask_token_id = 4

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        assert add_special_tokens is False, (
            "preprocess must call encode with add_special_tokens=False"
        )
        # Deterministic non-special ids (>= 5)
        return [5 + (ord(c) % 20) for c in text]


def test_iwslt14_mt_preprocess_fn_fields() -> None:
    tok = _MockTokenizer()
    row: Dict[str, Any] = {
        "id": "ex-1",
        "source_text": "hallo welt",
        "target_text": "hello world",
    }
    out = iwslt14_mt_preprocess_fn(row, tok)
    assert out["prompt_token_ids"] == tok.encode(
        "hallo welt", add_special_tokens=False
    )
    assert out["input_token_ids"] == tok.encode(
        "hello world", add_special_tokens=False
    )


def test_iwslt14_mt_preprocess_fn_no_special_tokens() -> None:
    tok = _MockTokenizer()
    specials = {
        tok.pad_token_id,
        tok.unk_token_id,
        tok.bos_token_id,
        tok.eos_token_id,
        tok.mask_token_id,
    }
    row: Dict[str, Any] = {
        "source_text": "dies ist ein test",
        "target_text": "this is a test",
    }
    out = iwslt14_mt_preprocess_fn(row, tok)
    for field in ("prompt_token_ids", "input_token_ids"):
        ids = out[field]
        assert ids, f"{field} should be non-empty"
        assert ids[0] not in specials
        assert ids[-1] not in specials
        assert not specials.intersection(ids)


def test_iwslt14_mt_preprocess_fn_empty_sides() -> None:
    tok = _MockTokenizer()
    out = iwslt14_mt_preprocess_fn({}, tok)
    assert out["prompt_token_ids"] == []
    assert out["input_token_ids"] == []


def test_packaged_joint_char_bpe_v1_sha256(monkeypatch) -> None:
    monkeypatch.delenv("IWSLT14_JOINT_CHAR_BPE_V1", raising=False)
    path = TOKENIZER_V1_DIR / "tokenizer.json"
    assert path.is_file()
    assert joint_char_bpe_v1_dir() == str(TOKENIZER_V1_DIR)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == _EXPECTED_TOKENIZER_JSON_SHA256

"""Unit tests for gemma-usdm corruption, loss, buffer, and the sampler."""

from __future__ import annotations

import torch
import torch.nn as nn

from gemma_usdm.datamodule_gemma_usdm import GemmaUsdmTrainCollator
from gemma_usdm.loss_gemma_usdm import GemmaUsdmLoss
from gemma_usdm.model_gemma_usdm import GemmaUsdmModel
from gemma_usdm.noise_gemma_usdm import corrupt_canvas
from gemma_usdm.predictor_gemma_usdm import (
    GemmaUsdmPredictor,
    canvas_stop_mask,
    entropy_bounded_commit,
)


class _Tok:
    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3
    mask_token_id = 4

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


def _tiny_model() -> GemmaUsdmModel:
    return GemmaUsdmModel(
        num_embeddings=16,
        d_model=32,
        num_layers=1,
        nhead=4,
        padding_idx=0,
        mask_idx=1,
        dim_feedforward=64,
        dropout=0.0,
        rotary_emb_dim=8,
        max_length=32,
    )


def test_replacement_rate_tracks_t_and_source_is_clean():
    torch.manual_seed(0)
    vocab = 12
    pad_id = 0
    batch, length = 64, 400
    # Canvas starts as a single non-pad id, so a redraw of that same id is the
    # only way a replacement fails to change the token.
    clean = torch.full((batch, length), 4)
    canvas = torch.zeros(batch, length, dtype=torch.bool)
    canvas[:, 20:] = True
    t_value = 0.3
    noisy = corrupt_canvas(
        clean, canvas, torch.full((batch,), t_value), vocab, pad_id
    )
    assert torch.equal(noisy[:, :20], clean[:, :20])
    changed = noisy[:, 20:] != clean[:, 20:]
    # vocab-1 legal draws, one of which equals the original token.
    expected = t_value * (vocab - 2) / (vocab - 1)
    assert abs(changed.float().mean().item() - expected) < 0.015
    drawn = noisy[canvas & (noisy != clean)]
    assert int(drawn.min()) >= 1
    assert int(drawn.max()) < vocab
    assert not (noisy == pad_id).any()


def test_t0_leaves_canvas_equal_to_target():
    clean = torch.tensor([[5, 6, 7, 8, 0]])
    canvas = torch.tensor([[False, False, True, True, True]])
    noisy = corrupt_canvas(
        clean, canvas, torch.zeros(1), vocab_size=10, pad_id=0
    )
    assert torch.equal(noisy, clean)


def test_loss_on_padding_targets_and_mean():
    examples = [{"prompt_ids": [5, 6], "input_ids": [7, 8]}]
    true_batch = GemmaUsdmTrainCollator(
        _Tok(), block_size=4, input_block_size=4, loss_on_padding=True
    )(examples)
    false_batch = GemmaUsdmTrainCollator(
        _Tok(), block_size=4, input_block_size=4, loss_on_padding=False
    )(examples)
    # prefix (2) + canvas [7, 8, EOS, pad] + tail pad to input_block+block.
    assert true_batch["target_ids"][0].tolist() == [
        -100, -100, 7, 8, 3, 0, -100, -100,
    ]
    assert false_batch["target_ids"][0].tolist() == [
        -100, -100, 7, 8, 3, -100, -100, -100,
    ]
    assert true_batch["canvas_mask"][0].tolist() == [
        False, False, True, True, True, True, False, False,
    ]

    long = GemmaUsdmTrainCollator(
        _Tok(), block_size=4, input_block_size=3, loss_on_padding=False
    )([{"prompt_ids": [1, 2, 3, 4, 5], "input_ids": [6, 7, 8, 9, 4, 5]}])
    # Left-truncated prefix keeps [3, 4, 5]. Canvas keeps suffix[:3] + EOS.
    assert long["input_ids"][0, :3].tolist() == [3, 4, 5]
    assert long["input_ids"][0, 3:7].tolist() == [6, 7, 8, 3]

    logits = torch.zeros(10)
    logits[7] = 1.0
    logits[8] = 2.0
    logits[3] = 4.0

    class Const(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 4, padding_idx=0)
            self.padding_idx = 0
            self.force_flash_attn = False

        def forward(self, x_t, attention_mask=None, positions=None, self_cond=None, block_mask=None):
            del attention_mask, positions, self_cond, block_mask
            return logits.view(1, 1, -1).expand(x_t.shape[0], x_t.shape[1], -1).clone()

    model = Const()
    model.eval()
    loss_obj = GemmaUsdmLoss(model=model, clean_prob=1.0)
    got_true = loss_obj(true_batch)["loss"]
    got_false = loss_obj(false_batch)["loss"]

    log_p = torch.log_softmax(logits, dim=-1)

    def nll(token: int) -> torch.Tensor:
        return -log_p[token]

    expect_true = torch.stack([nll(7), nll(8), nll(3), nll(0)]).mean()
    expect_false = torch.stack([nll(7), nll(8), nll(3)]).mean()
    assert torch.allclose(got_true, expect_true)
    assert torch.allclose(got_false, expect_false)
    assert not torch.allclose(got_false, nll(7))
    assert loss_obj.buffer.rows is None


def test_entropy_budget_commits_lowest_and_renoises_the_rest():
    tokens = torch.tensor([[10, 11, 12]])
    entropy = torch.tensor([[0.08, 0.08, 0.5]])
    canvas = torch.ones(1, 3, dtype=torch.bool)
    sampled = torch.tensor([[1, 2, 3]])
    uniform = torch.tensor([[7, 8, 9]])
    updated, commit = entropy_bounded_commit(
        tokens, entropy, canvas, sampled, uniform, budget=0.1
    )
    assert commit.tolist() == [[True, True, False]]
    assert updated.tolist() == [[1, 2, 9]]

    # A non-canvas slot is never committed or redrawn, even with low entropy.
    tokens = torch.tensor([[4, 10, 11, 12]])
    entropy = torch.tensor([[0.0, 0.08, 0.08, 0.5]])
    canvas = torch.tensor([[False, True, True, True]])
    sampled = torch.tensor([[1, 2, 3, 4]])
    uniform = torch.tensor([[6, 7, 8, 9]])
    updated, commit = entropy_bounded_commit(
        tokens, entropy, canvas, sampled, uniform, budget=0.1
    )
    assert commit.tolist() == [[False, True, True, False]]
    assert updated.tolist() == [[4, 2, 3, 9]]


def test_stop_mask_ignores_post_eos_when_padding_is_unsupervised():
    argmax = torch.tensor([[5, 3, 4, 9]])
    canvas = torch.tensor([[False, True, True, True]])
    masked = canvas_stop_mask(argmax, canvas, loss_on_padding=False, eos_id=3)
    assert masked.tolist() == [[False, True, False, False]]
    full = canvas_stop_mask(argmax, canvas, loss_on_padding=True, eos_id=3)
    assert full.tolist() == [[False, True, True, True]]


class _ScriptedDenoiser(nn.Module):
    """Row 0 is confident and stable. Row 1 stays high-entropy and moves."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.ffw = nn.Linear(4, 4)
        self.calls: list[torch.Tensor] = []
        self.n = 0

    def self_conditioning(self, soft, canvas_mask):
        z = self.ffw(soft)
        return z * canvas_mask.to(dtype=z.dtype).unsqueeze(-1)

    def forward(self, x_t, attention_mask=None, positions=None, self_cond=None, block_mask=None):
        del attention_mask, positions, self_cond, block_mask
        self.calls.append(x_t.detach().clone())
        logits = torch.zeros(x_t.shape[0], x_t.shape[1], 8)
        logits[0, :, 2] = 50
        peak = (self.n % 7) + 1
        logits[1, :, peak] = 0.2
        self.n += 1
        return logits


def test_per_sequence_nfe_and_frozen_tokens():
    model = _ScriptedDenoiser()
    predictor = GemmaUsdmPredictor(
        max_steps=4,
        max_new_tokens=3,
        entropy_budget=0.1,
        stop_entropy=0.005,
        loss_on_padding=True,
        tokenizer=_Tok(),
        model=model,
    )
    batch = {
        "input_ids": torch.tensor([[5, 6], [5, 6]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    preds = predictor.predict(batch)
    assert preds["steps_taken"].tolist() == [2, 4]
    assert preds["nfe"] == [2, 4]
    assert torch.equal(model.calls[1][0], model.calls[2][0])
    assert torch.equal(model.calls[2][0], model.calls[3][0])
    assert (preds["ids"][0, 2:] == 2).all()
    rows = predictor.to_dict(batch, preds)
    assert rows[0]["nfe"] == 2
    assert rows[1]["nfe"] == 4
    assert rows[0]["generated_text"] != ""


def test_self_conditioning_is_zero_on_the_source():
    model = _tiny_model()
    model.self_cond_ffw[-1].bias.data.fill_(1.0)
    soft = torch.randn(2, 6, model.embed_tokens.embedding_dim)
    canvas = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, True, True, True, True, False],
        ]
    )
    z = model.self_conditioning(soft, canvas)
    assert z.shape[-1] == model.embed_tokens.embedding_dim
    assert torch.count_nonzero(z[~canvas]) == 0
    assert torch.count_nonzero(z[canvas]) > 0


def _batch(token: int, batch: int = 2, seq: int = 6) -> dict:
    input_ids = torch.full((batch, seq), token, dtype=torch.long)
    input_ids[:, :2] = 5
    canvas = torch.zeros(batch, seq, dtype=torch.bool)
    canvas[:, 2:] = True
    target = input_ids.clone()
    target[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch, seq, dtype=torch.bool),
        "target_ids": target,
        "canvas_mask": canvas,
    }


def test_buffer_replays_once_then_evicts_and_trains_ffw():
    torch.manual_seed(0)
    model = _tiny_model()
    model.train()
    loss_obj = GemmaUsdmLoss(model=model, clean_prob=0.0)
    first = _batch(7)
    second = _batch(8)
    third = _batch(9)

    loss_obj(first)
    assert loss_obj.last_fresh_rows == 2
    assert loss_obj.last_forward_rows == 2
    ids1 = loss_obj.last_forward_input_ids.clone()
    soft1 = loss_obj.buffer.rows["soft"].clone()
    assert not soft1.requires_grad

    out2 = loss_obj(second)
    ids2 = loss_obj.last_forward_input_ids.clone()
    assert loss_obj.last_forward_rows == 4
    assert torch.equal(ids2[2:], ids1)
    assert torch.equal(loss_obj.last_replay_soft, soft1)
    assert not loss_obj.last_replay_soft.requires_grad
    out2["loss"].backward()
    grad = model.self_cond_ffw[-1].weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0

    loss_obj(third)
    ids3 = loss_obj.last_forward_input_ids
    assert torch.equal(ids3[2:], ids2[:2])
    assert not torch.equal(ids3[2:], ids1)

    model.eval()
    loss_obj(_batch(4))
    assert loss_obj.last_forward_rows == 2
    assert loss_obj.buffer.rows is None
    model.train()
    loss_obj(_batch(4))
    assert loss_obj.last_forward_rows == 2

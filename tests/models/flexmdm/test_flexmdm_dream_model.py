"""FlexMDMDreamModel: construction, extras remap and parity with the reference wrapper.

The parity test imports ``flexmdm.architecture.FlexMDM`` from the public
``genuine-any-order/FlexMDM`` release. Point ``FLEXMDM_REF_ROOT`` at that
directory (the one containing the ``flexmdm/`` package) to enable it; it is
skipped otherwise.
"""

import pytest
import torch

from flexmdm.configuration_flexmdm_dream import FlexMDMDreamConfig
from flexmdm.model_flexmdm_dream import (
    FlexMDMDreamModel,
    sinusoidal_timestep_embedding,
)
from xlm.backbones.dream.configuration_base import DreamConfigBase
from xlm.backbones.dream.modeling_dream import DreamModelCore

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=128,
    mask_token_id=3,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=0,
    use_cache=False,
    tie_word_embeddings=False,
)


def _tiny_flexmdm_dream(seed: int = 0) -> FlexMDMDreamModel:
    torch.manual_seed(seed)
    config = FlexMDMDreamConfig(**TINY)
    config._attn_implementation = "sdpa"
    return FlexMDMDreamModel(config, max_length=16).eval()


def _random_batch(B=2, L=10, vocab=64, pad=0, mask=3):
    torch.manual_seed(1)
    x = torch.randint(4, vocab, (B, L))
    x[0, 3] = mask
    x[1, 5] = mask
    x[1, 8:] = pad
    attention_mask = x.ne(pad)
    t = torch.tensor([0.25, 0.8])
    return x, t, attention_mask


def test_construct_and_forward_shapes():
    model = _tiny_flexmdm_dream()
    x, t, attention_mask = _random_batch()
    logits, log_rate = model(x, t, attention_mask)
    assert logits.shape == (2, 10, TINY["vocab_size"])
    assert log_rate.shape == (2, 10)
    assert torch.isfinite(logits).all() and torch.isfinite(log_rate).all()
    # log-rate head is clamped to [-15, 15]
    assert log_rate.min() >= -15 and log_rate.max() <= 15


def test_fresh_modulations_are_identity():
    """Zero-init AdaLN => output independent of t at init (reference behaviour)."""
    model = _tiny_flexmdm_dream()
    x, _, attention_mask = _random_batch()
    l0, _ = model(x, torch.tensor([0.1, 0.1]), attention_mask)
    l1, _ = model(x, torch.tensor([0.9, 0.9]), attention_mask)
    torch.testing.assert_close(l0, l1)
    for layer in model.model.layers:
        assert torch.all(layer.rms_norm_modulations.modulation.weight == 0)
        assert torch.all(layer.rms_norm_modulations.modulation.bias == 0)
    assert torch.all(model.insertion_head.proj2.bias == -1.0)


def test_extras_roundtrip():
    model = _tiny_flexmdm_dream()
    # perturb modulations so the roundtrip is non-trivial
    for layer in model.model.layers:
        layer.rms_norm_modulations.modulation.weight.data.normal_()
    extras = FlexMDMDreamModel.to_extras_state_dict(model.state_dict())
    assert set(k.split(".")[0] for k in extras) == {
        "insertion_head",
        "time_mlp",
        "temb_mods",
    }
    assert "temb_mods.0.weight" in extras and "temb_mods.1.bias" in extras
    other = _tiny_flexmdm_dream(seed=123)
    other.load_extra_state_dict(extras, strict=True)
    for k, v in FlexMDMDreamModel.to_extras_state_dict(other.state_dict()).items():
        torch.testing.assert_close(v, extras[k])


def test_param_groups_split():
    model = _tiny_flexmdm_dream()
    model.insertion_lr_multiplier = 2.0
    model.base_lr = 1e-5
    groups = model.get_param_groups()
    assert len(groups) == 2 and groups[1]["lr"] == pytest.approx(2e-5)
    n_ins = sum(p.numel() for p in groups[1]["params"])
    n_expected = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ("insertion_head" in n or "time_mlp" in n or "rms_norm_modulations" in n)
    )
    assert n_ins == n_expected
    n_total = sum(p.numel() for p in model.parameters())
    assert n_total == n_ins + sum(p.numel() for p in groups[0]["params"])


def test_sinusoidal_embedding_matches_reference_formula():
    t = torch.tensor([0.0, 0.37, 1.0])
    emb = sinusoidal_timestep_embedding(t, 32, time_scale=1000.0, max_period=10000)
    assert emb.shape == (3, 32)
    torch.testing.assert_close(emb[0, :16], torch.ones(16))  # cos(0)
    torch.testing.assert_close(emb[0, 16:], torch.zeros(16))  # sin(0)


# ---------------------------------------------------------------------------
# Parity with the reference wrapper (genuine-any-order/FlexMDM)
# ---------------------------------------------------------------------------


def _load_reference_architecture():
    from .ref_utils import import_reference_submodule

    return import_reference_submodule("architecture")


def test_parity_with_reference_wrapper():
    ref = _load_reference_architecture()

    ours = _tiny_flexmdm_dream()
    # Plain Dream backbone (no time conditioning) wrapped by the reference FlexMDM.
    plain_cfg = DreamConfigBase(**TINY, use_time_embedding=False)
    plain_cfg._attn_implementation = "sdpa"
    torch.manual_seed(7)
    backbone = DreamModelCore(plain_cfg).eval()
    ref_model = ref.FlexMDM(backbone=backbone, pad_token_id=0, max_length=16).eval()

    # Randomize the FlexMDM additions on our side (modulations are zero at init),
    # then copy every weight into the reference layout.
    torch.manual_seed(11)
    for layer in ours.model.layers:
        layer.rms_norm_modulations.modulation.weight.data.normal_(std=0.05)
        layer.rms_norm_modulations.modulation.bias.data.normal_(std=0.05)
    ours_sd = ours.state_dict()
    ref_sd = {}
    for k, v in ours_sd.items():
        if ".rms_norm_modulations.modulation." in k:
            layer_idx = k.split(".")[2]
            param = k.rsplit(".", 1)[1]
            ref_sd[f"temb_mods.{layer_idx}.{param}"] = v
        elif k.startswith(("insertion_head.", "time_mlp.")):
            ref_sd[k] = v
        else:
            ref_sd[f"backbone.{k}"] = v
    missing, unexpected = ref_model.load_state_dict(ref_sd, strict=False)
    # rotary inv_freq buffers are non-persistent on one side or the other
    assert not [k for k in missing if "inv_freq" not in k], missing
    assert not [k for k in unexpected if "inv_freq" not in k], unexpected

    x, t, attention_mask = _random_batch()
    with torch.no_grad():
        logits, log_rate = ours(x, t, attention_mask)
        out = ref_model(x, t, attention_mask=attention_mask)
    torch.testing.assert_close(logits, out["logits"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(log_rate, out["log_length"], atol=1e-5, rtol=1e-5)
    # AdaLN is actually active (sanity: t changes the output)
    with torch.no_grad():
        logits2, _ = ours(x, torch.tensor([0.9, 0.1]), attention_mask)
    assert not torch.allclose(logits, logits2)

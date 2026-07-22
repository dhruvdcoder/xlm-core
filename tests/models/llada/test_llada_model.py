"""CPU tests for the LLaDA backbone + xlm-models adapter.

Run the Hub checkpoint parity test (downloads ~16GB, needs ~35GB RAM or a GPU)
with::

    XLM_RUN_HUB_TESTS=1 pytest tests/models/llada/test_llada_model.py -k hub
"""

import os

import pytest
import torch


def tiny_config(**overrides):
    from llada import LLaDAConfig

    kwargs = dict(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_hidden_size=128,
        activation_type="silu",
        block_type="llama",
        rope=True,
        rope_theta=500000.0,
        layer_norm_type="rms",
        rms_norm_eps=1e-05,
        weight_tying=False,
        include_bias=False,
        include_qkv_bias=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        residual_dropout=0.0,
        max_sequence_length=64,
        vocab_size=100,
        embedding_size=128,
        mask_token_id=99,
        pad_token_id=98,
        eos_token_id=98,
        use_cache=False,
        init_device="cpu",
    )
    kwargs.update(overrides)
    return LLaDAConfig(**kwargs)


@pytest.fixture()
def tiny_llada_model():
    from llada import LLaDAXLMModel

    torch.manual_seed(0)
    model = LLaDAXLMModel(tiny_config(), init_params=True)
    model.eval()
    return model


class TestLLaDAConfig:
    def test_model_config_fields_survive_roundtrip(self):
        """transformers >= 5 drops some config kwargs; the base __init__ must restore them."""
        from dataclasses import fields

        from xlm.backbones.llada.configuration_llada import ModelConfig

        cfg = tiny_config()
        for f in fields(ModelConfig):
            assert hasattr(cfg, f.name), f"missing ModelConfig field: {f.name}"

    def test_model_config_roundtrip_into_inner_model(self):
        from xlm.backbones.llada.modeling_llada import (
            create_model_config_from_pretrained_config,
        )

        mc = create_model_config_from_pretrained_config(tiny_config())
        assert mc.d_model == 64
        assert mc.n_kv_heads is None or isinstance(mc.n_kv_heads, int)
        assert mc.effective_n_kv_heads == 4


class TestLLaDAXLMModel:
    def test_forward_shape(self, tiny_llada_model):
        x = torch.randint(0, 100, (2, 16))
        with torch.no_grad():
            logits = tiny_llada_model(x)
        assert logits.shape == (2, 16, 128)  # embedding_size, not vocab_size

    def test_mlm_protocol_signature(self, tiny_llada_model):
        """(x_t, attention_mask, positions) -> logits, like DreamXLMModel."""
        x = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.bool)
        attention_mask[1, -4:] = False
        positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = tiny_llada_model(x, attention_mask, positions)
        assert logits.shape == (2, 16, 128)

    def test_position_ids_default_parity(self, tiny_llada_model):
        """positions == arange must be bitwise identical to the default RoPE path."""
        x = torch.randint(0, 100, (2, 16))
        positions = torch.arange(16).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            base = tiny_llada_model(x)
            with_pos = tiny_llada_model(x, None, positions)
        assert torch.equal(base, with_pos)

    def test_padding_mask_changes_only_padded_context(self, tiny_llada_model):
        """Masking out trailing pads must not equal the unmasked forward."""
        x = torch.randint(0, 100, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.bool)
        attention_mask[:, -4:] = False
        with torch.no_grad():
            unmasked = tiny_llada_model(x)
            masked = tiny_llada_model(x, attention_mask)
        assert not torch.allclose(unmasked[:, :12], masked[:, :12])

    def test_state_dict_keys_match_hf_layout(self, tiny_llada_model):
        keys = set(tiny_llada_model.state_dict().keys())
        assert "model.transformer.wte.weight" in keys
        assert "model.transformer.ln_f.weight" in keys
        assert "model.transformer.ff_out.weight" in keys  # untied head
        assert "model.transformer.blocks.0.q_proj.weight" in keys
        assert "model.transformer.blocks.1.up_proj.weight" in keys

    def test_strict_state_dict_roundtrip(self, tiny_llada_model):
        from llada import LLaDAXLMModel

        fresh = LLaDAXLMModel(tiny_config(), init_params=False)
        fresh.load_state_dict(tiny_llada_model.state_dict(), strict=True)
        x = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            assert torch.equal(tiny_llada_model(x), fresh(x))


@pytest.mark.skipif(
    os.environ.get("XLM_RUN_HUB_TESTS", "0") != "1",
    reason="set XLM_RUN_HUB_TESTS=1 to download GSAI-ML/LLaDA-8B-Base and run parity",
)
class TestLLaDAHubParity:
    def test_checkpoint_logits_parity(self):
        """Our port must reproduce the trust_remote_code reference on real weights."""
        from transformers import AutoModel

        from llada import LLaDAXLMModel

        repo = "GSAI-ML/LLaDA-8B-Base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ours = LLaDAXLMModel.from_pretrained(repo, dtype=torch.bfloat16).to(device).eval()
        ref = (
            AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        torch.manual_seed(0)
        x = torch.randint(0, 126080, (2, 32), device=device)
        x[:, -8:] = 126336  # mask tokens
        with torch.no_grad():
            ref_logits = ref(input_ids=x).logits.float()
            our_logits = ours(x).float()
        assert torch.allclose(ref_logits, our_logits, atol=1e-3, rtol=1e-3)

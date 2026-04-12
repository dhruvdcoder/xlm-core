from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from xlm.datamodule import Tokenizer
from xlm.harness import Predictor
from xlm.noise import NoiseSchedule
import torch.distributions as dists
from .dreamon_model import DreamOnModel

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0

def _dreamon_mdm_batch_generate(
    model: DreamOnModel,
    x: torch.LongTensor,
    *,
    mask_id: int,
    expand_id: int,
    steps: int,
    eps: float,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    alg: str,
    alg_temp: Optional[float],
    show_progress: bool = False,
    decode_fn: Optional[Any] = None,
) -> torch.LongTensor:
    """Port of ``MDMGenerator.batch_generate`` using a Dream/DreamOn ``model`` forward."""
    device = x.device
    x = x.clone()
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    for i in range(steps):
        mask_index = x == mask_id
        logits = model(x).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = logits[mask_index]
        if (
            expand_id is not None
            and expand_id >= 0
            and expand_id < logits.shape[-1]
        ):
            logits[:, expand_id] -= 1e9
        t = timesteps[i]
        s = timesteps[i + 1]
        if torch.all(~mask_index):
            break

        if alg == "origin":
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + mask_id
            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
            _, x0[transfer_index_t_s] = sample_tokens(
                logits[transfer_index_t_s],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            x[mask_index] = x0.clone()
        else:
            if alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    margin_confidence=True,
                )
            elif alg == "entropy":
                confidence, x0 = sample_tokens(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            number_transfer_tokens = 1
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                else:
                    confidence = confidence / alg_temp
                    confidence = F.softmax(confidence, dim=-1)
                    transfer_index = torch.multinomial(
                        confidence, num_samples=number_transfer_tokens
                    )
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + mask_id
                x0_[transfer_index] = x0[transfer_index].clone()
                x[mask_index] = x0_

        if show_progress and decode_fn is not None:
            print("==" * 50 + f" step {i} " + "==" * 50)
            print(decode_fn(x[0].tolist()))

    return x


def _dreamon_mdm_batch_generate_with_expand(
    model: DreamOnModel,
    input_ids: torch.LongTensor,
    *,
    mask_id: int,
    expand_id: int,
    eos_id: int,
    max_tokens: int,
    max_gen_len: int,
    min_gen_len: int,
    steps: int,
    eps: float,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    alg: str,
    alg_temp: Optional[float],
    number_transfer_tokens: int,
    expand_budget: int,
    pad_eos_to_right: bool,
    delete_eos_token: bool,
    show_progress: bool = False,
    decode_fn: Optional[Any] = None,
) -> torch.LongTensor:
    """Port of ``MDMGenerator.batch_generate_with_expand_as_token`` for DreamOn."""
    device = input_ids.device
    max_tokens = min(
        max_tokens, input_ids.shape[1] + max_gen_len - min_gen_len
    )
    x = F.pad(
        input_ids,
        (0, max_tokens - input_ids.shape[1]),
        value=mask_id,
    )
    num_generation_tokens = min_gen_len
    expand_budget_local = expand_budget

    for i in range(steps):
        cur_generation_window_length = (
            input_ids.shape[1] - min_gen_len + num_generation_tokens
        )
        attention_mask = torch.ones(
            [input_ids.shape[0], cur_generation_window_length],
            dtype=torch.int16,
            device=device,
        )
        attention_mask = F.pad(
            attention_mask,
            (0, max_tokens - attention_mask.shape[1]),
            value=0,
        )

        mask_index = (x == mask_id) & (attention_mask == 1)
        if torch.all(~mask_index[:, :cur_generation_window_length]):
            break

        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)

        attention_mask_2d = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )

        output = model(x, attention_mask_2d, tok_idx)
        logits = output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = logits[mask_index]

        if cur_generation_window_length == max_tokens or expand_budget_local == 0:
            if 0 <= expand_id < logits.shape[-1]:
                logits[:, expand_id] -= 1e9

        if alg == "origin":
            raise NotImplementedError("alg='origin' is not supported for mask expansion")
        if alg == "maskgit_plus":
            confidence, x0 = sample_tokens(
                logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
        elif alg == "topk_margin":
            confidence, x0 = sample_tokens(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=True,
            )
        elif alg == "entropy":
            confidence, x0 = sample_tokens(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                neg_entropy=True,
            )
        else:
            raise RuntimeError(f"Unknown alg: {alg}")

        if number_transfer_tokens > 0:
            if alg_temp is None or alg_temp == 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
            else:
                confidence = confidence / alg_temp
                confidence = F.softmax(confidence, dim=-1)
                transfer_index = torch.multinomial(
                    confidence, num_samples=number_transfer_tokens
                )
            x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + mask_id
            x0_[transfer_index] = x0[transfer_index].clone()
            x[mask_index] = x0_

        if pad_eos_to_right:
            if x.shape[0] != 1:
                raise NotImplementedError(
                    "pad_eos_to_right=True requires batch size 1 (MDMGenerator parity)"
                )
            x_seq = x[0]
            eos_indices = (x_seq == eos_id).nonzero(as_tuple=True)
            if len(eos_indices[0]) > 0:
                first_eos_idx = eos_indices[0][0].item()
                position_mask = torch.arange(x_seq.size(0), device=device) >= first_eos_idx
                replace_mask = position_mask & mask_index[0]
                x_seq.masked_fill_(replace_mask, eos_id)
                x = x_seq.unsqueeze(0)

        if show_progress and decode_fn is not None:
            print("=" * 10 + f"Step {i}" + "=" * 10)
            print(decode_fn(x[0, :cur_generation_window_length].tolist()))

        expand_indices = (x[0] == expand_id).nonzero(as_tuple=False).squeeze(1)
        if expand_indices.numel() > 0:
            for idx in sorted(expand_indices.tolist(), reverse=True):
                x = torch.cat(
                    (
                        x[:, :idx],
                        torch.tensor([[mask_id, mask_id]], device=device),
                        x[:, idx + 1 :],
                    ),
                    dim=1,
                )
                num_generation_tokens += 1
                expand_budget_local -= 1
                if x.shape[1] > max_tokens:
                    x = x[:, :max_tokens]

        if delete_eos_token:
            eos_indices = ((x[0] == eos_id) & (mask_index[0] == 1)).nonzero(
                as_tuple=False
            ).squeeze(1)
            if len(eos_indices) > 0 and show_progress:
                print("delete token")
            for idx in sorted(eos_indices.tolist(), reverse=True):
                x = torch.cat(
                    (
                        x[:, :idx],
                        x[:, idx + 1 :],
                        torch.tensor([[mask_id]], device=device),
                    ),
                    dim=1,
                )
                num_generation_tokens -= 1

    return x, num_generation_tokens


class DreamOnPredictor(torch.nn.Module, Predictor[Any, Dict[str, Any]]):
    """Harness wrapper: MDM-style fixed canvas vs expand canvas, aligned with humaneval MDMGenerator."""

    def __init__(
        self,
        model: Optional[DreamOnModel] = None,
        tokenizer: Optional[Tokenizer] = None,
        noise_schedule: Optional[NoiseSchedule] = None,
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
        mask_expansion: bool = False,
    ):
        """``model`` / ``tokenizer`` default to None so Hydra can instantiate before Harness assigns them."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.diffusion_kwargs = diffusion_kwargs or {}
        self.mask_expansion = mask_expansion

    def _mdm_runtime_cfg(self) -> Dict[str, Any]:
        """Defaults mirror ``MDMGeneratorArgs`` / ``DreamOnGenerationConfig``; override via ``diffusion_kwargs``."""
        g = self.diffusion_kwargs
        mc = (
            getattr(self.model, "generation_config", None)
            if self.model is not None
            else None
        )
        return {
            "steps": int(g.get("steps", getattr(mc, "steps", 512) if mc else 512)),
            "eps": float(g.get("eps", getattr(mc, "eps", 1e-3) if mc else 1e-3)),
            "temperature": float(g.get("temperature", 0.0)),
            "top_p": g.get("top_p", getattr(mc, "top_p", None) if mc else None),
            "top_k": g.get("top_k", getattr(mc, "top_k", None) if mc else None),
            "alg": str(g.get("alg", getattr(mc, "alg", "entropy") if mc else "entropy")),
            "alg_temp": g.get("alg_temp", getattr(mc, "alg_temp", None) if mc else None),
            "show_progress": bool(g.get("show_progress", False)),
            "max_tokens": int(
                g.get("max_tokens", getattr(mc, "max_length", 2048) if mc else 2048)
            ),
            "max_gen_len": int(
                g.get("max_gen_len", getattr(mc, "max_new_tokens", 512) if mc else 512)
            ),
            "min_gen_len": int(
                g.get("min_gen_len", getattr(mc, "min_gen_len", 16) if mc else 16)
            ),
            "expand_budget": int(
                g.get("max_gen_len", getattr(mc, "max_new_tokens", 512) if mc else 512)
            ),
            "number_transfer_tokens": int(
                g.get(
                    "number_transfer_tokens",
                    getattr(mc, "number_transfer_tokens", 1) if mc else 1,
                )
            ),
            "pad_eos_to_right": bool(g.get("pad_eos_to_right", False)),
            "delete_eos_token": bool(g.get("delete_eos_token", False)),
        }

    @torch.inference_mode
    @torch.no_grad()
    @torch._dynamo.disable()
    def predict(
        self,
        batch: Any,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        del batch_idx, dataloader_idx, dataloader_name
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "DreamOnPredictor.model and .tokenizer must be set (Harness does this after Hydra instantiate)."
            )
        input_ids = batch["input_ids"].to(self.model.device)
        cfg = self._mdm_runtime_cfg()
        mask_id = int(self.tokenizer.mask_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        expand_id = int(self.tokenizer.expand_id)

        decode_fn = (
            (lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True))
            if cfg["show_progress"]
            else None
        )
        prefix_lens = batch["prefix_lens"]
        generations = []
        if not self.mask_expansion:
            out = _dreamon_mdm_batch_generate(
                self.model,
                input_ids,
                mask_id=mask_id,
                expand_id=expand_id,
                steps=cfg["steps"],
                eps=cfg["eps"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                alg=cfg["alg"],
                alg_temp=cfg["alg_temp"],
                show_progress=cfg["show_progress"],
                decode_fn=decode_fn,
            )
            generations.extend([self.tokenizer.decode(g[pl:pl+ml].tolist(), skip_special_tokens = True) for pl, ml, g in zip(prefix_lens, batch["middle_lens"], out)])

        else:
            out, num_generation_tokens = _dreamon_mdm_batch_generate_with_expand(
                self.model,
                input_ids,
                mask_id=mask_id,
                expand_id=expand_id,
                eos_id=eos_id,
                max_tokens=cfg["max_tokens"],
                max_gen_len=cfg["max_gen_len"],
                min_gen_len=cfg["min_gen_len"],
                steps=cfg["steps"],
                eps=cfg["eps"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                alg=cfg["alg"],
                alg_temp=cfg["alg_temp"],
                number_transfer_tokens=cfg["number_transfer_tokens"],
                expand_budget=cfg["expand_budget"],
                pad_eos_to_right=cfg["pad_eos_to_right"],
                delete_eos_token=cfg["delete_eos_token"],
                show_progress=cfg["show_progress"],
                decode_fn=decode_fn,
            )
            generations.append(self.tokenizer.decode(out[0,prefix_lens[0]:prefix_lens[0] + num_generation_tokens].tolist(), skip_special_tokens = True))

        return {"text": generations}

    def to_dict(
        self,
        batch: Any,
        preds: Dict[str, Any],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del batch, batch_idx, dataloader_idx, dataloader_name
        return [{"text": t} for t in preds["text"]]

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError(
            "DreamOnPredictor.generate is not implemented; use predict()."
        )

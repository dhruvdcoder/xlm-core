#!/usr/bin/env python
"""7B parity check: xlm-core FlexMDMDreamModel/Predictor vs the reference FlexMDM release.

Loads (a) our ``FlexMDMDreamModel`` from the merged checkpoint
(``dhruveshpatel/dream_flexmdm`` or a local folder) and (b) the reference
``flexmdm.FlexMDM`` wrapper from ``genuine-any-order/FlexMDM`` on the authors'
checkpoint (``yuyuanchen0/flexmdm``), then on a few HumanEval prompts compares

1. forward outputs (token logits and insertion log-rates) at several ``t`` on
   partially masked/inserted inputs, and
2. full sampling traces with the published decoding recipe under the same seed.

Both models use SDPA attention so the kernels match. Needs ~34 GB of GPU memory in
bf16 for both 7B models (one A100/H100 80GB), or run with ``--sequential`` to keep
one model resident at a time (slower, ~17 GB).

Example (Unity)::

    FLEXMDM_REF_ROOT=/path/to/genuine-any-order/FlexMDM \\
    python xlm-models/flexmdm/scripts/flexmdm_dream_parity.py \\
        --ours dhruveshpatel/dream_flexmdm --ref yuyuanchen0/flexmdm \\
        --n-prompts 3 --steps 512 --out /tmp/flexmdm_parity.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from xlm.utils.hf_hub import load_model_weights_into_model  # noqa: E402
from xlm.utils.model_loading import no_init_weights  # noqa: E402

from flexmdm.configuration_flexmdm_dream import FlexMDMDreamConfig  # noqa: E402
from flexmdm.datamodule_flexmdm_dream import encode_flexmdm_prompt_batch  # noqa: E402
from flexmdm.model_flexmdm_dream import FlexMDMDreamModel  # noqa: E402
from flexmdm.noise_flexmdm import FlexMDMNoiseSchedule, PowerSchedule  # noqa: E402
from flexmdm.predictor_flexmdm_dream import FlexMDMDreamPredictor  # noqa: E402

REF_ALIAS = "flexmdm_ref"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def resolve_dir(repo_or_dir: str, revision: str | None = None) -> Path:
    p = Path(repo_or_dir).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_or_dir, revision=revision))


def load_reference_package(root: str):
    pkg_dir = Path(root) / "flexmdm"
    if REF_ALIAS not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            REF_ALIAS, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[REF_ALIAS] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return importlib.import_module(f"{REF_ALIAS}.utils"), importlib.import_module(
        f"{REF_ALIAS}.inference"
    )


def load_ours(ckpt_dir: Path, max_length: int, device: torch.device, dtype: torch.dtype):
    hf_cfg = json.loads((ckpt_dir / "config.json").read_text())
    config = FlexMDMDreamConfig(
        vocab_size=hf_cfg["vocab_size"],
        hidden_size=hf_cfg["hidden_size"],
        intermediate_size=hf_cfg["intermediate_size"],
        num_hidden_layers=hf_cfg["num_hidden_layers"],
        num_attention_heads=hf_cfg["num_attention_heads"],
        num_key_value_heads=hf_cfg["num_key_value_heads"],
        max_position_embeddings=hf_cfg["max_position_embeddings"],
        rms_norm_eps=hf_cfg["rms_norm_eps"],
        rope_theta=hf_cfg["rope_theta"],
        tie_word_embeddings=hf_cfg["tie_word_embeddings"],
        mask_token_id=hf_cfg["mask_token_id"],
        pad_token_id=hf_cfg["pad_token_id"],
        bos_token_id=hf_cfg["bos_token_id"],
        eos_token_id=hf_cfg["eos_token_id"],
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with no_init_weights():
            model = FlexMDMDreamModel(config, max_length=max_length)
    finally:
        torch.set_default_dtype(prev)
    index = ckpt_dir / "model.safetensors.index.json"
    single = ckpt_dir / "model.safetensors"
    load_model_weights_into_model(
        model, str(index if index.is_file() else single), map_location="cpu", strict=True, weights_only=True
    )
    return model.to(device=device, dtype=dtype).eval()


def load_ref(ref_utils, ckpt_dir: Path, max_length: int, device: torch.device, dtype_name: str):
    model, tokenizer = ref_utils.load_model_and_tokenizer(
        checkpoint_dir=str(ckpt_dir),
        max_length=max_length,
        torch_dtype_name=dtype_name,
        attn_implementation="sdpa",
    )
    return model.to(device).eval(), tokenizer


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def humaneval_prompts(n: int) -> List[Dict[str, str]]:
    import datasets

    ds = datasets.load_dataset("openai/openai_humaneval", split="test")
    return [{"task_id": r["task_id"], "prompt": r["prompt"]} for r in list(ds)[:n]]


def make_noisy_input(batch: Dict[str, torch.Tensor], mask_id: int, pad_id: int, n_masks: int, seed: int):
    """Append ``n_masks`` mask tokens after the prompt and mask a few prompt-tail tokens' successors."""
    g = torch.Generator().manual_seed(seed)
    ids = batch["input_ids"].clone()
    attn = batch["attention_mask"].clone()
    for b in range(ids.shape[0]):
        L = int(attn[b].sum())
        k = min(n_masks, ids.shape[1] - L)
        ids[b, L : L + k] = mask_id
        attn[b, L : L + k] = True
        # sprinkle a few already-revealed tokens inside the masked block
        if k > 4:
            pos = L + torch.randperm(k, generator=g)[: k // 3]
            ids[b, pos] = torch.randint(1000, 30000, (pos.numel(),), generator=g)
    return ids, attn


@torch.no_grad()
def compare_forward(ours, ref, ids, attn, ts: List[float], device) -> Dict[str, Any]:
    out = []
    for t in ts:
        tt = torch.full((ids.shape[0],), float(t), device=device)
        lo, lr_o = ours(ids, tt, attn)
        r = ref(ids, tt, attention_mask=attn)
        lref, lr_ref = r["logits"], r["log_length"]
        valid = attn.bool()
        dl = (lo.float() - lref.float()).abs()[valid]
        dr = (lr_o.float() - lr_ref.float()).abs()[valid]
        argmax_agree = (lo.argmax(-1) == lref.argmax(-1))[valid].float().mean().item()
        out.append(
            {
                "t": t,
                "logits_max_abs_diff": dl.max().item(),
                "logits_mean_abs_diff": dl.mean().item(),
                "logits_argmax_agreement": argmax_agree,
                "log_rate_max_abs_diff": dr.max().item(),
                "log_rate_mean_abs_diff": dr.mean().item(),
                "logits_abs_scale": lref.float().abs()[valid].mean().item(),
            }
        )
        print(
            f"  t={t:.2f}  |dlogits|max={out[-1]['logits_max_abs_diff']:.3e} "
            f"mean={out[-1]['logits_mean_abs_diff']:.3e}  argmax-agree={argmax_agree:.4f}  "
            f"|dlograte|max={out[-1]['log_rate_max_abs_diff']:.3e}",
            flush=True,
        )
    return {"per_t": out}


@torch.no_grad()
def compare_traces(
    ours, ref, ref_inf, tokenizer, batch, *, steps, seed, device, train_exp, unmask_exp, infer_exp, temperature
) -> Dict[str, Any]:
    pad_id, mask_id = int(tokenizer.pad_token_id), int(tokenizer.mask_token_id)
    schedule = FlexMDMNoiseSchedule(PowerSchedule(train_exp), PowerSchedule(unmask_exp))
    pred = FlexMDMDreamPredictor(
        max_steps=steps,
        tokenizer=tokenizer,
        model=ours,
        noise_schedule=schedule,
        inference_insertion_schedule=PowerSchedule(infer_exp),
        temperature=temperature,
        confidence="top_k",
    )
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    pmask = batch["prompt_mask"].to(device)

    torch.manual_seed(seed)
    t0 = time.time()
    o = pred.sample(ids, attn, pmask, trace=True)
    t_ours = time.time() - t0

    class _Ref(torch.nn.Module):
        def forward(self, x, t, attention_mask=None):
            return ref(x, t, attention_mask=attention_mask)

    torch.manual_seed(seed)
    t0 = time.time()
    r = ref_inf.flexmdm_generate(
        _Ref(),
        steps=steps,
        input_ids=ids,
        mask_id=mask_id,
        pad_id=pad_id,
        attention_mask=attn,
        prompt_mask=pmask,
        temperature=temperature,
        confidence_method="top_k",
        trace=True,
        insertion_schedule="power",
        unmasking_schedule="power",
        insertion_exponent=infer_exp,
        unmasking_exponent=unmask_exp,
        train_insertion_schedule="power",
        train_insertion_exponent=train_exp,
    )
    t_ref = time.time() - t0

    # first step at which the traces diverge (per sample), and final-output agreement
    first_div: List[int | None] = []
    for b in range(ids.shape[0]):
        d = None
        for s, (a, c) in enumerate(zip(o["history"], r["history"])):
            if not torch.equal(a[b], c[b]):
                d = s
                break
        first_div.append(d)
    final_equal = [bool(torch.equal(o["sequences"][b], r["sequences"][b])) for b in range(ids.shape[0])]
    gen_ours = pred.decode_generated(o["sequences"], o["attention_mask"], pmask)
    gen_ref = pred.decode_generated(r["sequences"], r["attention_mask"], pmask)
    res = {
        "steps": steps,
        "seed": seed,
        "first_divergence_step": first_div,
        "final_sequences_equal": final_equal,
        "final_lengths_ours": o["attention_mask"].sum(1).tolist(),
        "final_lengths_ref": r["attention_mask"].sum(1).tolist(),
        "time_ours_s": t_ours,
        "time_ref_s": t_ref,
        "generated_ours": gen_ours,
        "generated_ref": gen_ref,
    }
    print(
        f"  traces: first divergence step per sample = {first_div}  final equal = {final_equal}  "
        f"(ours {t_ours:.0f}s, ref {t_ref:.0f}s)",
        flush=True,
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="dhruveshpatel/dream_flexmdm")
    ap.add_argument("--ref", default="yuyuanchen0/flexmdm")
    ap.add_argument("--ref-root", default=os.environ.get("FLEXMDM_REF_ROOT"), help="genuine-any-order/FlexMDM dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--n-masks", type=int, default=64)
    ap.add_argument("--ts", default="0.05,0.3,0.6,0.9")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--trace-prompts", type=int, default=2, help="prompts used for the (slow) trace comparison")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--train-exp", type=float, default=1.7)
    ap.add_argument("--unmask-exp", type=float, default=2.89)
    ap.add_argument("--infer-exp", type=float, default=2.9)
    ap.add_argument("--skip-traces", action="store_true")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()
    if not args.ref_root:
        sys.exit("--ref-root (or FLEXMDM_REF_ROOT) is required")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    ref_utils, ref_inf = load_reference_package(args.ref_root)

    print(f"[load] ours: {args.ours}", flush=True)
    ours = load_ours(resolve_dir(args.ours), args.max_length, device, dtype)
    print(f"[load] ref:  {args.ref}", flush=True)
    ref, tokenizer = load_ref(ref_utils, resolve_dir(args.ref), args.max_length, device, args.dtype)

    prompts = humaneval_prompts(args.n_prompts)
    batch = encode_flexmdm_prompt_batch(tokenizer, [p["prompt"] for p in prompts], max_length=args.max_length)
    results: Dict[str, Any] = {"ours": args.ours, "ref": args.ref, "prompts": [p["task_id"] for p in prompts]}

    print("[forward] prompt-only inputs", flush=True)
    ids, attn = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    ts = [float(x) for x in args.ts.split(",")]
    results["forward_prompt_only"] = compare_forward(ours, ref, ids, attn, ts, device)
    print(f"[forward] prompt + {args.n_masks} masks (partially revealed)", flush=True)
    nids, nattn = make_noisy_input(batch, int(tokenizer.mask_token_id), int(tokenizer.pad_token_id), args.n_masks, args.seed)
    results["forward_noisy"] = compare_forward(ours, ref, nids.to(device), nattn.to(device), ts, device)

    if not args.skip_traces:
        print(f"[traces] {args.steps} steps, {args.trace_prompts} prompt(s), seed {args.seed}", flush=True)
        sub = {k: v[: args.trace_prompts] for k, v in batch.items()}
        results["traces"] = compare_traces(
            ours, ref, ref_inf, tokenizer, sub,
            steps=args.steps, seed=args.seed, device=device,
            train_exp=args.train_exp, unmask_exp=args.unmask_exp, infer_exp=args.infer_exp,
            temperature=args.temperature,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

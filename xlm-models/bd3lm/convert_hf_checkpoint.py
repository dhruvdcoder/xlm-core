"""Write a released BD3-LM checkpoint to disk in the layout this package expects.

The released models keep everything under a `BD3LM.backbone` submodule, so their tensors
are named `backbone.blocks.0.attn_qkv.weight` and ours are `blocks.0.attn_qkv.weight` -
dropping that prefix is the whole conversion.

Only needed to work offline. `+pretrained=auto` does the same thing in memory at model
construction, and also handles the vocabulary mismatch.

    python -m bd3lm.convert_hf_checkpoint \\
        kuleshov-group/bd3lm-owt-block_size4 bd3lm_owt_bs4.safetensors [--drop-vocab]

    xlm job_type=train ... +model_only_checkpoint_path=bd3lm_owt_bs4.safetensors
"""

import argparse
import glob
import os

import torch

BACKBONE_PREFIX = "backbone."

# The only tensors whose shape depends on the vocabulary. Everything else is
# vocabulary-independent and transfers to any task.
VOCAB_DEPENDENT_KEYS = (
    "vocab_embed.embedding",
    "output_layer.linear.weight",
    "output_layer.linear.bias",
)


def load_released_state_dict(path_or_repo: str) -> dict:
    """Read the released weights, from a local directory or a HuggingFace repo."""
    path = path_or_repo
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        print(f"downloading {path_or_repo} ...", flush=True)
        path = snapshot_download(
            path_or_repo, allow_patterns=["*.safetensors", "*.bin", "*.json"]
        )
        print(f"  -> {path}")

    safetensors_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(path, "*.bin")))
    if safetensors_files:
        from safetensors.torch import load_file

        return load_file(safetensors_files[0])
    if bin_files:
        return torch.load(bin_files[0], map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no .safetensors or .bin weights found in {path}")


def strip_backbone_prefix(state_dict: dict) -> tuple[dict, list]:
    """Return (converted, dropped): keys under `backbone.` with the prefix removed."""
    converted = {
        key[len(BACKBONE_PREFIX):]: value
        for key, value in state_dict.items()
        if key.startswith(BACKBONE_PREFIX)
    }
    dropped = sorted(k for k in state_dict if not k.startswith(BACKBONE_PREFIX))
    return converted, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="HuggingFace repo id or local directory of a released checkpoint"
    )
    parser.add_argument("output", help="path to write, ending in .safetensors or .pt")
    parser.add_argument(
        "--drop-vocab",
        action="store_true",
        help="omit the three vocabulary-dependent tensors (vocab_embed.embedding and "
        "output_layer.linear.{weight,bias}). Required when fine-tuning on a task whose "
        "vocabulary is not GPT-2's: torch's load_state_dict(strict=False) tolerates "
        "missing keys but NOT shape mismatches, so leaving them in makes the load fail "
        "even with strict_model_only_load=false. The 12 transformer blocks, sigma_map "
        "and the rotary embedding still transfer; the dropped tensors train from "
        "scratch.",
    )
    args = parser.parse_args()

    released = load_released_state_dict(args.source)
    converted, dropped = strip_backbone_prefix(released)

    if not converted:
        raise RuntimeError(
            f"no keys under '{BACKBONE_PREFIX}' - this does not look like a released "
            f"BD3-LM checkpoint. Top-level names were: "
            f"{sorted({k.split('.')[0] for k in released})}"
        )

    print(f"read {len(released)} tensors, {len(converted)} under {BACKBONE_PREFIX!r}")
    if dropped:
        print(f"dropped {len(dropped)} tensor(s) outside the backbone: {dropped}")

    if args.drop_vocab:
        removed = [k for k in VOCAB_DEPENDENT_KEYS if k in converted]
        for key in removed:
            del converted[key]
        print(f"--drop-vocab: removed {len(removed)} vocabulary-dependent tensors, "
              f"which will train from scratch: {removed}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    if args.output.endswith(".safetensors"):
        from safetensors.torch import save_file

        # contiguous(): safetensors rejects shared/non-contiguous storage
        save_file({k: v.contiguous() for k, v in converted.items()}, args.output)
    else:
        torch.save(converted, args.output)

    print(f"wrote {len(converted)} tensors to {args.output}")
    print(
        "\nload it with:\n"
        f"  xlm job_type=train job_name=my_finetune experiment=<your experiment> \\\n"
        f"      model=bd3lm_small \\\n"
        f"      +model_only_checkpoint_path={args.output}\n"
        "add strict_model_only_load=false if your vocabulary differs from GPT-2."
    )


if __name__ == "__main__":
    main()

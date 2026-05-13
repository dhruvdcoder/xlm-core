"""Build FSDP grouping plans from structured Hydra / dict config."""

from __future__ import annotations

import importlib
from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

GroupEntry = Tuple[str, bool]


def make_layer_wrap_policy(*class_paths: str) -> Set[Type[Any]]:
    """Resolve dotted class paths into a set of layer classes for Lightning ``FSDPStrategy``.

    Use in Hydra YAML::

        auto_wrap_policy:
          _target_: xlm.utils.fsdp_grouping.make_layer_wrap_policy
          _args_:
            - xlm.backbones.dream.modeling_dream.DreamDecoderLayer

    This matches Lightning's recommended pattern of passing ``{nn.TransformerDecoderLayer}``
    style sets to ``auto_wrap_policy`` / ``activation_checkpointing_policy``.
    """
    classes: Set[Type[Any]] = set()
    for path in class_paths:
        module_path, class_name = path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        classes.add(getattr(mod, class_name))
    return classes


def fsdp_bf16_mixed_precision():
    """Default FSDP mixed precision: bf16 params, fp32 reductions (matches DreamOn reference)."""
    import torch
    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )


def _get_int(
    cfg: MutableMapping[str, Any], key: str, context: Dict[str, Any]
) -> int:
    raw = cfg[key]
    if isinstance(raw, str) and raw.startswith("${") and "}" in raw:
        inner = raw.strip("${}")
        if "." in inner:
            root, sub = inner.split(".", 1)
            return int(cast(Any, context[root])[sub])
        return int(context[inner])
    return int(raw)


def build_fsdp_grouping_plan_from_config(
    fsdp_cfg: Union[Dict[str, Any], Any],
    *,
    context: Dict[str, Any] | None = None,
) -> List[GroupEntry]:
    """
    Build a list of (module_name_prefix, wrap_output_bool) from a config dict.

    Example YAML::

        fsdp_grouping:
          embed: { path: model.embed_tokens, wrap: false }
          layers:
            path_template: "model.layers.{i}"
            wrap: false
            count: ${model.num_hidden_layers}
          head: { path: lm_head, wrap: true }

    ``context`` should include keys referenced by interpolation (e.g. ``model``).
    """
    context = context or {}
    if DictConfig is not None and isinstance(fsdp_cfg, DictConfig):
        cfg = cast(
            Dict[str, Any],
            OmegaConf.to_container(fsdp_cfg, resolve=True),  # type: ignore[union-attr]
        )
    else:
        cfg = dict(fsdp_cfg)

    plan: List[GroupEntry] = []

    if "embed" in cfg:
        e = cfg["embed"]
        plan.append((str(e["path"]), bool(e.get("wrap", False))))

    if "layers" in cfg:
        layer = cfg["layers"]
        tmpl = str(layer["path_template"])
        count = _get_int(layer, "count", {**context, **cfg})
        wrap = bool(layer.get("wrap", False))
        for i in range(count):
            plan.append((tmpl.format(i=i), wrap))

    if "head" in cfg:
        h = cfg["head"]
        plan.append((str(h["path"]), bool(h.get("wrap", True))))

    return plan

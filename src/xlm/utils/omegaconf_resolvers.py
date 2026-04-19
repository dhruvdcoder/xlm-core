from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Iterator, List
import omegaconf
import os
import torch
from .os import get_num_processes


def determine_accumulate_grad_batches(
    global_batch_size: int,
    per_device_batch_size: int,
    num_devices: int,
    num_nodes: int,
):
    # global_batch_size should be divisible by per_device_batch_size * num_devices * num_nodes
    if not num_devices:  # cpu
        num_devices = 1
    if (
        global_batch_size % (per_device_batch_size * num_devices * num_nodes)
        != 0
    ):
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by per device batch size {per_device_batch_size} * num devices {num_devices} * num nodes {num_nodes}"
        )
    acc = global_batch_size // (
        per_device_batch_size * num_devices * num_nodes
    )
    return acc


def register_resolvers():
    omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
    omegaconf.OmegaConf.register_new_resolver(
        "device_count", torch.cuda.device_count
    )
    omegaconf.OmegaConf.register_new_resolver("eval", eval)
    omegaconf.OmegaConf.register_new_resolver(
        "div_up", lambda x, y: (x + y - 1) // y
    )
    omegaconf.OmegaConf.register_new_resolver("num_cpus", get_num_processes)
    omegaconf.OmegaConf.register_new_resolver(
        "replace_str", lambda s, sel, rep: s.replace(sel, rep)
    )
    omegaconf.OmegaConf.register_new_resolver(
        "dict_to_list", lambda d: [f"{k}={v}" for k, v in d.items()]
    )
    omegaconf.OmegaConf.register_new_resolver(
        "find_grad_accum", determine_accumulate_grad_batches
    )
    omegaconf.OmegaConf.register_new_resolver(
        "if_else",
        lambda cond, true_val, false_val: true_val if cond else false_val,
    )
    omegaconf.OmegaConf.register_new_resolver(
        "min", lambda *args: min(float(arg) for arg in args)
    )
    omegaconf.OmegaConf.register_new_resolver(
        "max", lambda *args: max(float(arg) for arg in args)
    )
    omegaconf.OmegaConf.register_new_resolver(
        "min_int", lambda *args: min(int(arg) for arg in args)
    )
    omegaconf.OmegaConf.register_new_resolver(
        "max_int", lambda *args: max(int(arg) for arg in args)
    )


def dictconfig_filter_key(
    d: omegaconf.DictConfig, fn: Callable
) -> omegaconf.DictConfig:
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
    # Using d.items_ex(resolve=False) instead of d.items() since we want to keep the
    # ${datamodule:foo} unresolved for now.
    return omegaconf.DictConfig(
        {
            k: (
                dictconfig_filter_key(v, fn)
                if isinstance(v, omegaconf.DictConfig)
                else v
            )
            # for k, v in d.items_ex(resolve=False) if fn(k)})
            for k, v in d.items()
            if fn(k)
        }
    )


def remove_keys_with_double_underscores(
    d: omegaconf.DictConfig,
) -> omegaconf.DictConfig:
    """Remove keys with double underscores.
    This can be used to remove keys that are only present for computational purposes and are not part of the final config.
    """
    return dictconfig_filter_key(d, lambda k: not k.startswith("__"))

DEFAULT_DUMMY_RESOLVER_NAMES: tuple = (
    "datamodule",
    "tokenizer",
    "lightning_module",
    "global_components",
)


@contextmanager
def dummy_resolvers(
    names: Iterable[str] = DEFAULT_DUMMY_RESOLVER_NAMES,
) -> Iterator[None]:
    """Temporarily install passthrough OmegaConf resolvers for ``names``.

    For each name, a dummy resolver is registered (replacing any existing
    resolver) that returns the unresolved interpolation string
    ``"${<name>:<attr>}"``. This lets ``OmegaConf.resolve()`` succeed early
    in the config lifecycle, before the real resolvers (which depend on the
    datamodule, tokenizer, etc.) are available.

    On exit, the previously registered resolver is restored for each name; if
    no resolver was registered before, the dummy is removed.
    """
    # BaseContainer._resolvers is the underlying dict that OmegaConf consults
    # in OmegaConf._get_resolver / register_new_resolver. Saving and restoring
    # entries here preserves the exact wrapped callables (with their cache /
    # _parent_ / _node_ / _root_ behavior) rather than rewrapping them.
    from omegaconf.basecontainer import BaseContainer

    resolvers_dict: Dict[str, Callable] = (
        BaseContainer._resolvers  # type: ignore[attr-defined]
    )

    names = list(names)
    saved: Dict[str, Callable] = {
        name: resolvers_dict[name]
        for name in names
        if name in resolvers_dict
    }

    for name in names:
        omegaconf.OmegaConf.register_new_resolver(
            name,
            lambda attr, _name=name: "${" + _name + ":" + str(attr) + "}",
            replace=True,
        )

    try:
        yield
    finally:
        for name in names:
            if name in saved:
                resolvers_dict[name] = saved[name]
            else:
                omegaconf.OmegaConf.clear_resolver(name)

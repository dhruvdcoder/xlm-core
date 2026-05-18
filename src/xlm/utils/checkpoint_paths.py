"""Checkpoint path helpers for Lightning training resume paths."""

from pathlib import Path


def is_distcp_sharded_checkpoint_dir(path: Path) -> bool:
    """Return True if *path* looks like a Lightning ``state_dict_type: sharded`` checkpoint directory.

    Such checkpoints contain one ``*.distcp`` file per rank (flat layout under e.g. ``last.ckpt/``).
    """
    return path.is_dir() and any(path.glob("*.distcp"))


def is_consolidatable_lightning_sharded_dir(path: Path) -> bool:
    """Return True if *path* can be passed to Lightning's distributed checkpoint consolidation.

    Requires shard files plus ``meta.pt`` (rank-0 metadata written by Lightning).
    """
    return is_distcp_sharded_checkpoint_dir(path) and (path / "meta.pt").is_file()


def is_usable_lightning_train_checkpoint_path(path: Path) -> bool:
    """Return True if *path* can be passed to ``Trainer.fit(..., ckpt_path=...)``.

    Accepts a regular ``.ckpt`` **file** or a **directory** with at least one ``*.distcp`` shard.
    """
    if path.is_file():
        return True
    return is_distcp_sharded_checkpoint_dir(path)


def resolve_explicit_resume_checkpoint_path(path: str | Path) -> Path:
    """Validate ``resume_checkpoint_path`` from config.

    Raises:
        ValueError: If the path does not exist, or is a directory without ``*.distcp`` shards.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"Checkpoint path does not exist: {p}")
    if p.is_file():
        return p.resolve()
    if is_distcp_sharded_checkpoint_dir(p):
        return p.resolve()
    if p.is_dir():
        raise ValueError(
            "Checkpoint path is a directory but does not look like a sharded Lightning "
            f"checkpoint (expected at least one *.distcp file): {p}"
        )
    raise ValueError(f"Invalid checkpoint path: {p}")


def find_auto_resume_checkpoint(checkpointing_dir: str) -> Path | None:
    """Return ``on_exception.ckpt`` or ``last.ckpt`` under *checkpointing_dir* if usable.

    Prefers ``on_exception.ckpt`` when present. Paths may be files or sharded directories.
    """
    base = Path(checkpointing_dir).expanduser()
    for name in ("on_exception.ckpt", "last.ckpt"):
        candidate = base / name
        if is_usable_lightning_train_checkpoint_path(candidate):
            return candidate.resolve()
    return None

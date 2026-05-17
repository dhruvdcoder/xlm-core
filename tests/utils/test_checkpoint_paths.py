"""Tests for :mod:`xlm.utils.checkpoint_paths`."""

from pathlib import Path

import pytest

from xlm.utils.checkpoint_paths import (
    find_auto_resume_checkpoint,
    is_consolidatable_lightning_sharded_dir,
    is_distcp_sharded_checkpoint_dir,
    is_usable_lightning_train_checkpoint_path,
    resolve_explicit_resume_checkpoint_path,
)


class TestIsDistcpShardedCheckpointDir:
    def test_true_when_distcp_present(self, tmp_path: Path) -> None:
        d = tmp_path / "last.ckpt"
        d.mkdir()
        (d / "__0_0.distcp").touch()
        assert is_distcp_sharded_checkpoint_dir(d)

    def test_false_when_empty_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert not is_distcp_sharded_checkpoint_dir(d)

    def test_false_when_file(self, tmp_path: Path) -> None:
        f = tmp_path / "x.ckpt"
        f.touch()
        assert not is_distcp_sharded_checkpoint_dir(f)


class TestIsUsableLightningTrainCheckpointPath:
    def test_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.ckpt"
        f.touch()
        assert is_usable_lightning_train_checkpoint_path(f)

    def test_sharded_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "last.ckpt"
        d.mkdir()
        (d / "__0_0.distcp").touch()
        assert is_usable_lightning_train_checkpoint_path(d)

    def test_plain_dir_false(self, tmp_path: Path) -> None:
        d = tmp_path / "nodistcp"
        d.mkdir()
        assert not is_usable_lightning_train_checkpoint_path(d)

    def test_missing_false(self, tmp_path: Path) -> None:
        assert not is_usable_lightning_train_checkpoint_path(
            tmp_path / "missing.ckpt"
        )


class TestResolveExplicitResumeCheckpointPath:
    def test_accepts_file(self, tmp_path: Path) -> None:
        f = tmp_path / "w.ckpt"
        f.write_text("x")
        out = resolve_explicit_resume_checkpoint_path(f)
        assert out.is_file()
        assert out.name == "w.ckpt"

    def test_accepts_sharded_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "last.ckpt"
        d.mkdir()
        (d / "__1_0.distcp").touch()
        out = resolve_explicit_resume_checkpoint_path(d)
        assert out.is_dir()
        assert any(out.glob("*.distcp"))

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            resolve_explicit_resume_checkpoint_path(tmp_path / "nope")

    def test_plain_dir_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="at least one \\*\\.distcp"):
            resolve_explicit_resume_checkpoint_path(d)


class TestFindAutoResumeCheckpoint:
    def test_prefers_on_exception_when_both_exist(self, tmp_path: Path) -> None:
        exc = tmp_path / "on_exception.ckpt"
        exc.mkdir()
        (exc / "__0_0.distcp").touch()
        last = tmp_path / "last.ckpt"
        last.mkdir()
        (last / "__0_0.distcp").touch()
        found = find_auto_resume_checkpoint(str(tmp_path))
        assert found is not None
        assert found.name == "on_exception.ckpt"

    def test_falls_back_to_last(self, tmp_path: Path) -> None:
        last = tmp_path / "last.ckpt"
        last.mkdir()
        (last / "__0_0.distcp").touch()
        found = find_auto_resume_checkpoint(str(tmp_path))
        assert found is not None
        assert found.name == "last.ckpt"

    def test_accepts_regular_file_last(self, tmp_path: Path) -> None:
        (tmp_path / "last.ckpt").write_bytes(b"x")
        found = find_auto_resume_checkpoint(str(tmp_path))
        assert found is not None
        assert found.is_file()

    def test_none_when_no_checkpoint(self, tmp_path: Path) -> None:
        assert find_auto_resume_checkpoint(str(tmp_path)) is None

    def test_on_exception_file_skips_missing_last(self, tmp_path: Path) -> None:
        (tmp_path / "on_exception.ckpt").touch()
        found = find_auto_resume_checkpoint(str(tmp_path))
        assert found is not None
        assert found.name == "on_exception.ckpt"


class TestIsConsolidatableLightningShardedDir:
    def test_true_with_meta_and_distcp(self, tmp_path: Path) -> None:
        d = tmp_path / "last.ckpt"
        d.mkdir()
        (d / "__0_0.distcp").touch()
        (d / "meta.pt").touch()
        assert is_consolidatable_lightning_sharded_dir(d)

    def test_false_without_meta(self, tmp_path: Path) -> None:
        d = tmp_path / "last.ckpt"
        d.mkdir()
        (d / "__0_0.distcp").touch()
        assert not is_consolidatable_lightning_sharded_dir(d)

#!/usr/bin/env python3
"""Bump xlm-core version, push to main, and create a GitHub release.

Used locally and from .github/workflows/release.yml. The release tag is always
``v`` + the version read from ``src/xlm/version.py`` after the bump, matching
publish.yml / docs-release.yml.

Examples:
    python .github/release.py 0.1.4
    python .github/release.py 0.1.4 --dry-run
    python .github/release.py --publish-only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_PY = REPO_ROOT / "src" / "xlm" / "version.py"
TAG_PREFIX = "v"

_VERSION_ENV_KEYS = (
    "XLM_CORE_VERSION_MAJOR",
    "XLM_CORE_VERSION_MINOR",
    "XLM_CORE_VERSION_PATCH",
    "XLM_CORE_VERSION_SUFFIX",
)

_VERSION_DEFAULT_PATTERNS = {
    "major": re.compile(
        r'(_MAJOR = os\.environ\.get\("XLM_CORE_VERSION_MAJOR", ")([^"]*)("\))'
    ),
    "minor": re.compile(
        r'(_MINOR = os\.environ\.get\("XLM_CORE_VERSION_MINOR", ")([^"]*)("\))'
    ),
    "patch": re.compile(
        r'(_PATCH = os\.environ\.get\("XLM_CORE_VERSION_PATCH", ")([^"]*)("\))'
    ),
    "suffix": re.compile(
        r'(_SUFFIX = os\.environ\.get\("XLM_CORE_VERSION_SUFFIX", ")([^"]*)("\))'
    ),
}

_VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<suffix>.*)$"
)


@dataclass(frozen=True)
class VersionParts:
    major: str
    minor: str
    patch: str
    suffix: str

    @property
    def version(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}{self.suffix}"

    @property
    def tag(self) -> str:
        return f"{TAG_PREFIX}{self.version}"


def parse_version(version: str) -> VersionParts:
    """Parse a version string using the same shape as publish.yml."""
    version = version.strip()
    if version.startswith(TAG_PREFIX):
        version = version[len(TAG_PREFIX) :]
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(
            f"Invalid version {version!r}; expected MAJOR.MINOR.PATCH "
            f"(optional -suffix, e.g. 0.1.4 or 0.1.4-alpha)"
        )
    suffix = match.group("suffix")
    if suffix and not suffix.startswith("-"):
        raise ValueError(
            f"Invalid suffix {suffix!r}; pre-release suffix must start with '-' "
            f"(e.g. 0.1.4-alpha)"
        )
    return VersionParts(
        major=match.group("major"),
        minor=match.group("minor"),
        patch=match.group("patch"),
        suffix=suffix,
    )


def read_version(path: Path = VERSION_PY) -> str:
    """Read VERSION from version.py file defaults (ignore XLM_CORE_VERSION_* env)."""
    saved = {k: os.environ.pop(k) for k in _VERSION_ENV_KEYS if k in os.environ}
    try:
        namespace: dict = {}
        exec(path.read_text(encoding="utf-8"), namespace)  # noqa: S102
        return namespace["VERSION"]
    finally:
        os.environ.update(saved)


def write_version(parts: VersionParts, path: Path = VERSION_PY) -> None:
    """Update the default values in version.py."""
    text = path.read_text(encoding="utf-8")
    replacements = {
        "major": parts.major,
        "minor": parts.minor,
        "patch": parts.patch,
        "suffix": parts.suffix,
    }
    for key, value in replacements.items():
        pattern = _VERSION_DEFAULT_PATTERNS[key]
        if not pattern.search(text):
            raise RuntimeError(f"Could not find {key} default in {path}")
        text = pattern.sub(rf"\g<1>{value}\3", text, count=1)
    path.write_text(text, encoding="utf-8")


def run(cmd: list[str], *, dry_run: bool, cwd: Path = REPO_ROOT) -> None:
    display = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {display}")
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def confirm(message: str) -> bool:
    try:
        answer = input(f"{message} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", tag],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def gh_release_exists(tag: str) -> bool:
    result = subprocess.run(
        ["gh", "release", "view", tag],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def require_clean_tree(dry_run: bool) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or stash changes before releasing."
        )
    if dry_run:
        print("[dry-run] working tree is clean")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bump version.py, push to main, and publish a GitHub release.",
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Release version (e.g. 0.1.4 or v0.1.4-alpha). Omit with --publish-only.",
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help="Skip bump/commit; create a release for the version in version.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files, pushing, or creating a release.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Bump version.py and commit locally but do not push.",
    )
    parser.add_argument(
        "--skip-release",
        action="store_true",
        help="Bump and push only; do not run gh release create.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to commit and push (default: main).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args(argv)

    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    if in_ci:
        args.yes = True

    if args.publish_only:
        if args.version:
            parser.error("do not pass version with --publish-only")
        target = parse_version(read_version())
    else:
        if not args.version:
            parser.error("version is required unless --publish-only is set")
        target = parse_version(args.version)

    current = parse_version(read_version())
    print(f"Current version: {current.version}")
    print(f"Target version:  {target.version}")
    print(f"Release tag:     {target.tag}")

    if target.tag != current.tag and not args.publish_only:
        if not args.yes and not args.dry_run:
            if not confirm("Proceed with release?"):
                print("Aborted.")
                return 1
    elif args.publish_only and not args.yes and not args.dry_run:
        if not confirm(f"Create GitHub release {target.tag}?"):
            print("Aborted.")
            return 1

    if tag_exists(target.tag) or gh_release_exists(target.tag):
        raise RuntimeError(f"Release {target.tag} already exists")

    if not args.publish_only:
        require_clean_tree(args.dry_run)
        write_version(target)
        written = read_version()
        if written != target.version:
            raise RuntimeError(
                f"version.py mismatch after write: expected {target.version}, got {written}"
            )
        print(f"Updated {VERSION_PY.relative_to(REPO_ROOT)}")

        commit_msg = f"Release version {target.version}"
        run(["git", "add", str(VERSION_PY.relative_to(REPO_ROOT))], dry_run=args.dry_run)
        run(["git", "commit", "-m", commit_msg], dry_run=args.dry_run)

        if not args.skip_push:
            run(["git", "push", "origin", f"HEAD:{args.branch}"], dry_run=args.dry_run)

    if not args.skip_release:
        gh_cmd = [
            "gh",
            "release",
            "create",
            target.tag,
            "--title",
            target.tag,
            "--target",
            args.branch,
            "--generate-notes",
        ]
        run(gh_cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete; no changes were pushed and no release was created.")
    else:
        print(f"Done. Published {target.tag} — PyPI/docs workflows run on release publish.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

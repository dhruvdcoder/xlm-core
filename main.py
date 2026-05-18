"""MkDocs macros for linking to source files in the xlm-core GitHub repository."""

from __future__ import annotations


def _repo_url(env) -> str:
    return env.conf.repo_url.rstrip("/")


def define_env(env):
    """Register documentation macros (see https://mkdocs-macros-plugin.readthedocs.io/)."""

    @env.macro
    def gh(
        path: str,
        label: str | None = None,
        branch: str = "main",
        anchor: str | None = None,
    ) -> str:
        """Markdown link to a file: ``{{ gh('src/xlm/harness.py') }}``."""
        path = path.lstrip("/")
        text = label if label is not None else path.rsplit("/", 1)[-1]
        url = f"{_repo_url(env)}/blob/{branch}/{path}"
        if anchor:
            url += f"#{anchor.lstrip('#')}"
        return f"[{text}]({url})"

    @env.macro
    def gh_dir(path: str, label: str | None = None, branch: str = "main") -> str:
        """Markdown link to a directory: ``{{ gh_dir('src/xlm/tasks') }}``."""
        path = path.rstrip("/").lstrip("/")
        text = label if label is not None else f"{path.rsplit('/', 1)[-1]}/"
        url = f"{_repo_url(env)}/tree/{branch}/{path}"
        return f"[{text}]({url})"

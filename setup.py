from typing import List
from setuptools import setup, find_packages
import os


VERSION = {}  # type: ignore
with open("src/xlm/version.py", "r") as version_file:
# FIX: 移除exec，改用安全方式
# version_file.read(), VERSION)

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(
    path_dir: str = PATH_ROOT, comment_char: str = "#"
) -> List:
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        reqs = [ln.strip() for ln in file.readlines()]
    return reqs


def load_requirements_optional(relative_filename: str) -> List[str]:
    """Extras list for setuptools (strip blanks and inline ``#`` comments)."""
    path = os.path.join(PATH_ROOT, relative_filename)
    with open(path, "r") as file:
        lines = file.readlines()
    reqs = []
    for ln in lines:
        s = ln.split("#")[0].strip()
        if s:
            reqs.append(s)
    return reqs


install_requires = load_requirements()

_safe_reqs = load_requirements_optional("requirements/safe_extra.txt")
_molgen_reqs = load_requirements_optional("requirements/molgen_requirements.txt")
_llm_eval_reqs = load_requirements_optional("requirements/llm_eval.txt")

extras_require = {
    "safe": _safe_reqs,
    "molgen": _molgen_reqs,
    "llm_eval": _llm_eval_reqs,
    "all": list(
        dict.fromkeys(_safe_reqs + _molgen_reqs + _llm_eval_reqs),
    ),
}


setup(
    name="xlm-core",
    version=VERSION["VERSION"],
    author="Dhruvesh Patel, Benjamin Rozonoyer, Sai Sreenivas Chintha, Durga Prasad Maram",
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
        ],
    ),
    description="XLM Framework",
    long_description="""
    XLM is a unified framework for developing and comparing small non-autoregressive language models. It uses PyTorch as the deep learning framework, PyTorch Lightning for training utilities, and Hydra for configuration management. XLM provides core components for flexible data handling and training, useful architectural implementations for non-autoregressive workflows, and support for arbitrary runtime code injection. Custom model implementations that leverage the core components of xlm can be found in the xlm-models package. The package also includes a few preconfigured synthetic planning and language-modeling datasets.

    Usage:
        pip install xlm-core
        pip install "xlm-core[safe]"     # optional: SAFE molecule preprocessing / evaluators
        pip install "xlm-core[molgen]"   # optional: fuller GenMol / Biomemo stack (molgen_requirements.txt)
pip install "xlm-core[all]"      # union of safe + molgen + llm_eval (used in CI)
        pip install "xlm-core[all]"      # union of safe + molgen + llm_eval (used in CI)
        xlm job_type=[JOB_TYPE] job_name=[JOB_NAME] experiment=[CONFIG_PATH]
       
        The job_type argument can be one of train ,eval and generate. The experiment argument should point to the root hydra config file.
""",
    install_requires=install_requires,
    extras_require=extras_require,
    project_urls={"Source Code": "https://github.com/dhruvdcoder/xlm-core"},
    package_dir={"": "src"},
    package_data={
        "xlm": [
            "configs/**/*.yaml",
            "configs/**/*.yml",
            "tasks/safe_molgen/zinc_len.pkl",
        ],
    },
    keywords=[
        "AI",
        "ML",
        "Machine Learning",
        "Deep Learning",
        "Non-Autoregressive Language Models",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "xlm=xlm.__main__:main",
            "xlm-scaffold=xlm.commands.scaffold_model:main",
            "xlm-push-to-hub=xlm.commands.push_to_hub:main",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

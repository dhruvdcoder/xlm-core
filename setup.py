from typing import List
from setuptools import setup, find_packages
import os


VERSION = {}  # type: ignore
with open("src/xlm/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

PATH_ROOT = os.path.dirname(__file__)

def load_requirements(
    path_dir: str = PATH_ROOT, comment_char: str = "#"
) -> List:
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        reqs = [ln.strip() for ln in file.readlines()]
    return reqs

install_requires = load_requirements()

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
    description='XLM Framework',
    long_description="""
    XLM is a unified framework for developing and comparing small non-autoregressive language models. It uses PyTorch as the deep learning framework, PyTorch Lightning for training utilities, and Hydra for configuration management. XLM provides core components for flexible data handling and training, useful architectural implementations for non-autoregressive workflows, and support for arbitrary runtime code injection. Custom model implementations that leverage the core components of xlm can be found in the xlm-models package. The package also includes a few preconfigured synthetic planning and language-modeling datasets.

    Usage:
        pip install xlm-core

    Command usage:
        xlm job_type=[JOB_TYPE] job_name=[JOB_NAME] experiment=[CONFIG_PATH]
       
        The job_type argument can be one of train ,eval and generate. The experiment argument should point to the root hydra config file.
""",
    install_requires = install_requires,
    project_urls={
        "Source Code": "https://github.com/dhruvdcoder/xlm-core"
    },
    package_dir={"": "src"},
    package_data={
        "xlm": ["configs/**/*.yaml", "configs/**/*.yml"],
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
            "xlm-scaffold=xlm.commands.scaffold_model:main"
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

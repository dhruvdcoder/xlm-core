from typing import List
from setuptools import setup, find_packages
import os


VERSION = {}  # type: ignore
with open("src/xlm/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


PATH_ROOT = os.path.dirname(__file__)

with open("README.md", "r") as fh:
    long_description = fh.read()

def load_requirements(
    path_dir: str = PATH_ROOT, comment_char: str = "#"
) -> List:
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        reqs = [ln.strip() for ln in file.readlines()]
    return reqs

install_requires = load_requirements()

setup(
    name="xlm",
    version=VERSION["VERSION"],
    author="Dhruvesh Patel",
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
        ],
    ),
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
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "xlm=xlm.__main__:main",
            "xlm-scaffold=xlm.commands.scaffold_model:main",
        ],
    },
    python_requires=">=3.11",
)

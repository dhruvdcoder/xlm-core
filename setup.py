from typing import List
from setuptools import setup, find_packages
import os


VERSION = {}  # type: ignore
with open("src/xlm/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


PATH_ROOT = os.path.dirname(__file__)

with open("README.md", "r") as fh:
    long_description = fh.read()


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
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "xlm=xlm.__main__:main",
        ]
    },
    python_requires=">=3.11",
)

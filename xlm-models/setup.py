import os

from setuptools import setup

VERSION = {}  # type: ignore
with open(os.path.join(os.path.dirname(__file__), "version.py"), "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="xlm-models",
    version=VERSION["VERSION"],
    description="Collection of Language Models for XLM Framework",
    long_description="""
    Collection of language models for the XLM framework.
    
    Available models:
    - arlm: Auto-Regressive Language Model
    - mlm : Masked Language Model
    - ilm : Infilling Language Model
    - mdlm: Masked Diffusion Language Model
    - flexmdm: Flexible Masked Diffusion Language Model
    - dream: Dream diffusion LM 
    
    Usage:
        pip install xlm-models

        Model package names must be specified in the XLM_MODEL_PACKAGES environment variable as a colon-separated list (e.g., arlm:mlm:ilm:mdlm:dream)
    """,
    packages=["arlm", "mlm", "ilm", "mdlm", "flexmdm", "dream"],
    author="Dhruvesh Patel, Benjamin Rozonoyer, Sai Sreenivas Chintha, Durga Prasad Maram",
    package_dir={
        "arlm": "arlm",
        "mlm": "mlm",
        "ilm": "ilm",
        "mdlm": "mdlm",
        "flexmdm": "flexmdm",
        "dream": "dream",
    },
    package_data={
        "arlm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "mlm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "ilm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "mdlm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "flexmdm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "dream": ["configs/**/*.yaml", "configs/**/*.yml"],
    },
    install_requires=[
        f"xlm-core=={VERSION['VERSION']}",
    ],
    project_urls={
        "Source Code": "https://github.com/dhruvdcoder/xlm-core/tree/main/xlm-models"
    },
    include_package_data=True,
    python_requires=">=3.11",
    keywords=[
        "AI",
        "ML",
        "Machine Learning",
        "Deep Learning",
        "Non-Autoregressive Language Models",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

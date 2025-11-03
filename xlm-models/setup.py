from setuptools import setup

setup(
    name="xlm-models",
    version="0.1.0",
    description="Collection of Language Models for XLM Framework",
    long_description="""
    Collection of language models for the XLM framework.
    
    Available models:
    - arlm: Auto-Regressive Language Model
    - idlm: Iterative Diffusion Language Model  
    - ilm: Infilling Language Model
    - mlm: Masked Language Model
    - mdlm: Masked Diffusion Language Model
    - elm: Edit Language Model
    - indigo: Indigo Model
    - zlm: Zero Language Model
    
    All models are included when you install this package.
    
    Usage:
        pip install xlm-models
    """,
    packages=["arlm", "idlm", "ilm", "mlm", "mdlm", "elm", "indigo"],
    package_dir={
        "arlm": "arlm/arlm",
        "idlm": "idlm/idlm",
        "ilm": "ilm/ilm",
        "mlm": "mlm/mlm",
        "mdlm": "mdlm/mdlm",
        "elm": "elm/elm",
        "indigo": "indigo/indigo",
    },
    install_requires=[
        "xlm",  # Core XLM framework dependency
    ],
    python_requires=">=3.11",
    author="XLM Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

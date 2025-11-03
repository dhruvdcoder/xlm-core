from setuptools import setup, find_packages

setup(
    name="xlm-models",
    version="0.1.0",
    description="External Language Models for XLM Framework",
    long_description="""
    Collection of language models for the XLM framework. Each model can be installed
    independently or as part of a collection.
    
    Available models:
    - arlm: Auto-Regressive Language Model
    - idlm: Iterative Diffusion Language Model  
    - ilm: Infilling Language Model
    - mlm: Masked Language Model
    - mdlm: Masked Diffusion Language Model
    - elm: Edit Language Model
    - indigo: Indigo Model
    - zlm: Zero Language Model
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
    ],
    extras_require={
        # Note: Each model is a separate package and should be installed separately
        # Example: pip install -e ./arlm
        "arlm": [],  # Auto-Regressive Language Model
        "idlm": [],  # Iterative Diffusion Language Model
        "ilm": [],  # Infilling Language Model
        "mlm": [],  # Masked Language Model
        "mdlm": [],  # Masked Diffusion Language Model
        "elm": [],  # Edit Language Model
        "indigo": [],  # Indigo Model
        "zlm": [],  # Zero Language Model (placeholder)
        "all": [],  # Install all available models separately
    },
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

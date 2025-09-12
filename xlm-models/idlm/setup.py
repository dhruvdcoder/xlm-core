from setuptools import setup, find_packages

setup(
    name="idlm",
    version="0.1.0",
    description="IDLM - Iterative Diffusion Language Model for XLM framework",
    long_description="""
    Iterative Diffusion Language Model (IDLM) implementation for the XLM framework.
    
    This package provides:
    - DDITIDLMModel: Diffusion Transformer model with iterative refinement
    - IdlmLoss: Diffusion training loss function
    - IdlmPredictor: Iterative denoising text generation
    - Data collators for training and inference
    - Multiple noise schedules (Poisson, LogLinear, Geometric)
    - Comprehensive type definitions
    
    Originally part of xlm.lm.idlm, now available as an independent package.
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "idlm": ["configs/**/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.11",
    author="XLM Team",
    author_email="xlm-team@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="language-model, diffusion, iterative, nlp, machine-learning",
)

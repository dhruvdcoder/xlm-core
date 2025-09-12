from setuptools import setup, find_packages

setup(
    name="arlm",
    version="0.1.0",
    description="ARLM - Auto-Regressive Language Model for XLM framework",
    long_description="""
    Auto-Regressive Language Model (ARLM) implementation for the XLM framework.
    
    This package provides:
    - RotaryTransformerARLMModel: Transformer model with rotary embeddings
    - ARLMLoss: Causal language modeling loss function
    - ARLMPredictor: Auto-regressive text generation
    - Data collators for training and inference
    - Comprehensive type definitions
    
    Originally part of xlm.lm.arlm, now available as an independent package.
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "arlm": ["configs/**/*.yaml"],
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
    keywords="language-model, transformer, auto-regressive, nlp, machine-learning",
)

from setuptools import setup, find_packages

setup(
    name="mlm",
    version="0.1.0",
    description="MLM - Masked Language Model for XLM framework",
    long_description="""
    Masked Language Model (MLM) implementation for the XLM framework.
    
    This package provides:
    - RotaryTransformerMLMModel: Transformer model for masked language modeling
    - MLMLoss: Masked language modeling loss function
    - MLMPredictor: Text generation with masking
    - Data collators for training and inference
    - History tracking for generation
    - Comprehensive type definitions
    
    Originally part of xlm.lm.mlm, now available as an independent package.
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "mlm": ["configs/**/*.yaml"],
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
    keywords="language-model, masked-lm, transformer, nlp, machine-learning",
)

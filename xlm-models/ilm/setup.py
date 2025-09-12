from setuptools import setup, find_packages

setup(
    name="ilm",
    version="0.1.0",
    description="ILM - Infilling Language Model for XLM framework",
    long_description="""
    Infilling Language Model (ILM) implementation for the XLM framework.
    
    This package provides:
    - RotaryTransformerILMModel: Transformer model for text infilling
    - ILMLossWithMaskedCE: Masked cross-entropy loss for infilling
    - ILMPredictor: Text infilling and generation
    - Data collators for training and inference
    - Multiple model variants with classification heads
    - Comprehensive type definitions
    
    Originally part of xlm.lm.ilm, now available as an independent package.
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "ilm": ["configs/**/*.yaml"],
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
    keywords="language-model, infilling, transformer, nlp, machine-learning",
)

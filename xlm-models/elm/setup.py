from setuptools import setup, find_packages

setup(
    name="elm",
    version="0.1.0",
    description="ELM - Edit Language Model for XLM framework",
    long_description="""
    Edit Language Model (ELM) implementation for the XLM framework.
    
    This package provides:
    - RotaryTransformerELMModel: Transformer model for text editing
    - ELMLoss: Edit language modeling loss
    - ELMPredictor: Text editing and generation
    - Data collators for training and inference
    - Comprehensive type definitions
    
    Originally part of xlm.lm.elm, now available as an independent package.
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "elm": ["configs/**/*.yaml"],
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
    keywords="language-model, editing, transformer, nlp, machine-learning",
)

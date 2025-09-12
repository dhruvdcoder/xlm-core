from setuptools import setup, find_packages

setup(
    name="indigo",
    version="0.1.0",
    description="Indigo - Advanced Language Model for XLM framework",
    long_description="""
    Indigo Model implementation for the XLM framework.
    
    This package provides:
    - IndigoModel: Advanced transformer architecture 
    - IndigoLoss: Specialized loss function
    - IndigoPredictor: Text generation with advanced capabilities
    - Data collators for training and inference
    - Comprehensive type definitions
    
    Originally part of xlm.lm.indigo, now available as an independent package.
    """,
    packages=find_packages(),
    install_requires=[
        "xlm",  # Core XLM framework dependency
        "torch",
        "jaxtyping",
    ],
    package_data={
        "indigo": ["configs/**/*.yaml"],
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
    keywords="language-model, indigo, transformer, nlp, machine-learning",
)

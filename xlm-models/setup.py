from setuptools import setup, find_packages

setup(
    name="xlm-models",
    version="0.1.0",
    description="Collection of Language Models for XLM Framework",
    long_description="""
    Collection of language models for the XLM framework.
    
    Available models:
    - arlm: Auto-Regressive Language Model
    - mlm : Masked Language Model
    - ilm : Infilling Language Model
    - mdlm: Masked Diffusion Language Model
    
    Usage:
        pip install xlm-models
    """,
    packages=["arlm","mlm","ilm","mdlm"],
    author="Dhruvesh Patel, Benjamin Rozonoyer, Sai Sreenivas Chintha, Durga Prasad Maram",
    package_dir={
        "arlm": "arlm",
        "mlm": "mlm",
        "ilm": "ilm",
        "mdlm": "mdlm",
    },
    package_data={
        "arlm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "mlm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "ilm": ["configs/**/*.yaml", "configs/**/*.yml"],
        "mdlm": ["configs/**/*.yaml", "configs/**/*.yml"],
    },
    install_requires=[
        "xlm",  # Core XLM framework dependency
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
        "None Autoregressive Language Models",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

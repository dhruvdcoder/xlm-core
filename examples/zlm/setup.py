from setuptools import setup, find_packages

setup(
    name="zlm",
    version="0.1.0",
    description="ZLM - External Language Model for XLM framework",
    packages=find_packages(),
    install_requires=[
        "xlm",  # Main XLM package dependency
    ],
    package_data={
        "zlm": ["configs/**/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.11",
)

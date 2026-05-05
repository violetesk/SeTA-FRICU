from setuptools import setup, find_packages

setup(
    name="seta",
    version="1.0.0",
    description="SeTA: Semantic-Temporal Alignment for ICU Risk Prediction",
    author="SeTA Authors",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "accelerate>=1.0",
        "transformers>=4.40",
        "deepspeed>=0.14",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "tqdm>=4.60",
    ],
)

from skbuild import setup
from setuptools import find_packages

setup(
    name="axiom",
    version="0.1.0",
    description="Axiom ML/DL library with optional GPU support",
    author="Ishan Singh",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    cmake_args=[
        "-DAXIOM_USE_CUDA=ON"
    ],
    python_requires=">=3.11",
    install_requires=[
        # Add runtime Python dependencies here
    ],
)

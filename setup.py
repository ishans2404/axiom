from skbuild import setup
from setuptools import find_packages

setup(
    name="axiom",
    version="0.1.0",
    description="Axiom ML/DL library with optional GPU support",
    author="Ishan Singh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_args=[
        # Pass custom cmake flags here if needed, e.g. enable/disable CUDA
        # "-DAXIOM_USE_CUDA=ON"  # or OFF for CPU only
    ],
    python_requires=">=3.11",
    install_requires=[
        # Put your python dependencies here, e.g. numpy, torch etc
    ],
    # Add classifiers, keywords etc as you want
)

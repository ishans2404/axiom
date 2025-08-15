from skbuild import setup
from setuptools import find_packages
import sys
import os

# Platform-specific platlib path
platlib_dir = os.path.join(
    os.path.dirname(__file__),
    "python"  # relative path for where your .py files live
)

setup(
    name="axiom",
    version="0.1.0",
    description="Axiom ML/DL library with optional GPU support",
    author="Ishan Singh",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    cmake_args=[
        "-DAXIOM_USE_CUDA=ON",
        # f"-DSKBUILD_PLATLIB_DIR={platlib_dir}",
        "-DCMAKE_INSTALL_LIBDIR=."
    ],
    python_requires=">=3.11",
    install_requires=[
        # Add runtime Python dependencies here
    ],
)

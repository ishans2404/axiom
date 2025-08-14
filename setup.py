from skbuild import setup
from setuptools import find_packages
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "_axiom",
        [
            "src/axiom/bindings/module.cpp",
            "src/axiom/bindings/converters.cpp"
        ],
    ),
]

setup(
    name="axiom",
    version="0.1.0",
    description="Axiom ML/DL library with optional GPU support",
    author="Ishan Singh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_args=[
        # "-DAXIOM_USE_CUDA=ON",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Add your dependencies here
    ],
    ext_modules=ext_modules,
)

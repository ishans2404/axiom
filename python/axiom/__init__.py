"""
Axiom Python API
----------------
High-performance math utilities backed by C++/CUDA.
"""

import importlib

try:
    # Import the compiled extension module (built by pybind11)
    _axiom = importlib.import_module("_axiom")
except ModuleNotFoundError as e:
    raise ImportError(
        "The '_axiom' extension could not be found. "
        "Make sure you have built the package and installed it correctly."
    ) from e

# Re-export symbols from the compiled module
__all__ = [name for name in dir(_axiom) if not name.startswith("_")]

# Populate the current module's namespace
globals().update({name: getattr(_axiom, name) for name in __all__})

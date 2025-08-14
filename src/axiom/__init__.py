from ._axiom import *

# Optionally, add a friendly package-level docstring
__doc__ = """
Axiom Python package.

Provides Python bindings to the Axiom C++ core via the `_axiom` extension module.
"""

__all__ = _axiom.__all__ if hasattr(_axiom, '__all__') else dir(_axiom)

# Package metadata (optional)
__version__ = "0.1.0"
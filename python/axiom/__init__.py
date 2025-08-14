"""
Axiom Python API
================
This package provides Python bindings and utilities for the Axiom C++/CUDA core.

It automatically loads compiled `_axiom` extensions and exposes their
functionality to Python users in an easy-to-use interface.

Example:
    >>> import axiom
    >>> axiom.version()
    '0.1.0'
"""

# Import the compiled extension (built via pybind11)
try:
    from _axiom import *  # noqa: F403
except ImportError as e:
    raise ImportError(
        "The '_axiom' extension could not be loaded. "
        "Ensure that the package is built and installed correctly."
    ) from e

# Version (sync with your setup.py / pyproject.toml)
__version__ = "0.1.0"

# Public API exports
__all__ = [
    "__version__",
    # Add bindings from _axiom here if you want them top-level
    # e.g. "add", "multiply", "some_gpu_function"
]

def version():
    """Return the Axiom package version."""
    return __version__


def gpu_available():
    """
    Check if GPU support is available in the compiled backend.
    
    Returns:
        bool: True if GPU features are available, False otherwise.
    """
    try:
        # Assuming _axiom has `has_cuda` or similar flag
        return hasattr(__builtins__, "has_cuda") and has_cuda()  # noqa: F405
    except Exception:
        return False

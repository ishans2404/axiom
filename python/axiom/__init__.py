"""
Axiom Python API
----------------
High-performance math utilities backed by C++/CUDA.
"""

try:
    # Import the compiled extension module directly
    import _axiom
except ImportError as e:
    raise ImportError(
        f"The '_axiom' extension could not be found. "
        f"Make sure you have built and installed the package correctly. "
        f"Error: {e}"
    ) from e

# Re-export symbols from the compiled module
__all__ = []

# Dynamically add all public functions from _axiom
for name in dir(_axiom):
    if not name.startswith('_'):
        globals()[name] = getattr(_axiom, name)
        __all__.append(name)

# Version info
__version__ = "0.1.0"

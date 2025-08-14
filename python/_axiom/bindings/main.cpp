#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declare binding functions
void bind_math_utils(py::module_ &);

PYBIND11_MODULE(_axiom, m) {
    m.doc() = "Axiom Python bindings";
    bind_math_utils(m);
}

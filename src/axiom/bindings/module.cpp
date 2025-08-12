#include <pybind11/pybind11.h>
#include <string>
#include "axiom/core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(axiom_cpp, m) {
    m.doc() = "Axiom core Python bindings";
    m.def("hello", &hello_core, "Return a greeting from the C++ core");
}

#include <pybind11/pybind11.h>
#include <string>
#include "axiom/core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_axiom, m) {
    m.doc() = "Axiom core Python bindings";
    m.def("add", &add, "Add two integers");
    m.def("greet", &greet, "Greet someone by name");
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include core library headers
#include "core/core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_axiom, m) {
    m.doc() = "Python bindings for the Axiom ML/DL library";

    // Example binding: greet function from core
    m.def("greet", &axiom::greet, py::arg("name"),
          "Return a friendly greeting from Axiom");

    // You can add more bindings here, for ML algorithms, classes, etc.
    // Example for future expansion:
    // py::class_<axiom::Model>(m, "Model")
    //     .def(py::init<>())
    //     .def("train", &axiom::Model::train)
    //     .def("predict", &axiom::Model::predict);
}

#include <pybind11/pybind11.h>
#include "core/math_utils.hpp"

#ifdef HAVE_CUDA
#include "gpu/math_utils.cuh"
#endif

namespace py = pybind11;

void bind_math_utils(py::module_ &m) {
    m.def("add", [](int a, int b) {
    #ifdef HAVE_CUDA
        return axiom::gpu::add(a, b); // GPU version
    #else
        return axiom::core::add(a, b); // CPU version
    #endif
    });
}

#include "utils/math_utils.hpp"

#ifdef HAVE_CUDA
#include "gpu/math_utils.cuh"
#endif

int add(int a, int b) {
#ifdef HAVE_CUDA
    return add_gpu(a, b);  // GPU implementation
#else
    return a + b;  // CPU implementation
#endif
}

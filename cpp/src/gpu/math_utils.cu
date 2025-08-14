#include "gpu/math_utils.cuh"

namespace axiom {
namespace gpu {

__host__ __device__ int add(int a, int b) {
    return a + b;
}

} // namespace gpu
} // namespace axiom

#pragma once

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace axiom {
namespace core {

int add(int a, int b);

} // namespace core
} // namespace axiom

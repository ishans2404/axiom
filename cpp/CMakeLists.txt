add_library(core STATIC
    src/utils/math_utils.cpp
    src/core.cpp
)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (HAVE_CUDA)
    add_library(core_gpu STATIC
        src/gpu/math_utils.cu
    )
    target_include_directories(core_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(core_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # 🔹 Define HAVE_CUDA for both core and core_gpu
    target_compile_definitions(core PRIVATE HAVE_CUDA)
    target_compile_definitions(core_gpu PRIVATE HAVE_CUDA)
endif()

# Python extension
pybind11_add_module(_axiom python_bindings/main.cpp)
target_link_libraries(_axiom PRIVATE core)

if (HAVE_CUDA)
    target_link_libraries(_axiom PRIVATE core_gpu CUDA::cudart)
    target_compile_definitions(_axiom PRIVATE HAVE_CUDA)
endif()

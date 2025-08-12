set(CPP_SOURCES
    core.cpp
    utils.cpp
)

set(CPP_HEADERS
    ${PROJECT_SOURCE_DIR}/include/axiom/core.hpp
    ${PROJECT_SOURCE_DIR}/include/axiom/utils.hpp
)

# Optional CUDA sources
if (AXIOM_USE_CUDA)
    list(APPEND CPP_SOURCES
        ${PROJECT_SOURCE_DIR}/src/axiom/cpp/cuda/gpu_kernels.cu
        ${PROJECT_SOURCE_DIR}/src/axiom/cpp/cuda/tensor_ops.cu
    )
    set_source_files_properties(${PROJECT_SOURCE_DIR}/src/axiom/cpp/cuda/gpu_kernels.cu ${PROJECT_SOURCE_DIR}/src/axiom/cpp/cuda/tensor_ops.cu PROPERTIES LANGUAGE CUDA)

endif()

# Build the Python extension
pybind11_add_module(axiom_cpp
    ${PROJECT_SOURCE_DIR}/src/axiom/bindings/module.cpp
    ${PROJECT_SOURCE_DIR}/src/axiom/bindings/converters.cpp
    ${CPP_SOURCES}
    ${CPP_HEADERS}
)

# Include paths for your code
target_include_directories(axiom_cpp PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Link CUDA if needed
if (AXIOM_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(axiom_cpp PRIVATE CUDA::cudart)
else()
    message(STATUS "Building CPU-only version without CUDA dependencies.")
endif()


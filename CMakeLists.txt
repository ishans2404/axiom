cmake_minimum_required(VERSION 3.18)
project(axiom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------
# Pybind11
# ---------------------------------
add_subdirectory(${PROJECT_SOURCE_DIR}/external/pybind11)

# ---------------------------------
# CUDA detection
# ---------------------------------
set(AXIOM_USE_CUDA OFF)

# Try enabling CUDA (optional)
enable_language(CUDA OPTIONAL)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA detected: ${CMAKE_CUDA_COMPILER}")
    set(AXIOM_USE_CUDA ON)
else()
    message(STATUS "CUDA not found — building CPU-only version.")
endif()

# ---------------------------------
# Add source directory
# ---------------------------------
add_subdirectory(src/axiom/cpp)

# ---------------------------------
# Pass CUDA flag to code 
# ---------------------------------
if (AXIOM_USE_CUDA)
    target_compile_definitions(axiom_cpp PRIVATE AXIOM_WITH_CUDA=1)
else()
    target_compile_definitions(axiom_cpp PRIVATE AXIOM_WITH_CUDA=0)
endif()

cmake_minimum_required(VERSION 3.18)
project(axiom LANGUAGES CXX)

# Make sure C++17 is enabled
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to control CUDA usage
option(AXIOM_USE_CUDA "Build with CUDA support" ON)

# Try to detect CUDA if enabled
if(AXIOM_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
        set(HAVE_CUDA TRUE)
    else()
        message(WARNING "CUDA not found. Building without GPU support.")
        set(HAVE_CUDA FALSE)
    endif()
else()
    set(HAVE_CUDA FALSE)
endif()

# Add pybind11
add_subdirectory(external/pybind11)

# Add C++ sources
add_subdirectory(cpp)

# Install Python package files
install(
    DIRECTORY python/axiom
    DESTINATION ${SKBUILD_PLATLIB_DIR}
)

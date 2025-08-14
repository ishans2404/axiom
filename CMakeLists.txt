cmake_minimum_required(VERSION 3.18)
project(axiom LANGUAGES CXX CUDA) # <-- add CUDA here, no need for enable_language()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AXIOM_USE_CUDA "Build with CUDA support" ON)

if(AXIOM_USE_CUDA)
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

# Make HAVE_CUDA available to all subdirectories
add_compile_definitions($<$<BOOL:${HAVE_CUDA}>:HAVE_CUDA>)

# Add pybind11
add_subdirectory(external/pybind11)

# Add C++ sources
add_subdirectory(cpp)

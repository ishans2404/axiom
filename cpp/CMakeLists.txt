# ---- CPU core sources ----
file(GLOB_RECURSE CORE_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
)

add_library(core STATIC ${CORE_SRC})
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- GPU sources (optional) ----
if (HAVE_CUDA)
    file(GLOB_RECURSE GPU_SRC CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cu
    )

    add_library(core_gpu STATIC ${GPU_SRC})
    target_include_directories(core_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(core_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Define for both
    target_compile_definitions(core PRIVATE HAVE_CUDA)
    target_compile_definitions(core_gpu PRIVATE HAVE_CUDA)
endif()

# ---- Python bindings ----
file(GLOB_RECURSE PYBIND_SRC CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/_axiom/bindings/*.cpp
)

pybind11_add_module(_axiom ${PYBIND_SRC})
target_link_libraries(_axiom PRIVATE core)

if (HAVE_CUDA)
    target_link_libraries(_axiom PRIVATE core_gpu CUDA::cudart)
    target_compile_definitions(_axiom PRIVATE HAVE_CUDA)
endif()

install(TARGETS _axiom
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION .
    RUNTIME DESTINATION .
)
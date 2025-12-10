cmake_minimum_required(VERSION 3.18)
project(gpumon_client
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Header-only GPU monitoring client library"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(gpumon INTERFACE)

add_compile_definitions(GPUMON_BACKEND_CUDA)

target_include_directories(gpumon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(gpumon INTERFACE cxx_std_17)
add_library(gpumon::gpumon ALIAS gpumon)

# Python Bindings (pybind11)
option(BUILD_PYTHON "Build Python bindings" OFF)

if(BUILD_PYTHON)
    find_package(CUDAToolkit REQUIRED)

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(_gpumon_client python/bindings.cpp)

    target_link_libraries(_gpumon_client PRIVATE
        CUDA::cudart gpumon)

    target_include_directories(_gpumon_client PRIVATE include)
    install(TARGETS _gpumon_client DESTINATION gpumon)
endif()
# Optional: Build example
option(BUILD_GPUMON_EXAMPLE "Build gpumon example application" ON)
if(BUILD_GPUMON_EXAMPLE AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(example/cuda)
endif()

# Installation support
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install targets
install(TARGETS gpumon
    EXPORT gpumon_clientTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT gpumon_clientTargets
    FILE gpumon_clientTargets.cmake
    NAMESPACE gpumon::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpumon_client
)

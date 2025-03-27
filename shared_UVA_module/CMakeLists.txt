cmake_minimum_required(VERSION 3.18)
project(shared_UVA LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

include_directories(${MPI_INCLUDE_PATH})  # Ensure MPI headers are found
message(STATUS "MPI Include Path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI Found: ${MPI_FOUND}")
message(STATUS "MPI Include Path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI CXX Include Path: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI Compilation Flags: ${MPI_COMPILE_FLAGS}")
message(STATUS "MPI Libraries: ${MPI_LIBRARIES}")
message(STATUS "MPI Executable: ${MPIEXEC_EXECUTABLE}")

add_library(shared_UVA MODULE shared_UVA.cu)
set_target_properties(shared_UVA PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

target_include_directories(shared_UVA PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(shared_UVA PRIVATE pybind11::module MPI::MPI_CXX cuda /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so)
set_target_properties(shared_UVA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


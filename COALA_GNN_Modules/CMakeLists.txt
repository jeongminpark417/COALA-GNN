cmake_minimum_required(VERSION 3.3.0)
project(SSD_GNN_Pybind LANGUAGES CXX CUDA)

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules
)

find_package(pybind11 REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

include_directories(${MPI_INCLUDE_PATH})  # Ensure MPI headers are found
# message(STATUS "MPI Include Path: ${MPI_INCLUDE_PATH}")
# message(STATUS "MPI Found: ${MPI_FOUND}")
# message(STATUS "MPI Include Path: ${MPI_INCLUDE_PATH}")
# message(STATUS "MPI CXX Include Path: ${MPI_CXX_INCLUDE_DIRS}")
# message(STATUS "MPI Compilation Flags: ${MPI_COMPILE_FLAGS}")
# message(STATUS "MPI Libraries: ${MPI_LIBRARIES}")
# message(STATUS "MPI Executable: ${MPIEXEC_EXECUTABLE}")

message(STATUS "MPI4Py Include Path: ${MPI4Py_INCLUDE_DIR}")

#$set(NVSHMEM_HOME "/usr/lib/x86_64-linux-gnu/")
#set(NVSHMEM_HOME "/usr/local/nvshmem")
#set(NVSHMEM_HOME "/u/jpark346/nvshmem")

set(NVSHMEM_HOME "" CACHE PATH "Path to NVSHMEM installation")

# Check if NVSHMEM_HOME is set
if(NOT NVSHMEM_HOME)
	    message(FATAL_ERROR "Please set NVSHMEM_HOME using -DNVSHMEM_HOME=/path/to/nvshmem when running cmake.")
    endif()


set(NVSHMEM_HOST_LIB "${NVSHMEM_HOME}/lib/libnvshmem_host.so")
set(NVSHMEM_DEVICE_LIB "${NVSHMEM_HOME}/lib/libnvshmem_device.a")


include_directories(${MPI_INCLUDE_PATH} ../bam/include ../bam/include/freestanding/include/ ${NVSHMEM_HOME}/include ${MPI4Py_INCLUDE_DIR})
link_libraries(  ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so pybind11::module MPI::MPI_CXX cuda  ${NVSHMEM_HOME}/lib ${MPI_HOME}/lib)

# include_directories(${MPI_INCLUDE_PATH} ../bam/include ../bam/include/freestanding/include/ /home/jpark346/miniconda3/lib/python3.9/site-packages/mpi4py/include)
# #link_libraries(  ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so pybind11::module MPI::MPI_CXX cuda  /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so)
# link_libraries(  ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so pybind11::module MPI::MPI_CXX cuda  ${NVSHMEM_HOME} /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so)


add_library(COALA_GNN_Pybind MODULE COALA_GNN_Pybind.cu)

set_target_properties(COALA_GNN_Pybind PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/COALA_GNN_Pybind"
)

target_compile_options(COALA_GNN_Pybind PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -std=c++17;
    -rdc=true;
    -O3;
    -gencode=arch=compute_80,code=sm_80;
    -I${NVSHMEM_HOME}/include;
    -L${NVSHMEM_HOME}/lib;
    #-L${NVSHMEM_HOME}; 
  >
)

target_link_libraries(COALA_GNN_Pybind PRIVATE  ${NVSHMEM_HOST_LIB} ${NVSHMEM_DEVICE_LIB}  cuda cudart pybind11::module)


CONFIGURE_FILE( COALA_GNN_Pybind/__init__.py
	${CMAKE_CURRENT_BINARY_DIR}/COALA_GNN_Pybind/__init__.py
)

# Ditto for the setup file.
CONFIGURE_FILE( COALA_GNN_Pybind/setup.py
	${CMAKE_CURRENT_BINARY_DIR}/COALA_GNN_Pybind/setup.py
)

CMAKE_MINIMUM_REQUIRED( VERSION 3.3.0 )
PROJECT( cmake-cpp-pybind11 CXX CUDA)

SET( CMAKE_CXX_STANDARD 11 )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )

LIST( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/Modules
)

set(CMAKE_CUDA_ARCHITECTURES 80)


########################################################################
# Macros and additional flags
########################################################################

INCLUDE( CheckCXXCompilerFlag )

FILE( GLOB filenames "cmake/Macros/*.cmake" )
FOREACH( filename ${filenames} )
  INCLUDE( ${filename} )
ENDFOREACH()

########################################################################
# Library setup
########################################################################

SET( PACKAGE_VERSION "0.1.1" )

find_package(pybind11 REQUIRED)


FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs   3 REQUIRED)


# #IF( PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND PYBIND11_FOUND )
#   INCLUDE_DIRECTORIES(
#     ${PYTHON_INCLUDE_DIRS}
#     ${PYBIND11_INCLUDE_DIRS}
#   )


set(NVSHMEM_HOME "/usr/local/nvshmem")
file(GLOB NVSHMEM_LIBRARIES "${NVSHMEM_HOME}/lib/libnvshmem_host.so" "${NVSHMEM_HOME}/lib/libnvshmem_device.a")


include_directories(${NVSHMEM_HOME}/include  ${PYTHON_INCLUDE_DIRS} /usr/local/cuda/include  ./include ../bam/include ../bam/include/freestanding/include/ /usr/mpi/gcc/openmpi-4.1.5a1/include)
link_directories(${NVSHMEM_HOME}/lib /usr/local/cuda/lib64 ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so /usr/mpi/gcc/openmpi-4.1.5a1/lib/libmpi.so)


add_library(Dist_Cache MODULE gids_nvme.cu)

set_target_properties(Dist_Cache PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)


target_compile_options(Dist_Cache PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -rdc=true;
    -O3;
    -gencode=arch=compute_80,code=sm_80;
    -I${NVSHMEM_HOME}/include;
    >
)

target_link_libraries(Dist_Cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../bam/build/lib/libnvm.so /usr/mpi/gcc/openmpi-4.1.5a1/lib/libmpi.so ${NVSHMEM_LIBRARIES} cuda cudart pybind11::module)

SET_TARGET_PROPERTIES( Dist_Cache
  PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Dist_Cache"
)

  # Place the initialization file in the output directory for the Python
  # bindings. This will simplify the installation.
  CONFIGURE_FILE( Dist_Cache/__init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/Dist_Cache/__init__.py
  )

  # Ditto for the setup file.
  CONFIGURE_FILE( Dist_Cache/setup.py
    ${CMAKE_CURRENT_BINARY_DIR}/Dist_Cache/setup.py
  )






from setuptools import setup, Extension
import os

# Define the path to the shared object (.so) file
so_file_path = os.path.join(os.path.dirname(__file__), '../build/shared_UVA.so')

# Ensure the .so file exists
if not os.path.isfile(so_file_path):
    raise FileNotFoundError(f"Shared object file not found at {so_file_path}")

# Define the Python extension
shared_UVA_module = Extension(
    name="shared_UVA",  # Name of the Python module to be used
    sources=[],  # No source files needed; we are using the compiled .so file
    library_dirs=[os.path.dirname(so_file_path)],  # Directory where .so is located
    libraries=["shared_UVA"],  # Name of the shared object (without 'lib' prefix)
    runtime_library_dirs=[os.path.dirname(so_file_path)],  # Ensure runtime linker can find it
    extra_link_args=['-Wl,-rpath,' + os.path.dirname(so_file_path)]  # Ensure the runtime loader can find the .so at runtime
)

setup(
    name="shared_UVA_install",
    version="0.1",
    packages = ['shared_UVA'],
    package_dir={'shared_UVA': '../build'},  # Install the .so file from the build directory
    package_data={'shared_UVA': ['shared_UVA.so']},  # Include the shared object file
    include_package_data=True
#    package_data = {'': ['shared_UVA.so']}
    #package_dir = [os.path.join(os.path.dirname(__file__), '../build/shared_UVA.so')]
#    ext_modules=[shared_UVA_module],
)


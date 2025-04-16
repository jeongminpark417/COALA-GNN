from distutils.core import setup

import sys
if sys.version_info < (3,0):
  sys.exit('Sorry, Python < 3.0 is not supported')

setup(
  name        = 'COALA_GNN_Pybind11',
  version     = '1.0.0', # TODO: might want to use commit ID here
  packages    = [ 'COALA_GNN_Pybind' ],
  package_dir = {
    '': '${CMAKE_CURRENT_BINARY_DIR}'
  },
  package_data = {
    '': ['COALA_GNN_Pybind.so']
  }
)

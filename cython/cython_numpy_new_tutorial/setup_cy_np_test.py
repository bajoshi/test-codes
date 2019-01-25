from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("cython_np_test.pyx"),
    include_dirs=[numpy.get_include()]
)
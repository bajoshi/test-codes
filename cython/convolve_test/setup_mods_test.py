from distutils.core import setup, Extension

#from Cython.Compiler.Options import get_directive_defaults
#
#directive_defaults = get_directive_defaults()
#
#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("test_model_mods_cython.pyx"),
    include_dirs=[numpy.get_include()]
)
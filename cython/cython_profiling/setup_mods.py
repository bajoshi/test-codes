from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
defaults = Cython.Compiler.Options.get_directive_defaults()

defaults['linetrace'] = True
defaults['binding'] = True

extensions = [
Extension("model_mods", ["model_mods.pyx"],
    define_macros=[('CYTHON_TRACE', '1')]
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[numpy.get_include()]
)
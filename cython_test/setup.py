from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = 'simple convolution and for loop in cython',
	ext_modules = cythonize("convolve1.pyx")
)
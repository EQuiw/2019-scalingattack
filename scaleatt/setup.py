from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy
import sys

# define an extension that will be cythonized and compiled
ext = Extension(name="defenses.prevention.cythmodule.randomfiltering", sources=["defenses/prevention/cythmodule/randomfiltering.pyx"])

ext2 = Extension(name="defenses.prevention.cythmodule.medianfiltering",
                 sources=["defenses/prevention/cythmodule/medianfiltering.pyx"],
                 language='c++', extra_compile_args=["-std=c++11"])

ext3 = Extension(name="attack.adaptive_attack.cythmodule.adaptivemedianfiltering",
                 sources=["attack/adaptive_attack/cythmodule/adaptivemedianfiltering.pyx"],
                 language='c++', extra_compile_args=["-std=c++11"])

ext4 = Extension(name="attack.adaptive_attack.cythmodule.adaptiverandomfiltering",
                 sources=["attack/adaptive_attack/cythmodule/adaptiverandomfiltering.pyx"],
                 language='c++', extra_compile_args=["-std=c++11"])

setup(ext_modules=cythonize([ext, ext2, ext3, ext4], annotate=True, compiler_directives={'language_level': sys.version_info[0]}),
      include_dirs=[numpy.get_include()],
      )

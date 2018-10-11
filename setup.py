import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

def make_extension(package_path, name, include_dirs=None):
    if include_dirs is None:
        include_dirs = []
    cy_dir = os.path.join(*package_path)
    package_prefix = '.'.join(package_path)+'.'
    ext = Extension(package_prefix+name,
                [os.path.join(cy_dir, name+'.pyx')],
                include_dirs=include_dirs)
    return ext

extensions = []
# Tabular env base classes
tabular_pkg = ['rlutil', 'envs', 'tabular_cy']
extensions.append(make_extension(tabular_pkg, 'tabular_env'))
extensions.append(make_extension(tabular_pkg, 'q_iteration'))

# Tabular gridworld
grid_pkg = ['rlutil', 'envs', 'gridcraft']
extensions.append(make_extension(grid_pkg, 'grid_env_cy'))
extensions.append(make_extension(grid_pkg, 'grid_spec_cy'))

setup(
    ext_modules=cythonize(extensions)
)

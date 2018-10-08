import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

package_path = ['rlutil', 'envs', 'tabular_cy']
cy_env_dir = os.path.join(*package_path)
package_prefix = '.'.join(package_path)+'.'


extensions = [Extension(package_prefix+'tabular_env', 
                        [os.path.join(cy_env_dir, 'tabular_env.pyx')],
                        include_dirs=[]),
              Extension(package_prefix+'q_iteration', 
                        [os.path.join(cy_env_dir, 'q_iteration.pyx')],
                        include_dirs=[numpy.get_include()])
        ]

setup(
    ext_modules=cythonize(extensions)
)

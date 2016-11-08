# -*- coding: UTF-8 -*-

"""
This file is part of RT1.
(c) 2016- Alexander Loew
For COPYING and LICENSE details, please refer to the LICENSE file
"""

# good introduction into packing can be found in
# https://python-packaging-user-guide.readthedocs.org/en/latest/index.html

from setuptools import setup
from distutils.core import setup as setup_dist  # todo use only one setup

import os
import glob

# the setuptools are supposed to be used as a standard. Thats why we ommit
# usage of distutils here

# example of setup.py can be found here:
# https://github.com/pypa/sampleproject/blob/master/setup.py


# a small example how to build dependencies is given here:
# http://stackoverflow.com/questions/11010151/distributing-a-shared-library-and-some-c-code-with-a-cython-extension-module

import os
import numpy as np
import json

from setuptools import setup, Extension
from setuptools import find_packages  # Always prefer setuptools over distutils
from Cython.Distutils import build_ext
from Cython.Build import cythonize



# requires scipy:
# http://stackoverflow.com/questions/11128070/cannot-import-minimize-in-scipy
#~ install_requires = ["numpy>0.1", "cdo>1.2", "netCDF4", "pytz",
                    #~ "matplotlib", 'shapely', 'cartopy', 'cython', 'scipy']
install_requires = []



#~ def get_packages():
    #find_packages(exclude=['contrib', 'docs', 'tests*']),
    #~ return find_packages()


setup(name='rt1',

      version='0.1-dev',

      description='rt1 - bistatic single scattering radiative transfer model',

      # You can just specify the packages manually here if your project is
      # simple. Or you can use find_packages().
      # packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

      #~ packages=get_packages(),
      #~ package_dir={'pycmbs': 'pycmbs'},
      #~ package_data={'pycmbs': ['benchmarking/configuration/*',
                               #~ 'benchmarking/logo/*', 'version.json']},

      author="Alexander Loew",
      author_email='alexander.loew@lmu.de',
      maintainer='Alexander Loew',
      maintainer_email='alexander.loew@lmu.de',

      #~ license='APACHE 2',

      url='https://github.com/pygeo/rt1',

      long_description='xxxx',

      # List run-time dependencies here. These will be installed by pip when your
      # project is installed. For an analysis of "install_requires" vs pip's
      # requirements files see:
      # https://packaging.python.org/en/latest/technical.html#install-requires-vs-requirements-files
      install_requires=install_requires,

      keywords=["physics", "radiative transfer"],

      # To provide executable scripts, use entry points in preference to the
      # "scripts" keyword. Entry points provide cross-platform support and allow
      # pip to create the appropriate form of executable for the target
      # platform.

      #~ entry_points={
          #~ 'console_scripts': [
              #~ 'pycmbs_benchmarking = pycmbs_benchmarking:main'
          #~ ]},

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          # 3 - Alpha
          # 4 - Beta
          # 5 - Production/Stable
          # 'Development Status :: 4 - beta',
          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Atmospheric Science',

          # Pick your license as you wish (should match "license" above)
          #~ 'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 2.7'
      ],

      #~ ext_modules=[ext_polygon_utils],
      #~ cmdclass={'build_ext': build_ext}

    #~ ext_modules=cythonize(
        #~ ["./geoval/polygon/polygon_utils.pyx"]),
    # this is needed to get proper information on numpy headers
    #~ include_dirs=[np.get_include()]

      )




########################################################################
# Some useful information on shipping packages
########################################################################

# PIP
#~ python setup.py register
#~ python setup.py sdist
#~ python setup.py upload

# -*- coding: UTF-8 -*-

"""
This file is part of RT1.
(c) 2016- Raphael Quast
For COPYING and LICENSE details, please refer to the LICENSE file
"""

from setuptools import setup, find_packages
#from setuptools import find_packages
from rt1 import __version__

setup(name='rt1',

      version=__version__,

      description='RT1 - bistatic single scattering radiative transfer model',

      packages=find_packages(),  # ['rt1'],
      package_dir={'rt1': 'rt1'},
      include_package_data=False,

      author="Raphael Quast",
      author_email='raphael.quast@geo.tuwien.ac.at',
      maintainer='Raphael Quast',
      maintainer_email='raphael.quast@geo.tuwien.ac.at',

      #~ license='APACHE 2',

      url='https://github.com/TUW-GEO/rt1',

      long_description=('A module to perform forward-simulation and ' +
                        'parameter-inversion of incidence-angle dependent ' +
                        'backscatter observations based on a first-order ' +
                        'radiative-transfer model describing a rough surface' +
                        'covered by a homogeneous layer of scattering' +
                        'media.'),

      install_requires=["numpy>=1.16", "sympy>=1.4", "scipy>=1.2",
                        "pandas>=0.24", "matplotlib>=3.0"],
      extras_require={'full' : ["symengine>=0.4", "xarray>=0.16"]},

      keywords=["physics", "radiative transfer"],

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Atmospheric Science',

          # Pick your license as you wish (should match "license" above)
          #~ 'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3.7'
      ],

      )



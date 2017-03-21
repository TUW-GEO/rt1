# -*- coding: UTF-8 -*-

"""
This file is part of RT1.
(c) 2016- Alexander Loew
For COPYING and LICENSE details, please refer to the LICENSE file
"""

from setuptools import setup
from setuptools import find_packages  

install_requires = ["numpy", "sympy", "symengine"]  

def get_packages():
    find_packages(exclude=['contrib', 'docs', 'tests*']),
    return find_packages()


setup(name='rt1',

      version='0.1.2',

      description='rt1 - bistatic single scattering radiative transfer model',

      packages=get_packages(),
      package_dir={'rt1': 'rt1'},

      author="Alexander Loew",
      author_email='alexander.loew@lmu.de',
      maintainer='Alexander Loew',
      maintainer_email='alexander.loew@lmu.de',

      #~ license='APACHE 2',

      url='https://github.com/pygeo/rt1',

      long_description='xxxx',
      install_requires=install_requires,

      keywords=["physics", "radiative transfer"],

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Atmospheric Science',

          # Pick your license as you wish (should match "license" above)
          #~ 'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 2.7'
      ],

      )



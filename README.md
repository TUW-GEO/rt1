[![Windows_build](https://github.com/TUW-GEO/rt1/workflows/RT1_windows/badge.svg)](https://github.com/TUW-GEO/rt1/actions/workflows/RT1_windows.yml)
[![Ubuntu_build](https://github.com/TUW-GEO/rt1/workflows/RT1_ubuntu/badge.svg)](https://github.com/TUW-GEO/rt1/actions/workflows/RT1_ubuntu.yml)
[![Coverage Status](https://codecov.io/gh/TUW-GEO/rt1/branch/dev/graph/badge.svg?token=tVCw5zvIe3)](https://codecov.io/gh/TUW-GEO/rt1)
[![pypi version](https://img.shields.io/pypi/v/rt1)](https://pypi.org/project/rt1/)
[![Documentation Status](https://readthedocs.org/projects/rt1/badge/?version=latest)](http://rt1.readthedocs.io/)
[![DOI](https://zenodo.org/badge/69531751.svg)](https://zenodo.org/badge/latestdoi/69531751)

# RT1 - bistatic scattering model for first order scattering of random media

The package implements a first order scattering radiative transfer model
for random volume over ground as documented in *Quast & Wagner (2016)* and
*Quast, Albergel, Calvet, Wagner (2019)*

The documentation of the package is found [here](http://rt1.readthedocs.io/).
(Note that the documentation is still under construction.)

- 🔨 **rt1.rt1**

  - generic implementation of radiative transfer calculations for a random
    volume over ground

  - symbolic evaluation of first-order interaction contribution estimates

- ⚙️ **rt1.rtfits**

  - a flexible interface to perform parameter estimation from incidence-angle  
    dependent backscatter-timeseries via non-linear least-squares fitting routines.

  - integrates with a set of pre-defined plot-functions and interactive   
    widgets that can be used to quickly analyze the obtained results

- :rocket: **rt1.rtprocess**

  - a versatile interface to setup and run parallelized processing

- 📑 **rt1.rtparse**

  - a configparser to set up processing-routines via .ini files

- 🏞️ **rt1.volume** and **rt1.surface**

  - a collection of useful surface- and volume scattering functions



## Usage
Any usage of this code is subject to the following conditions:

❗ Full compliance with the license (see LICENSE file) is given  
❗ In publications or public presentations, credit should be given to the
  authors by:

  - citing the references below ✔️
  - pointing to this github repository ✔️

## Installation
For a basic install, the following line should be fine:

    pip install rt1

In order to get a significant speedup in the symbolic computations and enable
NetCDF functionalities, it is recommended to install the module with the
optional dependencies `symengine` and `xarray` by using:

    pip install rt1[full]

## References
* Quast & Wagner (2016): [doi:10.1364/AO.55.005379](https://doi.org/10.1364/AO.55.005379)
* Quast, Albergel, Calvet, Wagner (2019) : [doi:10.3390/rs11030285](https://doi.org/10.3390/rs11030285)


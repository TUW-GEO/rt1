[![Build Status](https://travis-ci.org/TUW-GEO/rt1.svg?branch=dev)](https://travis-ci.org/TUW-GEO/rt1) [![Documentation Status](https://readthedocs.org/projects/rt1/badge/?version=latest)](http://rt1.readthedocs.io/) [![Coverage Status](https://coveralls.io/repos/github/TUW-GEO/rt1/badge.svg?branch=dev)](https://coveralls.io/github/TUW-GEO/rt1?branch=dev)

# RT1 - bistatic scattering model for first order scattering of random media

The package implements a first order scattering radiative transfer model
for random volume over ground as documented in *Quast & Wagner (2016)* and
*Quast, Albergel, Calvet, Wagner (2019)*

The documentation of the package is found [here](http://rt1.readthedocs.io/).
(Note that the documentation is still under construction.)

## Usage

Any usage of this code is subject to the following conditions

* full compliance with the license (see LICENSE file) is given
* In publications or public presentations, credit should be given to the
  authors by mainly a) citing the references below, b) pointing to this
  github repository

## Installation
For a basic install, the following line should be fine:

    pip install rt1

In order to get a speedup in the symbolic computations,
the module can be installed with the optional symengine-dependency, i.e.:

    pip install rt1[symengine]

## References
* Quast & Wagner (2016): [doi:10.1364/AO.55.005379](https://doi.org/10.1364/AO.55.005379)
* Quast, Albergel, Calvet, Wagner (2019) : [doi:10.3390/rs11030285](https://doi.org/10.3390/rs11030285)


"""
class to provide coefficients for RT modelling
currently, the coefficients need to be calculated analytically first and implemented here
for each possible combination of phase function and BRDF
"""

import numpy as np


class Fn(object):
    """
    Main class for coefficient calculation
    """
    def __init__(self, **kwargs):
        pass


class RayleighIsotropic(Fn):
    """
    class for coefficients of
    * Rayleigh volume scattering phase function
    * Isotropic surface BRDF
    """
    def __init__(self, **kwargs):
        super(RayleighIsotropic, self).__init__(**kwargs)

    def fn(self, mu_i, nmax):
        """
        returns coefficients

        Parameters
        ----------
        mu_i : float
            cosine of incidence angle
        nmax : int
            truncation for coefficient
        """
        c = np.zeros(int(nmax))

        c[0] = (3./(16.*np.pi)) * (3.-mu_i**2.)
        c[1] = 0.
        c[2] = (3./(16.*np.pi)) * (3.*mu_i**2. - 1.)

        # remaining coefficients are zero

        return c





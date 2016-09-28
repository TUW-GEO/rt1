"""
Definition of BRDF functions
"""

import numpy as np


class Surface(object):
    """
    basic class
    """
    def __init__(self, **kwargs):
        pass

    def brdf(self, ctheta):
        """
        Calculate BRDF as function of geometry

        Parameters
        ----------
        ctheta : float
            cosine of scattering angle
        """
        assert False, 'This subroutine should be implemented on sub-class level'


class Isotropic(Surface):
    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)

    def brdf(self, ctheta):
        """
        define BRDF function

        Parameters
        ----------
        ctheta : float
            cosine of scattering angle
        """
        return 1./np.pi

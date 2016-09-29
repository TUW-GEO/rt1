"""
Definition of volume phase scattering functions
"""

import numpy as np

class Volume(object):
    def __init__(self, **kwargs):
        self.omega = kwargs.pop('omega', None)
        assert self.omega is not None, 'Single scattering albedo needs to be provided'

        self.tau = kwargs.pop('tau', None)
        assert self.tau is not None, 'Optical depth needs to be provided'

        assert self.omega >= 0.
        assert self.tau >= 0.

        if self.tau == 0.:
            assert self.omega == 0., 'ERROR: If optical depth is equal to zero, then OMEGA can not be larger than zero'



    def p(self, mu_0, mu_s):
        """
        phase function wrapper

        Parameters
        ----------
        ctheta : float
            cosine of scattering angle
        """
        assert False, 'phase function to be defined in sub-classes'



class Rayleigh(Volume):
    """
    class to define Rayleigh scattering
    """
    def __init__(self, **kwargs):
        super(Rayleigh, self).__init__(**kwargs)

    def p(self, ctheta):
        """
        Parameters
        ----------
        ctheta : float
            cosine of scattering angle
        """
        # calculate cosine of scattering angle
        return (3./(16.*np.pi)) * (1. + ctheta**2.)







"""
Definition of volume phase scattering functions
"""

import numpy as np
from scatter import Scatter
import sympy as sp

class Volume(Scatter):
    def __init__(self, **kwargs):
        self.omega = kwargs.pop('omega', None)
        assert self.omega is not None, 'Single scattering albedo needs to be provided'

        self.tau = kwargs.pop('tau', None)
        assert self.tau is not None, 'Optical depth needs to be provided'

        assert self.omega >= 0.
        assert self.omega <= 1.
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
        self._set_function()
        self._set_legcoefficients()


    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
        #def pfunkt(t0):
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')
        self._func = 3./(16.*sp.pi)*(1.+self.thetap(theta_i,theta_s,phi_i,phi_s)**2.)

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        n = sp.Symbol('n')
        self._legcoefs = (3./(16.*sp.pi))*((4./3.)*sp.KroneckerDelta(0,n)+(2./3.)*sp.KroneckerDelta(2,n))


    def p(self, ctheta):
        """
        Parameters
        ----------
        ctheta : float
            cosine of scattering angle
        """
        # calculate cosine of scattering angle
        return (3./(16.*np.pi)) * (1. + ctheta**2.)







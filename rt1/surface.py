"""
Definition of BRDF functions
"""

import numpy as np
from scatter import Scatter
import sympy as sp


class Surface(Scatter):
    """
    basic class
    """
    def __init__(self, **kwargs):
        pass

    def brdf(self, t0,ts,p0,ps):
        """
        Calculate BRDF as function of geometry

        Parameters
        ----------
        ctheta : float
            cosine of scattering angle

        Returns
        float
        """
        # define sympy objects
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')

        # replace arguments and evaluate expression
        return self._func.xreplace({theta_i:t0, theta_s:ts, phi_i:p0, phi_s:ps}).evalf()


class Isotropic(Surface):
    """
    define an isotropic surface
    """
    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        pass

    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
        #def pfunkt(t0):
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')
        self._func = 1./sp.pi   #<<<< todo gere a different cosine theta is required dthan for volume scattering phase function


class CosineLobe(Surface):
    """
    define a surface like in the example 1 of the paper
    which is a cosine lobe representation based on a 10-coefficient Legendre polynomial
    approximation
    """
    def __init__(self, **kwargs):
        super(CosineLobe, self).__init__(**kwargs)
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        #self.legcoefs = ((2.*n+1.)*15.*sp.sqrt(sp.pi))/(16.*sp.gamma((7.-n)/2.)*sp.gamma((8.+n)/2.))    # A13
        pass

    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
        #def pfunkt(t0):
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')

        self._func = sp.Max(self.thetaBRDF(theta_i,theta_s,phi_i,phi_s)**5., 0.)  # eq. A13


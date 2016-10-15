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





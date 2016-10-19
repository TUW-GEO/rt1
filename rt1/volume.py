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

    def p(self, t0,ts,p0,ps):
        """
        calculate phase function by subsituting current geometry in function
        and then evaluate result

        Parameters
        ----------
        geometries of angles
        to : theta incidence
        ts : theta scattering
        p0 : azimuth incident
        ps : azimuth scattering

        All in radians

        """
        # define sympy objects
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')

        # replace arguments and evaluate expression
        return self._func.xreplace({theta_i:t0, theta_s:ts, phi_i:p0, phi_s:ps}).evalf()

    def legexpansion(self):
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')

        NP = self.ncoefs
        n = sp.Symbol('n')
        return sp.Sum(self.legcoefs*sp.legendre(n,self.thetap(theta_i,theta_s,phi_i,phi_s)),(n,0,NP))  #.doit()  # this generates a code still that is not yet evaluated; doit() will result in GMMA error due to potential negative numbers


class Rayleigh(Volume):
    """
    class to define Rayleigh scattering function
    """
    def __init__(self, **kwargs):
        super(Rayleigh, self).__init__(**kwargs)
        self._set_function()
        self._set_legcoefficients()

    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
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
        self.ncoefs = 10
        n = sp.Symbol('n')
        self.legcoefs = ((3./(16.*sp.pi))*((4./3.)*sp.KroneckerDelta(0,n)+(2./3.)*sp.KroneckerDelta(2,n))).expand()






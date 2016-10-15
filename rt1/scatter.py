"""
Define general object for scattering calculations
This is the basis object for any Surface and Vollume objects
"""

import sympy as sp


class Scatter(object):
    def __init__(self):
        pass

    def thetaBRDF(self, thetai,thetas, phii, phis):
        """
        Parameters
        ----------
        incident and scattering angles as sympy symbols
        """
        return sp.cos(thetai)*sp.cos(thetas)+sp.sin(thetai)*sp.sin(thetas)*sp.cos(phii)*sp.cos(phis)+sp.sin(thetai)*sp.sin(thetas)*sp.sin(phii)*sp.sin(phis)

    def thetap(self, thetai, thetas, phii, phis):
        return -sp.cos(thetai)*sp.cos(thetas)+sp.sin(thetai)*sp.sin(thetas)*sp.cos(phii)*sp.cos(phis)+sp.sin(thetai)*sp.sin(thetas)*sp.sin(phii)*sp.sin(phis)

    def _get_legcoef(self, n0):
        """
        function to evaluate legendre coefficients
        used mainly for testing purposes
        the actual coefficient are used in the symbolic expansion
        """
        n = sp.Symbol('n')
        return self.legcoefs.xreplace({n:int(n0)}).evalf()


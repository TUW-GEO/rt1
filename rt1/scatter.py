"""
Define general object for scattering calculations
This is the basis object for any Surface and Vollume objects
"""

import sympy as sc


class Scatter(object):
    def __init__(self):
        pass

    def thetaBRDF(self, thetai,thetas,phii,phis):
        """
        Parameters
        ----------
        incident and scattering angles as sympy symbols
        """
        return sc.cos(thetai)*sc.cos(thetas)+sc.sin(thetai)*sc.sin(thetas)*sc.cos(phii)*sc.cos(phis)+sc.sin(thetai)*sc.sin(thetas)*sc.sin(phii)*sc.sin(phis)

    def thetap(self, thetai, thetas, phii, phis):
        return -sc.cos(thetai)*sc.cos(thetas)+sc.sin(thetai)*sc.sin(thetas)*sc.cos(phii)*sc.cos(phis)+sc.sin(thetai)*sc.sin(thetas)*sc.sin(phii)*sc.sin(phis)

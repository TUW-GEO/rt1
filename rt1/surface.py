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


    def legexpansion(self, t_i, t_ex, p_0, p_ex, geometry):
        assert self.ncoefs > 0

        """
        Definition of the legendre-expansion of the BRDF

        The geometry-parameter consists of 4 characters that define the
        geometry of the experiment-setup:

        The 4 characters represent in order: theta_i, theta_ex, phi_i, phi_ex

        'f' indicates that the angle is treated 'fixed'
        'v' indicates that the angle is treated 'variable'

        Passing  geometry = 'mono'  indicates a monstatic geometry
        (i.e.:  theta_i = theta_ex, phi_ex = phi_i + pi)
        """

        theta_s = sp.Symbol('theta_s')
        phi_s = sp.Symbol('phi_s')


        NBRDF = self.ncoefs
        n = sp.Symbol('n')

        # define sympy variables based on chosen geometry
        if geometry == 'mono':
            theta_i = sp.Symbol('theta_i')
            theta_ex = theta_i
            phi_ex = p_0 + sp.pi
        else:
            if geometry[0] == 'v':
                theta_i = sp.Symbol('theta_i')
            elif geometry[0] == 'f':
                theta_i = t_i
            else:
                raise AssertionError('wrong choice of theta_i geometry')

            if geometry[1] == 'v':
                theta_ex = sp.Symbol('theta_ex')
            elif geometry[1] == 'f':
                theta_ex = t_ex
            else:
                raise AssertionError('wrong choice of theta_ex geometry')

            if geometry[2] == 'v':
                phi_i = sp.Symbol('phi_i')
            elif geometry[2] == 'f':
                phi_i = p_0
            else:
                raise AssertionError('wrong choice of phi_i geometry')

            if geometry[3] == 'v':
                phi_ex = sp.Symbol('phi_ex')
            elif geometry[3] == 'f':
                phi_ex = p_ex
            else:
                raise AssertionError('wrong choice of phi_ex geometry')


        print 'BRDF: ', self.thetaBRDF(theta_s,theta_ex,phi_s,phi_ex)
        return sp.Sum(self.legcoefs*sp.legendre(n,self.thetaBRDF(theta_s,theta_ex,phi_s,phi_ex)),(n,0,NBRDF-1))  ###.doit()  # this generates a code still that is not yet evaluated; doit() will result in GMMA error due to potential negative numbers



class Isotropic(Surface):
    """
    define an isotropic surface
    """
    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        self.ncoefs = 1
        n = sp.Symbol('n')
        self.legcoefs = (1./sp.pi)*sp.KroneckerDelta(0,n)

    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
        #def pfunkt(t0):
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')
        self._func = 1./sp.pi


class CosineLobe(Surface):
    """
    define a surface like in the example 1 of the paper
    which is a cosine lobe representation based on a 10-coefficient Legendre polynomial
    approximation
    """
    def __init__(self, ncoefs=None, **kwargs):
        super(CosineLobe, self).__init__(**kwargs)
        assert ncoefs is not None, 'Error: number of coefficients needs to be provided!'
        assert ncoefs > 0
        self.ncoefs = int(ncoefs)  # is the maximum degree used for polynomlial (e.g. if ncoefs=2, then evaluations until P(n=2) are done --> 3 coefficients are used
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        n = sp.Symbol('n')
        self.legcoefs = (((2.*n+1.)*15.*sp.sqrt(sp.pi))/(16.*sp.gamma(((sp.Rational(7.)-n)/sp.Rational(2.)))*sp.gamma((sp.Rational(8.)+n)/sp.Rational(2.)))).expand()    # A13   The Rational(is needed as otherwise a Gamma function Pole error is issued)
        #self.legcoefs = (15.*sp.sqrt(sp.pi))/(16.*sp.gamma(((sp.Rational(7.)-n)/sp.Rational(2.)))*sp.gamma((sp.Rational(8.)+n)/sp.Rational(2.)))
        #~ print 'legcoefs: ', self.legcoefs
        #~ print 'coef(0)', self._get_legcoef(0)


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







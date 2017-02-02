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
        # set scattering angle generalization-matrix to 1 if it is not explicitly provided by the chosen class
        self.a = getattr(self, 'a', [1.,1.,1.])

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


    def legexpansion(self, mu_0, mu_ex, p_0, p_ex, geometry):
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
                theta_i = np.arccos(mu_0)
            else:
                raise AssertionError('wrong choice of theta_i geometry')

            if geometry[1] == 'v':
                theta_ex = sp.Symbol('theta_ex')
            elif geometry[1] == 'f':
                theta_ex = np.arccos(mu_ex)
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


        #print 'BRDF: ', self.thetaBRDF(theta_s,theta_ex,phi_s,phi_ex)
        return sp.Sum(self.legcoefs*sp.legendre(n,self.thetaBRDF(theta_s,theta_ex,phi_s,phi_ex, self.a)),(n,0,NBRDF-1))  ###.doit()  # this generates a code still that is not yet evaluated; doit() will result in GMMA error due to potential negative numbers




class BRDFfunction(Surface):
    """
    dummy-Surface-class object used to generate linear-combinations of BRDF-functions
    """
    def __init__(self, **kwargs):
        super(BRDFfunction, self).__init__(**kwargs)
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
        self._func = 0.

    def _set_legcoefficients(self):
        n = sp.Symbol('n')
        self.legcoefs = 0.



class Isotropic(Surface):
    """
    define an isotropic surface
    """
    def __init__(self, NormBRDF = 1. , **kwargs):
        super(Isotropic, self).__init__(**kwargs)
        self.NormBRDF = NormBRDF
        assert isinstance(NormBRDF,float), 'Error: NormBRDF must be a floating-point number'
        assert NormBRDF >= 0. , 'Error: NormBRDF must be greater than 0'
        self._set_function()
        self._set_legcoefficients()


    def _set_legcoefficients(self):
        self.ncoefs = 1
        n = sp.Symbol('n')
        self.legcoefs = (self.NormBRDF/sp.pi)*sp.KroneckerDelta(0,n)

    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
        #def pfunkt(t0):
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')
        self._func = self.NormBRDF/sp.pi





class CosineLobe(Surface):
    """
    define a generalized cosine-lobe of power i.

    the parameter a=[-,-,-] provides the diagonal-elements of the generalized scattering
    angle as defined in:
    'E.P.F. Lafortune et.al : Non-Linear Approximation of Reflectance Functions,
    Proceedings of SIGGRAPH'97, pages 117-126, 1997'


    if the a-parameter is not provided explicitly or if it is set 
    to a=[1.,1.,1.], LafortuneLobe is the equal to the ordinary CosineLobe.
    """

    def __init__(self, ncoefs=None, i=None, NormBRDF = 1. , a=[1.,1.,1.],  **kwargs):
        assert ncoefs is not None, 'Error: number of coefficients needs to be provided!'
        assert i is not None, 'Error: Cosine lobe power needs to be specified!'
        super(CosineLobe, self).__init__(**kwargs)
        assert ncoefs > 0
        self.i = i
        assert isinstance(self.i,int), 'Error: Cosine lobe power needs to be an integer!'
        assert i >= 0, 'ERROR: Power of Cosine-Lobe needs to be greater than 0'
        self.NormBRDF = NormBRDF
        assert isinstance(NormBRDF,float), 'Error: NormBRDF must be a floating-point number'
        assert NormBRDF >= 0. , 'Error: NormBRDF must be greater than 0'        
        self.a = a
        assert isinstance(self.a,list), 'Error: Generalization-parameter needs to be a list'
        assert len(a)==3, 'Error: Generalization-parameter list must contain 3 values'
        assert all(type(x)==float for x in a), 'Error: Generalization-parameter array must contain only floating-point values!'
        self.ncoefs = int(ncoefs)
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        n = sp.Symbol('n')
        self.legcoefs = self.NormBRDF/sp.pi * ((2**(-2-self.i)*(1+2*n)*sp.sqrt(sp.pi)*sp.gamma(1+self.i))/(sp.gamma((2-n+self.i)*sp.Rational(1,2))*sp.gamma((3+n+self.i)*sp.Rational(1,2))))    # A13   The Rational(is needed as otherwise a Gamma function Pole error is issued)

    def _set_function(self):
        """
        define phase function as sympy object for later evaluation
        """
        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')

        #self._func = sp.Max(self.thetaBRDF(theta_i,theta_s,phi_i,phi_s, a=self.a), 0.)**self.i  # eq. A13

        # alternative formulation avoiding the use of sp.Max()
        #     (this is done because   sp.lambdify('x',sp.Max(x), "numpy")   generates a function
        #      that can not interpret array inputs.)
        x = self.thetaBRDF(theta_i,theta_s,phi_i,phi_s, a=self.a)
        self._func = self.NormBRDF/sp.pi * (x*(1.+sp.sign(x))/2.)**self.i  # eq. A13





class HenyeyGreenstein(Surface):
    """
    class to define HenyeyGreenstein scattering function
    for use as BRDF approximation function.
    """
    def __init__(self, t=None, ncoefs=None, NormBRDF = 1. , a=[1.,1.,1.], **kwargs):
        assert t is not None, 't parameter needs to be provided!'
        assert ncoefs is not None, 'Number of coefficients needs to be specified'
        super(HenyeyGreenstein, self).__init__(**kwargs)
        self.t = t
        self.ncoefs = ncoefs
        assert self.ncoefs > 0
        self.NormBRDF = NormBRDF
        assert isinstance(NormBRDF,float), 'Error: NormBRDF must be a floating-point number'
        assert NormBRDF >= 0. , 'Error: NormBRDF must be greater than 0'                
        self.a = a
        assert isinstance(self.a,list), 'Error: Generalization-parameter needs to be a list'
        assert len(a)==3, 'Error: Generalization-parameter list must contain 3 values'
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
        self._func = self.NormBRDF * (1.-self.t**2.) / ((sp.pi)*(1.+self.t**2.-2.*self.t*self.thetaBRDF(theta_i,theta_s,phi_i,phi_s,self.a))**1.5)

    def _set_legcoefficients(self):
        n = sp.Symbol('n')
        self.legcoefs = self.NormBRDF * (1./(sp.pi)) * (2.*n+1)*self.t**n

















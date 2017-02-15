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
        # set scattering angle generalization-matrix to [1,1,1] if it is not explicitly provided by the chosen class.
        # this results in a peak in specular-direction which is suitable for describing surface BRDF's
        self.a = getattr(self, 'a', [1.,1.,1.])
        self.NormBRDF = kwargs.get('NormBRDF', 1.)
        assert isinstance(self.NormBRDF,float), 'Error: NormBRDF must be a floating-point number'
        assert self.NormBRDF >= 0. , 'Error: NormBRDF must be greater than 0'

        
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


        #print 'BRDF: ', self.scat_angle(theta_s,theta_ex,phi_s,phi_ex)
        return sp.Sum(self.legcoefs*sp.legendre(n,self.scat_angle(theta_s,theta_ex,phi_s,phi_ex, self.a)),(n,0,NBRDF-1))  ###.doit()  # this generates a code still that is not yet evaluated; doit() will result in GMMA error due to potential negative numbers





class LinCombSRF(Surface):
        '''
        Class to generate linear-combinations of volume-class elements
        '''
        def __init__(self, SRFchoices=None, **kwargs):
            '''
            Parameters
            ----------

            SRFchoices : [ [float, Surface]  ,  [float, Surface]  ,  ...]
                        a list that contains the the individual phase-functions (Surface-objects)
                        and the associated weighting-factors (floats) of the linear-combination.

            '''
            super(LinCombSRF, self).__init__(**kwargs)

            self.SRFchoices = SRFchoices
            self._set_function()
            self._set_legexpansion()

        def _set_function(self):
            """
            define phase function as sympy object for later evaluation
            """
            theta_i = sp.Symbol('theta_i')
            theta_s = sp.Symbol('theta_s')
            phi_i = sp.Symbol('phi_i')
            phi_s = sp.Symbol('phi_s')
            self._func = self._SRFcombiner()._func

        def _set_legexpansion(self):
            '''
            set legexpansion to the combined legexpansion
            '''
            self.ncoefs = self._SRFcombiner().ncoefs
            self.legexpansion = self._SRFcombiner().legexpansion



        def _SRFcombiner(self):
            '''
            Returns a Surface-class element based on an input-array of Surface-class elements.
            The array must be shaped in the form:
                SRFchoices = [  [ weighting-factor   ,   Surface-class element ]  ,  [ weighting-factor   ,   Surface-class element ]  , .....]

            ATTENTION: the .legexpansion()-function of the combined surface-class element is no longer related to its legcoefs (which are set to 0.)
                       since the individual legexpansions of the combined surface-class elements are possibly evaluated with a different a-parameter
                       of the generalized scattering angle! This does not affect any calculations, since the evaluation is exclusively based on the
                       use of the .legexpansion()-function.
            '''

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


            # initialize a combined phase-function class element
            SRFcomb = BRDFfunction()
            SRFcomb.ncoefs = max([SRF[1].ncoefs for SRF in self.SRFchoices])     # set ncoefs of the combined volume-class element to the maximum
                                                                    #   number of coefficients within the chosen functions.
                                                                    #   (this is necessary for correct evaluation of fn-coefficients)

            # find BRDF functions with equal a parameters
            equals = [np.where((np.array([VV[1].a for VV in self.SRFchoices])==tuple(V[1].a)).all(axis=1))[0] for V in self.SRFchoices]
            # evaluate index of BRDF-functions that have equal a parameter
            equal_a = list({tuple(row) for row in equals})          # find phase functions where a-parameter is equal


            # evaluation of combined expansion in legendre-polynomials
            dummylegexpansion = []
            for i in range(0,len(equal_a)):

                SRFdummy = BRDFfunction()
                SRFequal = np.take(self.SRFchoices,equal_a[i],axis=0)        # select SRF choices where a parameter is equal

                SRFdummy.ncoefs = max([SRF[1].ncoefs for SRF in SRFequal])  # set ncoefs to the maximum number within the choices with equal a-parameter

                for SRF in SRFequal:                                    # loop over phase-functions with equal a-parameter

                    # set parameters based on chosen phase-functions and evaluate combined legendre-expansion
                    SRFdummy.a = SRF[1].a
                    SRFdummy.NormBRDF = SRF[1].NormBRDF
                    SRFdummy._func = SRFdummy._func + SRF[1]._func * SRF[0]
                    SRFdummy.legcoefs = SRFdummy.legcoefs + SRF[1].legcoefs * SRF[0]

                dummylegexpansion = dummylegexpansion + [SRFdummy.legexpansion]

            # combine legendre-expansions for each a-parameter based on given combined legendre-coefficients
            SRFcomb.legexpansion = lambda mu_0,mu_ex,p_0,p_ex,geometry : np.sum([lexp(mu_0,mu_ex,p_0,p_ex,geometry) for lexp in dummylegexpansion])


            for SRF in self.SRFchoices:
                # set parameters based on chosen classes to define analytic function representation
                SRFcomb._func = SRFcomb._func + SRF[1]._func * SRF[0]
            return SRFcomb







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

    def __init__(self, ncoefs=None, i=None, a=[1.,1.,1.],  **kwargs):
        assert ncoefs is not None, 'Error: number of coefficients needs to be provided!'
        assert i is not None, 'Error: Cosine lobe power needs to be specified!'
        super(CosineLobe, self).__init__(**kwargs)
        assert ncoefs > 0
        self.i = i
        assert isinstance(self.i,int), 'Error: Cosine lobe power needs to be an integer!'
        assert i >= 0, 'ERROR: Power of Cosine-Lobe needs to be greater than 0'
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

        #self._func = sp.Max(self.scat_angle(theta_i,theta_s,phi_i,phi_s, a=self.a), 0.)**self.i  # eq. A13

        # alternative formulation avoiding the use of sp.Max()
        #     (this is done because   sp.lambdify('x',sp.Max(x), "numpy")   generates a function
        #      that can not interpret array inputs.)
        x = self.scat_angle(theta_i,theta_s,phi_i,phi_s, a=self.a)
        self._func = self.NormBRDF/sp.pi * (x*(1.+sp.sign(x))/2.)**self.i  # eq. A13





class HenyeyGreenstein(Surface):
    """
    class to define HenyeyGreenstein scattering function
    for use as BRDF approximation function.
    """
    def __init__(self, t=None, ncoefs=None, a=[1.,1.,1.], **kwargs):
        assert t is not None, 't parameter needs to be provided!'
        assert ncoefs is not None, 'Number of coefficients needs to be specified'
        super(HenyeyGreenstein, self).__init__(**kwargs)
        self.t = t
        self.ncoefs = ncoefs
        assert self.ncoefs > 0

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
        self._func = self.NormBRDF * (1.-self.t**2.) / ((sp.pi)*(1.+self.t**2.-2.*self.t*self.scat_angle(theta_i,theta_s,phi_i,phi_s,self.a))**1.5)

    def _set_legcoefficients(self):
        n = sp.Symbol('n')
        self.legcoefs = self.NormBRDF * (1./(sp.pi)) * (2.*n+1)*self.t**n

















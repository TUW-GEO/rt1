"""
Definition of volume phase scattering functions
"""

import numpy as np
from scatter import Scatter
import sympy as sp

class Volume(Scatter):
    def __init__(self, **kwargs):
        self.omega = kwargs.pop('omega', None)
        self.tau = kwargs.pop('tau', None)
     
        # set scattering angle generalization-matrix to [-1,1,1] if it is not explicitly provided by the chosen class
        # this results in a peak in forward-direction which is suitable for describing volume-scattering phase-functions
        self.a = getattr(self, 'a', [-1.,1.,1.])

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

    def legexpansion(self,mu_0,mu_ex,p_0,p_ex,geometry):
        assert self.ncoefs > 0

        """
        Definition of the legendre-expansion of the volume-phase-function.

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

        NP = self.ncoefs
        n = sp.Symbol('n')

        # define sympy variables based on chosen geometry
        if geometry == 'mono':
            theta_i = sp.Symbol('theta_i')
            theta_ex = theta_i
            phi_i = p_0
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

        #correct for backscattering
        return sp.Sum(self.legcoefs*sp.legendre(n,self.scat_angle(sp.pi-theta_i,theta_s,phi_i,phi_s, self.a)),(n,0,NP-1))  #.doit()  # this generates a code still that is not yet evaluated; doit() will result in GMMA error due to potential negative numbers




class LinCombV(Volume):
        '''
        Class to generate linear-combinations of volume-class elements
        '''
        def __init__(self, Vchoices=None, **kwargs):
            '''
            Parameters
            ----------

            tau : float
                optical depth of the combined phase-function
                ATTENTION: tau-values provided within the Vchoices-list will not be considered!
            omega : float
                single scattering albedo of the combined phase-function
                ATTENTION: omega-values provided within the Vchoices-list will not be considered!

            Vchoices : [ [float, Volume]  ,  [float, Volume]  ,  ...]
                     a list that contains the the individual phase-functions (Volume-objects)
                     and the associated weighting-factors (floats) of the linear-combination.
                     ATTENTION: since the normalization of the phase-function is fixed, the weighting-factors must equate to 1 !

            '''
            super(LinCombV, self).__init__(**kwargs)

            self.Vchoices = Vchoices
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
            self._func = self._Vcombiner()._func

        def _set_legexpansion(self):
            '''
            set legexpansion to the combined legexpansion
            '''
            self.ncoefs = self._Vcombiner().ncoefs
            self.legexpansion = self._Vcombiner().legexpansion


        def _Vcombiner(self):
            '''
            Returns a Volume-class element based on an input-array of Volume-class elements.
            The array must be shaped in the form:
                Vchoices = [  [ weighting-factor   ,   Volume-class element ]  ,  [ weighting-factor   ,   Volume-class element ]  , .....]

            In order to keep the normalization of the phase-functions correct,
            the sum of the weighting factors must equate to 1!


            ATTENTION: the .legexpansion()-function of the combined volume-class element is no longer related to its legcoefs (which are set to 0.)
                       since the individual legexpansions of the combined volume-class elements are possibly evaluated with a different a-parameter
                       of the generalized scattering angle! This does not affect any calculations, since the evaluation is exclusively based on the
                       use of the .legexpansion()-function.

            '''

            class Phasefunction(Volume):
                """
                dummy-Volume-class object used to generate linear-combinations of volume-phase-functions
                """
                def __init__(self, **kwargs):
                    super(Phasefunction, self).__init__(**kwargs)
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
                    """
                    set Legrende coefficients
                    needs to be a function that can be later evaluated by subsituting 'n'
                    """

                    n = sp.Symbol('n')
                    self.legcoefs = 0.


            # test if the weighting-factors equate to 1.
            np.testing.assert_almost_equal(desired = 1.,actual = np.sum([V[0] for V in self.Vchoices]), verbose = False, err_msg='The sum of the phase-function weighting-factors must equate to 1 !'),

            # find phase functions with equal a parameters
            equals = [np.where((np.array([VV[1].a for VV in self.Vchoices])==tuple(V[1].a)).all(axis=1))[0] for V in self.Vchoices]
            # evaluate index of phase-functions that have equal a parameter
            equal_a = list({tuple(row) for row in equals})

            # initialize a combined phase-function class element
            Vcomb = Phasefunction(tau=self.tau, omega=self.omega)           # set tau and omega to the values for the combined phase-function
            Vcomb.ncoefs = max([V[1].ncoefs for V in self.Vchoices])        # set ncoefs of the combined volume-class element to the maximum
                                                                            #   number of coefficients within the chosen functions.
                                                                            #   (this is necessary for correct evaluation of fn-coefficients)

            # evaluation of combined expansion in legendre-polynomials
            dummylegexpansion = []
            for i in range(0,len(equal_a)):

                Vdummy = Phasefunction()
                Vequal = np.take(self.Vchoices,equal_a[i],axis=0)       # select V choices where a parameter is equal

                Vdummy.ncoefs = max([V[1].ncoefs for V in Vequal])      # set ncoefs to the maximum number within the choices with equal a-parameter

                for V in Vequal:                                        # loop over phase-functions with equal a-parameter

                    # set parameters based on chosen phase-functions and evaluate combined legendre-expansion
                    Vdummy.a = V[1].a
                    Vdummy._func = Vdummy._func + V[1]._func * V[0]
                    Vdummy.legcoefs = Vdummy.legcoefs + V[1].legcoefs * V[0]

                dummylegexpansion = dummylegexpansion + [Vdummy.legexpansion]

            # combine legendre-expansions for each a-parameter based on given combined legendre-coefficients
            Vcomb.legexpansion = lambda mu_0,mu_ex,p_0,p_ex,geometry : np.sum([lexp(mu_0,mu_ex,p_0,p_ex,geometry) for lexp in dummylegexpansion])


            for V in self.Vchoices:
                # set parameters based on chosen classes to define analytic function representation
                Vcomb._func = Vcomb._func + V[1]._func * V[0]

            return Vcomb








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
        self._func = 3./(16.*sp.pi)*(1.+self.scat_angle(theta_i,theta_s,phi_i,phi_s, self.a)**2.)

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        self.ncoefs = 3    #only 3 coefficients are needed to correctly represent the Rayleigh scattering function
        n = sp.Symbol('n')
        self.legcoefs = ((3./(16.*sp.pi))*((4./3.)*sp.KroneckerDelta(0,n)+(2./3.)*sp.KroneckerDelta(2,n))).expand()



class HenyeyGreenstein(Volume):
    """
    class to define HenyeyGreenstein scattering function
    """
    def __init__(self, t=None, ncoefs=None, a=[-1.,1.,1.] , **kwargs):
        assert t is not None, 't parameter needs to be provided!'
        assert ncoefs is not None, 'Number of coefficients needs to be specified'
        super(HenyeyGreenstein, self).__init__(**kwargs)
        self.t = t
        self.a = a
        assert isinstance(self.a,list), 'Error: Generalization-parameter needs to be a list'
        assert len(a)==3, 'Error: Generalization-parameter list must contain 3 values'
        assert all(type(x)==float for x in a), 'Error: Generalization-parameter array must contain only floating-point values!'
        self.ncoefs = ncoefs
        assert self.ncoefs > 0
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
        self._func = (1.-self.t**2.) / ((4.*sp.pi)*(1.+self.t**2.-2.*self.t*self.scat_angle(theta_i,theta_s,phi_i,phi_s,self.a))**1.5)

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        n = sp.Symbol('n')
        self.legcoefs = (1./(4.*sp.pi)) * (2.*n+1)*self.t**n





class HGRayleigh(Volume):
    """
    class to define HenyeyGreenstein-Rayleigh scattering function as proposed in:
    
    'Quanhua Liu and Fuzhong Weng: Combined henyey-greenstein and rayleigh phase function,
    Appl. Opt., 45(28):7475-7479, Oct 2006. doi: 10.1364/AO.45.'
    """

    def __init__(self, t=None, ncoefs=None, a=[-1.,1.,1.] , **kwargs):
        assert t is not None, 't parameter needs to be provided!'
        assert ncoefs is not None, 'Number of coefficients needs to be specified'
        super(HGRayleigh, self).__init__(**kwargs)
        self.t = t
        self.a = a
        assert isinstance(self.a,list), 'Error: Generalization-parameter needs to be a list'
        assert len(a)==3, 'Error: Generalization-parameter list must contain 3 values'
        assert all(type(x)==float for x in a), 'Error: Generalization-parameter array must contain only floating-point values!'
        self.ncoefs = ncoefs
        assert self.ncoefs > 0
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
        self._func = 3./(8.*sp.pi)*1./(2.+self.t**2)*(1+self.scat_angle(theta_i,theta_s,phi_i,phi_s, self.a)**2)*(1.-self.t**2.) / ((1.+self.t**2.-2.*self.t*self.scat_angle(theta_i,theta_s,phi_i,phi_s, self.a))**1.5)


    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        n = sp.Symbol('n')
        self.legcoefs = sp.Piecewise(
        (3./(8.*sp.pi) * 1./(2.+self.t**2) * ((n+2.)*(n+1.)/(2.*n+3)*self.t**(n+2.) + (n+1.)**2./(2.*n+3.)*self.t**n + (5.*n**2.-1.)/(2.*n-1.)*self.t**n)
        ,n<2),
        (3./(8.*sp.pi) * 1./(2.+self.t**2) * (n*(n-1.)/(2.*n-1.) * self.t**(n-2.) + (n+2.)*(n+1.)/(2.*n+3)*self.t**(n+2.) + (n+1.)**2./(2.*n+3.)*self.t**n + (5.*n**2.-1.)/(2.*n-1.)*self.t**n)
        ,True)
        )










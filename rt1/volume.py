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
        
        # set scattering angle generalization-matrix to standard value if it is not explicitly provided by the chosen class
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
        return sp.Sum(self.legcoefs*sp.legendre(n,self.thetap(sp.pi-theta_i,theta_s,phi_i,phi_s, self.a)),(n,0,NP-1))  #.doit()  # this generates a code still that is not yet evaluated; doit() will result in GMMA error due to potential negative numbers


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
        self._func = (1.-self.t**2.) / ((4.*sp.pi)*(1.+self.t**2.-2.*self.t*self.thetap(theta_i,theta_s,phi_i,phi_s,self.a))**1.5)

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
        self._func = 3./(8.*sp.pi)*1./(2.+self.t**2)*(1+self.thetap(theta_i,theta_s,phi_i,phi_s, self.a)**2)*(1.-self.t**2.) / ((1.+self.t**2.-2.*self.t*self.thetap(theta_i,theta_s,phi_i,phi_s, self.a))**1.5)


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







class DoubleHG(Volume):
    """
    class to define a generalized double-HenyeyGreenstein scattering function
    """
    def __init__(self, t1=None, t2=None, fraction=None,  ncoefs=None, a=[-1.,1.,1.], **kwargs):
        assert t1 is not None, 't1 parameter needs to be provided!'
        assert t2 is not None, 't2 parameter needs to be provided!'
        assert fraction is not None, 'fraction parameter needs to be provided!'
        assert ncoefs is not None, 'Number of coefficients needs to be specified'
        super(DoubleHG, self).__init__(**kwargs)
        self.t1 = t1
        self.t2 = t2
        self.fraction = fraction
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
        self._func = self.fraction*(1.-self.t1**2.) / ((4.*sp.pi)*(1.+self.t1**2.-2.*self.t1*self.thetap(theta_i,theta_s,phi_i,phi_s, self.a))**1.5) + (1.-self.fraction)*(1.-self.t2**2.) / ((4.*sp.pi)*(1.+self.t2**2.-2.*self.t2*self.thetap(theta_i,theta_s,phi_i,phi_s, self.a))**1.5)

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        n = sp.Symbol('n')
        self.legcoefs = (1./(4.*sp.pi)) * (2.*n+1)*(self.fraction*self.t1**n + (1.-self.fraction)*self.t2**n)












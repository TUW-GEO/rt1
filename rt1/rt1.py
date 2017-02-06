"""
Core module for implementation of 1st order scattering model
using arbitrary BRDF and phase functions

References
----------
Quast & Wagner (2016): doi:10.1364/AO.55.005379
"""

import numpy as np
from scipy.special import expi
from scipy.special import expn

from sympy.simplify.fu import TR5

import sympy as sp
import time

import multiprocessing

def _get_fn_wrapper1(x):
    return _get_fn_wrapper(x[0], x[1], x[2], x[3], x[4], x[5])

def _get_fn_wrapper(fn, n, t0, p0, tex, pex):
    """
    function to evaluate expansion coefficients
    as function of incident geometry

    independent of class
    """
    theta_i = sp.Symbol('theta_i')
    phi_i = sp.Symbol('phi_i')
    theta_ex = sp.Symbol('theta_ex')
    phi_ex = sp.Symbol('phi_ex')
    # potential speed up here through evaluation of sin cosine functions
    # only once
    # print t0
    # print fn[n].xreplace({theta_i:t0, phi_i:p0, theta_ex:tex, phi_ex:pex})
    return fn[n].xreplace({theta_i:t0, phi_i:p0, theta_ex:tex, phi_ex:pex}).evalf()

    


class RT1(object):
    """
    main class to perform RT simulations
    """

    def __init__(self, I0, mu_0, mu_ex, phi_0, phi_ex, RV=None, SRF=None, fn=None, geometry='vvvv', ncpu=None):
        """
        Parameters
        ----------
        I0 : float
            incidence radiation
        RV : Volume
            random volume object or array of weighting-factors (w_i) and volume-objects (RV_i)
            shaped in the form  RV = [ [w_1, RV_1], [w_2, RV_2], ...]    with   sum(w_i) = 1
        SRF: Surface
            random surface object or array of weighting-factors (w_i) and surface-objects (SRF_i)
            shaped in the form  SRF = [ [w_1, SRF_1], [w_2, SRF_2], ...]
        fn : sympy expression
            precalculated coefficient expression; otherwise it will be automatically calculated
            usefull for speedup when caling with different geometries
        geometry : str
            4 character string specifying which components of the angles should be fixed or variable
            This is done to significantly speed up the calculations in the coefficient calculations

            mono = monostatic configuration
            v
            f
            TODO --> describe setups here
        """
        self.I0 = I0
        self.mu_0 = mu_0
        self.mu_ex = mu_ex
        self.phi_0 = phi_0
        self.phi_ex = phi_ex

        self.geometry = geometry
        assert isinstance(geometry,str), 'ERROR: geometry must be a 4-character string'
        assert len(self.geometry) == 4


        assert RV is not None, 'ERROR: needs to provide volume information'

        # if an array is provided for RV, call Vcombiner function to generate a combined phase-function element
        if isinstance(RV,(list,np.ndarray)):
            self.RV = self._Vcombiner(RV)
        else:
            self.RV = RV


        assert SRF is not None, 'ERROR: needs to provide surface information'

        # if an array is provided for SRF, call SRFcombiner function to generate a combined BRDF-function element
        if isinstance(SRF,(list,np.ndarray)):
            self.SRF = self._SRFcombiner(SRF)
        else:
            self.SRF = SRF


        if ncpu is None:
            self.ncpu = multiprocessing.cpu_count()
        else:
            self.ncpu = ncpu
        #~ print 'NCPU: ', self.ncpu


        if fn is None:
        # precalculate the expansiion coefficients for the interaction term
            expr_int = self._calc_interaction_expansion()

            # now we have the integral formula ready. The next step is now to
            # extract the expansion coefficients
            #~ print 'Integral expansion before extraction:'
            self.fn = self._extract_coefficients(expr_int)
        else:
            self.fn = fn


    def _get_theta0(self):
        return np.arccos(self.mu_0)
    theta_0 = property(_get_theta0)

    def _get_thetaex(self):
        return np.arccos(self.mu_ex)
    theta_ex = property(_get_thetaex)

    def _extract_coefficients(self, expr):
        """
        extract Fn coefficients from given forumula
        This is done by setting all the exponents to zero and look at the
        remainer for each power of cosine
        """
        theta_s = sp.Symbol('theta_s')
        replacementsnull= {sp.cos(theta_s) : 0.}

        # construct a list of coefficients
        fn=[]
        fn= fn + [expr.xreplace(replacementsnull)]

        for nn in range(1,self.SRF.ncoefs+self.RV.ncoefs+1):
            replacementsnn = [(sp.cos(theta_s)**i,0.)  for i in range(1,self.SRF.ncoefs+self.RV.ncoefs+1) if i !=nn]  # replace integer exponents
            replacementsnn = replacementsnn + [(sp.cos(theta_s)**float(i),0.)  for i in range(1,self.SRF.ncoefs+self.RV.ncoefs+1) if i !=nn]  # replace float exponents
            replacementsnn = dict(replacementsnn + [(sp.cos(theta_s)**nn,1.)] + [(sp.cos(theta_s)**float(nn),1.)]   )

            fn = fn + [(expr.xreplace(replacementsnn)-fn[0])]

                                              
        # simplify gained coefficients for faster evaluation
        # the TR5 function performs the replacement sin^2(x) = 1-cos(x)^2 to get rid of remaining sin(x)-terms
        # this results in a significant speedup for monostatic evaluations (and a moderate effect on bistatic calculations)
        fn = [TR5(i, max=self.SRF.ncoefs+self.RV.ncoefs+1).expand() for i in fn]               
        return fn


    def _calc_interaction_expansion(self):
        """
        calculate expensions to be able to analytically estimate Eq.23 needed for the interaction term
        The approach is as follows
        1) expand the legrende coefficents of the surface and volume phase functions
        2) replace the cosine terms in the Legrende polynomials by sines which corresponds to the intergration between 0 and 2*pi
        """
        # preevaluate expansions for volume and surface phase functions
        # this returns symbolic code to be then further used

        volexp = self.RV.legexpansion(self.mu_0,self.mu_ex,self.phi_0,self.phi_ex,self.geometry).doit()
        brdfexp = self.SRF.legexpansion(self.mu_0,self.mu_ex,self.phi_0,self.phi_ex,self.geometry).doit()
        #   preparation of the product of p*BRDF for coefficient retrieval
        fPoly =(2*sp.pi*volexp*brdfexp).expand().doit()  # this is the eq.23. and would need to be integrated from 0 to 2pi




        #~print fPoly
        #~print volexp
        #~print brdfexp


        # do integration of eq. 23
        expr = self._integrate_0_2pi_phis(fPoly)

        # now we do still simplify the expression to be able to express things as power series of cos(theta_s)
        theta_s = sp.Symbol('theta_s')
        replacements = [(sp.sin(theta_s)**i,((1.-sp.cos(theta_s)**2.)**sp.Rational(i,2)).expand())  for i in range(1,self.SRF.ncoefs+self.RV.ncoefs+1) if i % 2 == 0]
        res = expr.xreplace(dict(replacements)).expand()

        # o.k., by now we have the integral formulation ready.
        #~ print res
        return res


    def _cosintegral(self, i):
        """
        integral of cos(x)**i in the interval 0 ... 2*pi
        """
        if i % 2==0:
            return (2**(-i))*sp.binomial(i,i*sp.Rational(1,2))
        else:
            # for odd exponents result is always zero
            return 0.


    def _integrate_0_2pi_phis(self, expr):
        """
        integrate from zero to 2pi for phi_s
        and return similified expression
        """
        phi_s = sp.Symbol('phi_s')

        # replace first all odd powers of sin(phi_s) as these are all zero for the integral
        replacements1 = [(sp.sin(phi_s)**i, 0.) for i in range(1,self.SRF.ncoefs+self.RV.ncoefs+1) if i % 2 == 1]

        # then substitute the sine**2 by 1-cos**2
        replacements1 = replacements1 + [(sp.sin(phi_s)**i, ((1.-sp.cos(phi_s)**2)**sp.Rational(i,2)).expand()) for i in range(2,self.SRF.ncoefs+self.RV.ncoefs+1) if i % 2 == 0]
        res = expr.xreplace(dict(replacements1)).expand()

        # replacements need to be done simultaneously, otherwise all remaining sin(phi_i)**even will be replaced by 0

        # integrate the cosine terms
        replacements3 = [(sp.cos(phi_s)**i,self._cosintegral(i)) for i in range(1,self.SRF.ncoefs+self.RV.ncoefs+1)]
        res = res.xreplace(dict(replacements3)).expand()
        return res





    def _get_fn(self, n, t0, p0, tex, pex):
        """ wrapper function is used to have no effect on tests, external function needed for parallelization """
        return _get_fn_wrapper(self.fn, n, t0, p0, tex, pex)





    def calc(self):
        """
        Perform actual calculation of bistatic scattering at top of the random volume (z=0; tau(z) = 0)

        Returns
        -------
        specific intensities Itot, Isurf, Ivol, Iint
        """
        # (16)
        Isurf = self.surface()
        if self.RV.tau > 0.:  # explicit differentiation for non-existing canopy, as otherwise NAN values
            Ivol = self.volume()
            Iint = self.interaction()
        else:
            Ivol = 0.
            Iint = 0.
        return Isurf + Ivol + Iint, Isurf, Ivol, Iint

    def surface(self):
        """
        (17)
        """
        return self.I0 * np.exp(-(self.RV.tau / self.mu_0) - (self.RV.tau/self.mu_ex)) * self.mu_0 * self.SRF.brdf(self.theta_0, self.theta_ex, self.phi_0, self.phi_ex)

    def volume(self):
        """
        (18)
        """
        return (self.I0*self.RV.omega*self.mu_0/(self.mu_0+self.mu_ex)) * (1.-np.exp(-(self.RV.tau/self.mu_0)-(self.RV.tau/self.mu_ex))) * self.RV.p(self.theta_0, self.theta_ex, self.phi_0, self.phi_ex)

    def interaction(self):
        """
        (19)
        """
        Fint1 = self._calc_Fint(self.mu_0, self.mu_ex, self.phi_0, self.phi_ex)
        Fint2 = self._calc_Fint(self.mu_ex, self.mu_0, self.phi_ex, self.phi_0)
        return self.I0 * self.mu_0 * self.RV.omega * (np.exp(-self.RV.tau/self.mu_ex) * Fint1 + np.exp(-self.RV.tau/self.mu_0)*Fint2 )

    def _calc_Fint(self, mu1, mu2, phi1, phi2):
        """
        (37)
        first order interaction term

        in the original paper there is no dependency on PHI, but here it is
        as the we don not assume per se that PHI1=0 like it is done in the
        mansucript.
        """
        #~ S = 0.
        nmax = self.SRF.ncoefs+self.RV.ncoefs+1

        hlp1 = np.exp(-self.RV.tau/mu1)*np.log(mu1/(1.-mu1)) - expi(-self.RV.tau) + np.exp(-self.RV.tau/mu1)*expi(self.RV.tau/mu1-self.RV.tau)

        #~ if False:
            #~ # standard way
            #~ for n in xrange(nmax):
#~
                #~ S2 = np.sum(mu1**(-k) * (expn(k+1., self.RV.tau) - np.exp(-self.RV.tau/mu1)/k) for k in range(1,(n+1)+1))
                #~ fn = self._get_fn(n, np.arccos(mu1), phi1, np.arccos(mu2), phi2)
                #~ S += fn * mu1**(n+1) * (hlp1 + S2)
        #~ else:
        # hopefully faster
        # try to seaparate loops
        S2 = np.array([np.sum(mu1**(-k) * (expn(k+1., self.RV.tau) - np.exp(-self.RV.tau/mu1)/k) for k in range(1,(n+1)+1)) for n in xrange(nmax)])

        if True:  # regular processing
            fn = np.array([_get_fn_wrapper(self.fn, n, np.arccos(mu1), phi1, np.arccos(mu2), phi2) for n in xrange(nmax)])   # this is the by far slowes part!!
        else:  # parallel processing (IS MUCH SLOWER AT THE MOMENT!! REAL OVERHEAD DUE TO PARALLELIZATION)
            pool = multiprocessing.Pool(processes=self.ncpu)
            args = [(self.fn, n, np.arccos(mu1), phi1, np.arccos(mu2), phi2) for n in xrange(nmax)]
            fn = pool.map(_get_fn_wrapper1, args)

        #fn = np.random.random(nmax)
        mu = np.array([mu1**(n+1) for n in xrange(nmax)])
        S = np.sum(fn * mu * (S2 + hlp1))

        return S


#~ np.sum(fnfunktexp(n,t0)*CC(n+1,tau,t0) for n in range(0,Np+NBRDF+1))


#   function that evaluates the coefficients
#~ def fnfunktexp(n,t0):
    #~ return fn[n].xreplace({thetaex:t0})
#~
#~
#~ #   definition of surface- volume and first-order interaction-term
#~ def CC(n,tau,tex):
    #~ if n==0:
        #~ return np.exp(-tau/np.cos(tex))*np.log(np.cos(tex)/(1-np.cos(tex)))-scipy.special.expi(-tau)+np.exp(-tau/np.cos(tex))*scipy.special.expi(tau/np.cos(tex)-tau)
    #~ else:
        #~ return CC(n-1,tau,tex)*np.cos(tex)-(np.exp(-tau/np.cos(tex))/n-scipy.special.expn(n+1,tau))
#~
#~ def intback(t0,tau,omega):
    #~ return omega*np.cos(t0)*np.exp(-tau/np.cos(t0))*np.sum(fnfunktexp(n,t0)*CC(n+1,tau,t0) for n in range(0,Np+NBRDF+1))




    def _Vcombiner(self, Vchoices):
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
        from volume import Phasefunction

        # test if the weighting-factors equate to 1.
        np.testing.assert_almost_equal(desired = 1.,actual = np.sum([V[0] for V in Vchoices]), verbose = False, err_msg='The sum of the phase-function weighting-factors must equate to 1 !'),

        # find phase functions with equal a parameters
        equals = [np.where((np.array([VV[1].a for VV in Vchoices])==tuple(V[1].a)).all(axis=1))[0] for V in Vchoices]
        # evaluate index of phase-functions that have equal a parameter
        equal_a = list({tuple(row) for row in equals})

        # initialize a combined phase-function class element
        Vcomb = Phasefunction(tau=0.7, omega=0.3)
        Vcomb.ncoefs = max([V[1].ncoefs for V in Vchoices])     # set ncoefs of the combined volume-class element to the maximum
                                                                #   number of coefficients within the chosen functions.
                                                                #   (this is necessary for correct evaluation of fn-coefficients)

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0,len(equal_a)):

            Vdummy = Phasefunction(tau=0.7, omega=0.3)
            Vequal = np.take(Vchoices,equal_a[i],axis=0)        # select V choices where a parameter is equal

            Vdummy.ncoefs = max([V[1].ncoefs for V in Vequal])  # set ncoefs to the maximum number within the choices with equal a-parameter

            for V in Vequal:                                    # loop over phase-functions with equal a-parameter

                # set parameters based on chosen phase-functions and evaluate combined legendre-expansion
                Vdummy.a = V[1].a
                Vdummy.tau = V[1].tau
                Vdummy.omega = V[1].omega
                Vdummy._func = Vdummy._func + V[1]._func * V[0]
                Vdummy.legcoefs = Vdummy.legcoefs + V[1].legcoefs * V[0]

            dummylegexpansion = dummylegexpansion + [Vdummy.legexpansion]

        # combine legendre-expansions for each a-parameter based on given combined legendre-coefficients
        Vcomb.legexpansion = lambda mu_0,mu_ex,p_0,p_ex,geometry : np.sum([lexp(mu_0,mu_ex,p_0,p_ex,geometry) for lexp in dummylegexpansion])


        for V in Vchoices:
            # set parameters based on chosen classes to define analytic function representation
            Vcomb.tau = V[1].tau
            Vcomb.omega = V[1].omega
            Vcomb._func = Vcomb._func + V[1]._func * V[0]

        return Vcomb




    def _SRFcombiner(self, SRFchoices):
        '''
        Returns a Surface-class element based on an input-array of Surface-class elements.
        The array must be shaped in the form:
            SRFchoices = [  [ weighting-factor   ,   Surface-class element ]  ,  [ weighting-factor   ,   Surface-class element ]  , .....]


        ATTENTION: the .legexpansion()-function of the combined surface-class element is no longer related to its legcoefs (which are set to 0.)
                   since the individual legexpansions of the combined surface-class elements are possibly evaluated with a different a-parameter
                   of the generalized scattering angle! This does not affect any calculations, since the evaluation is exclusively based on the
                   use of the .legexpansion()-function.

        '''

        from surface import BRDFfunction


        # initialize a combined phase-function class element
        SRFcomb = BRDFfunction()
        SRFcomb.ncoefs = max([SRF[1].ncoefs for SRF in SRFchoices])     # set ncoefs of the combined volume-class element to the maximum
                                                                #   number of coefficients within the chosen functions.
                                                                #   (this is necessary for correct evaluation of fn-coefficients)

        # find BRDF functions with equal a parameters
        equals = [np.where((np.array([VV[1].a for VV in SRFchoices])==tuple(V[1].a)).all(axis=1))[0] for V in SRFchoices]
        # evaluate index of BRDF-functions that have equal a parameter
        equal_a = list({tuple(row) for row in equals})
                           # find phase functions where a-parameter is equal

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0,len(equal_a)):

            SRFdummy = BRDFfunction()
            SRFequal = np.take(SRFchoices,equal_a[i],axis=0)        # select SRF choices where a parameter is equal

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


        for SRF in SRFchoices:
            # set parameters based on chosen classes to define analytic function representation
            SRFcomb._func = SRFcomb._func + SRF[1]._func * SRF[0]
        return SRFcomb

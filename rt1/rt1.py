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
#import time


class RT1(object):
    """
    main class to perform RT simulations
    """

    def __init__(self, I0, t_0, t_ex, p_0, p_ex, RV=None, SRF=None, fn=None, geometry='vvvv'):
        """
        Parameters
        ----------
        I0 : float
            incidence radiation     
        t_0 : float
                 incident zenith-angle
        p_0 : float
               incident azimuth-angle
        t_ex : float
                  exit zenith-angle
                  (if geometry is set to 'mono', theta_ex is automatically set to theta_0 !)
        p_ex : float
                exit azimuth-angle
                (if geometry is set to 'mono', phi_ex is automatically set to phi_0 + np.pi !)
        RV : Volume
            random volume object
        SRF: Surface
            random surface object
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

        self.geometry = geometry
        assert isinstance(geometry, str), 'ERROR: geometry must be a 4-character string'
        assert len(self.geometry) == 4

        self.I0 = I0

        # use only theta_0 and phi_0 if geometry is set to mono
        if self.geometry == 'mono':
            self.t_0 = t_0
            self.t_ex = t_0
            self.p_0 = p_0
            self.p_ex = p_0 + np.pi
        else:
            self.t_0 = t_0
            self.t_ex = t_ex
            self.p_0 = p_0
            self.p_ex = p_ex

        assert RV is not None, 'ERROR: needs to provide volume information'
        self.RV = RV
        # the asserts for omega & tau are performed inside the RT1-class rather than the Volume-class
        # to allow calling Volume-elements without providing omega & tau which is needed to generate
        # linear-combinations of Volume-elements with unambiguous tau- & omega-specifications

        assert self.RV.omega is not None, 'Single scattering albedo needs to be provided'
        assert self.RV.tau is not None, 'Optical depth needs to be provided'

        assert self.RV.omega >= 0.
        assert self.RV.omega <= 1.
        assert self.RV.tau >= 0.

        if self.RV.tau == 0.:
            assert self.RV.omega == 0., 'ERROR: If optical depth is equal to zero, then OMEGA can not be larger than zero'

        assert SRF is not None, 'ERROR: needs to provide surface information'
        self.SRF = SRF


        if fn is None:
        # precalculate the expansiion coefficients for the interaction term
            expr_int = self._calc_interaction_expansion()

            # now we have the integral formula ready. The next step is now to
            # extract the expansion coefficients
            #~ print 'Integral expansion before extraction:'
            self.fn = self._extract_coefficients(expr_int)
        else:
            self.fn = fn

    # calculate cosines of incident- and exit angle
    def _get_mu_0(self):
        return np.cos(self.t_0)
    _mu_0 = property(_get_mu_0)

    def _get_mu_ex(self):
        return np.cos(self.t_ex)
    _mu_ex = property(_get_mu_ex)

    def _extract_coefficients(self, expr):
        """
        extract Fn coefficients from given forumula.

        This is done by collecting the terms of expr with respect to powers of cos(theta_s) and
        simplifying the gained coefficients by applying a simple trigonometric identity.
        """

        theta_s = sp.Symbol('theta_s')
        # collect terms with equal powers of cos(theta_s)
        expr_sort = sp.collect(expr, sp.cos(theta_s), evaluate=False)

        # convert generated dictionary to list of coefficients
        # the use of  .get() is necessary for getting the dict-values since otherwise coefficients that are actually 0.
        # would not appear in the list of fn-coefficients

        # the gained coefficients are further simplified using trigonometric identities to speed up numerical evaluation
        # the TR5 function performs the replacement sin^2(x) = 1-cos(x)^2 to get rid of remaining sin(x)-terms
        # this results in a significant speedup for monostatic evaluations (and a moderate effect on bistatic calculations)
        fn = [sp.expand(TR5(expr_sort.get(sp.cos(theta_s) ** n, 0.), max=self.SRF.ncoefs + self.RV.ncoefs + 1)) for n in range(self.SRF.ncoefs + self.RV.ncoefs + 1)]

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

        volexp = self.RV.legexpansion(self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry).doit()
        brdfexp = self.SRF.legexpansion(self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry).doit()
        #   preparation of the product of p*BRDF for coefficient retrieval
        fPoly = (2 * sp.pi * volexp * brdfexp).expand().doit()  # this is the eq.23. and would need to be integrated from 0 to 2pi

        # do integration of eq. 23
        expr = self._integrate_0_2pi_phis(fPoly)

        # now we do still simplify the expression to be able to express things as power series of cos(theta_s)
        theta_s = sp.Symbol('theta_s')
        replacements = [(sp.sin(theta_s) ** i, ((1. - sp.cos(theta_s) ** 2) ** sp.Rational(i, 2)).expand())  for i in range(1, self.SRF.ncoefs + self.RV.ncoefs + 1) if i % 2 == 0]
        res = expr.xreplace(dict(replacements)).expand()

        return res

    def _cosintegral(self, i):
        """
        integral of cos(x)**i in the interval 0 ... 2*pi
        """
        if i % 2 == 0:
            return (2 ** (-i)) * sp.binomial(i, i * sp.Rational(1, 2))
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
        replacements1 = [(sp.sin(phi_s) ** i, 0.) for i in range(1, self.SRF.ncoefs + self.RV.ncoefs + 1) if i % 2 == 1]

        # then substitute the sine**2 by 1-cos**2
        replacements1 = replacements1 + [(sp.sin(phi_s) ** i, ((1. - sp.cos(phi_s) ** 2) ** sp.Rational(i, 2)).expand()) for i in range(2, self.SRF.ncoefs + self.RV.ncoefs + 1) if i % 2 == 0]
        res = expr.xreplace(dict(replacements1)).expand()

        # replacements need to be done simultaneously, otherwise all remaining sin(phi_s)**even will be replaced by 0

        # integrate the cosine terms
        replacements3 = [(sp.cos(phi_s) ** i, self._cosintegral(i)) for i in range(1, self.SRF.ncoefs + self.RV.ncoefs + 1)]
        res = res.xreplace(dict(replacements3)).expand()
        return res

    def _get_fn(self, fn, n, t_0, p_0, t_ex, p_ex):
        """
        function to evaluate expansion coefficients
        as function of incident geometry
        """
        theta_0 = sp.Symbol('theta_0')
        phi_0 = sp.Symbol('phi_0')
        theta_ex = sp.Symbol('theta_ex')
        phi_ex = sp.Symbol('phi_ex')

        # the destinction between zero and nonzero fn-coefficients is necessary because sympy treats
        # any symbol multiplied by 0 as 0, which results in a function that returns 0 instead of an array of zeroes!
        # -> see  https://github.com/sympy/sympy/issues/3935

        if n >= len(fn):
            return 0.
        else:
            if fn[n] == 0:
                def fnfunc(theta_0, phi_0, theta_ex, phi_ex):
                    return np.ones_like(theta_0) * np.ones_like(phi_0) * np.ones_like(theta_ex) * np.ones_like(phi_ex) * 0.
            else:
                fnfunc = sp.lambdify((theta_0, phi_0, theta_ex, phi_ex), fn[n], modules=["numpy", "sympy"])

            return fnfunc(t_0, p_0, t_ex, p_ex)



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
        return self.I0 * np.exp(-(self.RV.tau / self._mu_0) - (self.RV.tau / self._mu_ex)) * self._mu_0 * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex)

    def volume(self):
        """
        (18)
        """
        return (self.I0 * self.RV.omega * self._mu_0 / (self._mu_0 + self._mu_ex)) * (1. - np.exp(-(self.RV.tau / self._mu_0) - (self.RV.tau / self._mu_ex))) * self.RV.p(self.t_0, self.t_ex, self.p_0, self.p_ex)

    def interaction(self):
        """
        (19)
        """
        Fint1 = self._calc_Fint(self._mu_0, self._mu_ex, self.p_0, self.p_ex)
        Fint2 = self._calc_Fint(self._mu_ex, self._mu_0, self.p_ex, self.p_0)
        return self.I0 * self._mu_0 * self.RV.omega * (np.exp(-self.RV.tau / self._mu_ex) * Fint1 + np.exp(-self.RV.tau / self._mu_0) * Fint2)

    def _calc_Fint(self, mu1, mu2, phi1, phi2):
        """
        (37)
        first order interaction term

        in the original paper there is no dependency on PHI, but here it is
        as the we don not assume per se that PHI1=0 like it is done in the
        mansucript.
        """
        nmax = len(self.fn)
        hlp1 = np.exp(-self.RV.tau / mu1) * np.log(mu1 / (1. - mu1)) - expi(-self.RV.tau) + np.exp(-self.RV.tau / mu1) * expi(self.RV.tau / mu1 - self.RV.tau)
        S2 = np.array([np.sum(mu1 ** (-k) * (expn(k + 1., self.RV.tau) - np.exp(-self.RV.tau / mu1) / k) for k in range(1, (n + 1) + 1)) for n in range(nmax)])
        fn = np.array([self._get_fn(self.fn, n, np.arccos(mu1), phi1, np.arccos(mu2), phi2) for n in range(nmax)])

        mu = np.array([mu1 ** (n + 1) for n in range(nmax)])
        S = np.sum(fn * mu * (S2 + hlp1), axis=0)
        return S

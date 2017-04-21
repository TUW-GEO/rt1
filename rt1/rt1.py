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

import sympy as sp
#import time

try:
    # if symengine is available, use it to perform series-expansions
    # this try-exept is necessary since symengine does currently not
    # build correctly with conda using a python 2.7 environment
    from symengine import expand
except ImportError:
    from sympy import expand


class RT1(object):
    """ Main class to perform RT-simulations


    Parameters
    ----------
    I0 : scalar(float)
         incidence intensity

    t_0 : array_like(float)
          array of incident zenith-angles in radians

    p_0 : array_like(float)
          array of incident azimuth-angles in radians

    t_ex : array_like(float)
           array of exit zenith-angles in radians
           (if geometry is set to 'mono', theta_ex is automatically set to t_0 !)

    p_ex : array_like(float)
           array of exit azimuth-angles in radians
           (if geometry is set to 'mono', phi_ex is automatically set to p_0 + np.pi !)

    RV : rt1.volume
         random object from rt1.volume class

    SRF : surface
          random object from rt1.surface class

    fn : array_like(sympy expression), optional (default = None)
         optional input of pre-calculated array of sympy-expressions to speedup
         calculations where the same fn-coefficients can be used.
         if None, the coefficients will be calculated automatically by calling rt1.fn

    geometry : str
        4 character string specifying which components of the angles should be fixed or variable
        This is done to significantly speed up the evaluation-process of the fn-coefficient generation

        The 4 characters represent in order the properties of: t_0, t_ex, p_0, p_ex

        - 'f' indicates that the angle is treated 'fixed' (i.e. as a numerical constant)
        - 'v' indicates that the angle is treated 'variable' (i.e. as a sympy-variable)
        - Passing  geometry = 'mono'  indicates a monstatic geometry
          (i.e.:  t_ex = t_0, p_ex = p_0 + pi)
          If monostatic geometry is used, the input-values of t_ex and p_ex
          have no effect on the calculations!

        For detailed information on the specification of the geometry-parameter,
        please have a look at the "Evaluation Geometries" section of the documentation
        (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

    """

    def __init__(self, I0, t_0, t_ex, p_0, p_ex, RV=None, SRF=None, fn=None, geometry='vvvv'):
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

        # TODO assert self.RV.omega >= 0.
        # TODO assert self.RV.omega <= 1.
        # TODO assert self.RV.tau >= 0.

        # TODO if self.RV.tau == 0.:
        # TODO     assert self.RV.omega == 0., 'ERROR: If optical depth is equal to zero, then OMEGA can not be larger than zero'

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

        Parameters
        ----------

        expr : sympy expression
               prepared sympy-expression (output of _calc_interaction_expansion())
               to be used for extracting the fn-coefficients

        Returns
        --------
        fn : list(sympy expressions)
             A list of sympy expressions that represent the fn-coefficients associated
             with the given input-equation (expr).

        """

        theta_s = sp.Symbol('theta_s')
        # collect terms with equal powers of cos(theta_s)
        expr_sort = sp.collect(expr, sp.cos(theta_s), evaluate=False)

        # convert generated dictionary to list of coefficients
        # the use of  .get() is necessary for getting the dict-values since otherwise coefficients that are actually 0.
        # would not appear in the list of fn-coefficients

        fn = [expr_sort.get(sp.cos(theta_s) ** n, 0.) for n in range(self.SRF.ncoefs + self.RV.ncoefs + 1)]

        return fn

    def _calc_interaction_expansion(self):
        """
        Evaluation of the polar-integral from the definition of the fn-coefficients.
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-fn_coef_definition)

        The approach is as follows:

        1. Expand the Legrende coefficents of the surface and volume phase functions
        2. Apply the function _integrate_0_2pi_phis() to evaluate the integral
        3. Replace remaining sin(theta_s) terms in the Legrende polynomials by cos(theta_s) to prepare for fn-coefficient extraction
        4. Expand again to ensure that a fully expanded expression is returned (to be used as input in _extract_coefficients() )

        Returns
        --------
        res : sympy expression
              A fully expanded expression that can be used as input for _extract_coefficients()

        """
        # preevaluate expansions for volume and surface phase functions
        # this returns symbolic code to be then further used

        volexp = self.RV.legexpansion(self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry).doit()
        brdfexp = self.SRF.legexpansion(self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry).doit()
        #   preparation of the product of p*BRDF for coefficient retrieval
        fPoly = expand(2 * sp.pi * volexp * brdfexp)  # this is the eq.23. and would need to be integrated from 0 to 2pi

        # do integration of eq. 23
        expr = self._integrate_0_2pi_phis(fPoly)

        # now we do still simplify the expression to be able to express things as power series of cos(theta_s)
        theta_s = sp.Symbol('theta_s')
        replacements = [(sp.sin(theta_s) ** i, expand((1. - sp.cos(theta_s) ** 2) ** sp.Rational(i, 2))) for i in range(1, self.SRF.ncoefs + self.RV.ncoefs + 1) if i % 2 == 0]
        res = expand(expr.xreplace(dict(replacements)))

        return res

    def _cosintegral(self, i):
        """
        Analytical solution to the integral of cos(x)**i in the interval 0 ... 2*pi

        Parameters
        ----------
        i : scalar(int)
            Power of the cosine function to be integrated, i.e.  cos(x)^i

        Returns
        -------
        float
              Numerical value of the integral of cos(x)^i in the interval 0 ... 2*pi
        """
        if i % 2 == 0:
            return (2 ** (-i)) * sp.binomial(i, i * sp.Rational(1, 2))
        else:
            # for odd exponents result is always zero
            return 0.

    def _integrate_0_2pi_phis(self, expr):
        """
        Perforn symbolic integration of a pre-expanded power-series in sin(phi_s) and cos(phi_s)
        over the variable phi_s in the interval 0 ... 2*pi

        The approach is as follows:

        1. Replace all appearing sin(phi_s)^odd with 0 since the integral vanishes
        2. Replace all remaining sin(phi_s)^even with their representation in terms of cos(phi_s)
        3. Replace all cos(phi_s)^i terms with _cosintegral(i)
        4. Expand the gained solution for further processing

        Parameters
        ----------
        expr : sympy expression
               pre-expanded product of the legendre-expansions of RV.legexpansion() and SRF.legexpansion()

        Returns
        -------
        res : sympy expression
              resulting symbolic expression that results from integrating expr over the variable phi_s in the interval 0 ... 2*pi
        """
        phi_s = sp.Symbol('phi_s')

        # replace first all odd powers of sin(phi_s) as these are all zero for the integral
        replacements1 = [(sp.sin(phi_s) ** i, 0.) for i in range(1, self.SRF.ncoefs + self.RV.ncoefs + 1) if i % 2 == 1]

        # then substitute the sine**2 by 1-cos**2
        replacements1 = replacements1 + [(sp.sin(phi_s) ** i, expand((1. - sp.cos(phi_s) ** 2) ** sp.Rational(i, 2))) for i in range(2, self.SRF.ncoefs + self.RV.ncoefs + 1) if i % 2 == 0]
        res = expand(expr.xreplace(dict(replacements1)))

        # replacements need to be done simultaneously, otherwise all remaining sin(phi_s)**even will be replaced by 0

        # integrate the cosine terms
        replacements3 = [(sp.cos(phi_s) ** i, self._cosintegral(i)) for i in range(1, self.SRF.ncoefs + self.RV.ncoefs + 1)]
        res = expand(res.xreplace(dict(replacements3)))
        return res

    def _get_fn(self, n, t_0, p_0, t_ex, p_ex):
        """
        Function to numerically evaluate the fn-coefficients as function of incident geometry.

        Parameters
        ----------
        n : scalar(int)
            Number of fn-coefficient to be evaluated (starting with 0)

        t_0 : array_like(float)
              array of incident zenith-angles in radians

        p_0 : array_like(float)
              array of incident azimuth-angles in radians

        t_ex : array_like(float)
               array of exit zenith-angles in radians

        p_ex : array_like(float)
               array of exit azimuth-angles in radians

        Returns
        -------
        array_like(float)
              Numerical value of the n^th fn-coefficient evaluated at the given angles.
        """

        theta_0 = sp.Symbol('theta_0')
        phi_0 = sp.Symbol('phi_0')
        theta_ex = sp.Symbol('theta_ex')
        phi_ex = sp.Symbol('phi_ex')

        # the destinction between zero and nonzero fn-coefficients is necessary because sympy treats
        # any symbol multiplied by 0 as 0, which results in a function that returns 0 instead of an array of zeroes!
        # -> see  https://github.com/sympy/sympy/issues/3935

        if n >= len(self.fn):
            return 0.
        else:
            if (self.fn[n] == 0. or self.fn[n] == 0):
                def fnfunc(theta_0, phi_0, theta_ex, phi_ex):
                    return np.ones_like(theta_0) * np.ones_like(phi_0) * np.ones_like(theta_ex) * np.ones_like(phi_ex) * 0.
            else:
                fnfunc = sp.lambdify((theta_0, phi_0, theta_ex, phi_ex), self.fn[n], modules=["numpy", "sympy"])

            return fnfunc(t_0, p_0, t_ex, p_ex)

    def calc(self):
        """
        Perform actual calculation of bistatic scattering at top of the random volume (z=0)
        for the specified geometry. For details please have a look at the documentation
        (http://rt1.readthedocs.io/en/latest/theory.html#first-order-solution-to-the-rte)


        Returns
        -------
        Itot : array_like(float)
               Total scattered intensity (Itot = Isurf + Ivol + Iint)

        Isurf : array_like(float)
                Surface contribution

        Ivol : array_like(float)
               Volume contribution

        Iint : array_like(float)
               Interaction contribution
        """
        # (16)

        # the following if-else query ensures that volume- and interaction-terms
        # are only calculated if tau > 0. (to avoid nan-values from invalid function-evaluations)

        if np.isscalar(self.RV.tau):
            Isurf = self.surface()
            if self.RV.tau > 0.:  # explicit differentiation for non-existing canopy, as otherwise NAN values
                Ivol = self.volume()
                Iint = self.interaction()
            else:
                Ivol = 0.
                Iint = 0.
        else:
            # calculate surface-term (valid for any tau-value)
            Isurf = self.surface()

            # store initial parameter-values
            old_tau = self.RV.tau
            old_omega = self.RV.omega
            old_NN = self.SRF.NormBRDF

            # set mask for tau > 0.
            mask = old_tau > 0.
            valid_index = np.where(mask)
            invalid_index = np.where(~mask)

            # set parameter-values to valid values for calculation
            self.RV.tau = old_tau[mask]
            self.RV.omega = old_omega[mask]
            self.SRF.NormBRDF = old_NN[mask]

            # calculate volume and surface term where tau-values are valid
            _Ivol = self.volume()
            _Iint = self.interaction()

            # reset parameter values to old values
            self.RV.tau = old_tau
            self.RV.omega = old_omega
            self.SRF.NormBRDF = old_NN

            # combine calculated volume-contributions for valid tau-values
            # with zero-arrays for invalid tau-values
            Ivol = np.ones_like(self.t_0)
            Ivol[valid_index] = _Ivol
            Ivol[invalid_index] = np.ones_like(Ivol[~mask]) * 0.

            # combine calculated interaction-contributions for valid tau-values
            # with zero-arrays for invalid tau-values
            Iint = np.ones_like(self.t_0)
            Iint[valid_index] = _Iint
            Iint[invalid_index] = np.ones_like(Iint[~mask]) * 0.

        return Isurf + Ivol + Iint, Isurf, Ivol, Iint




    def surface(self):
        """
        Numerical evaluation of the surface-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        --------
        array_like(float)
                          Numerical value of the surface-contribution for the given set of parameters
        """
        return self.I0 * np.exp(-(self.RV.tau / self._mu_0) - (self.RV.tau / self._mu_ex)) * self._mu_0 * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex)

    def volume(self):
        """
        Numerical evaluation of the volume-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#volume_contribution)

        Returns
        --------
        array_like(float)
                          Numerical value of the volume-contribution for the given set of parameters
        """
        return (self.I0 * self.RV.omega * self._mu_0 / (self._mu_0 + self._mu_ex)) * (1. - np.exp(-(self.RV.tau / self._mu_0) - (self.RV.tau / self._mu_ex))) * self.RV.p(self.t_0, self.t_ex, self.p_0, self.p_ex)

    def interaction(self):
        """
        Numerical evaluation of the volume-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#interaction_contribution)

        Returns
        --------
        array_like(float)
                          Numerical value of the interaction-contribution for the given set of parameters
        """
        Fint1 = self._calc_Fint(self._mu_0, self._mu_ex, self.p_0, self.p_ex)
        Fint2 = self._calc_Fint(self._mu_ex, self._mu_0, self.p_ex, self.p_0)
        return self.I0 * self._mu_0 * self.RV.omega * (np.exp(-self.RV.tau / self._mu_ex) * Fint1 + np.exp(-self.RV.tau / self._mu_0) * Fint2)

    def _calc_Fint(self, mu1, mu2, phi1, phi2):
        """
        Numerical evaluation of the F_int() function used in the definition of the interaction-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#F_int)

        Parameters
        -----------
        mu1 : array_like(float)
              cosine of the first polar-angle argument

        mu2 : array_like(float)
              cosine of the second polar-angle argument

        phi1 : array_like(float)
               first azimuth-angle argument in radians

        phi2 : array_like(float)
               second azimuth-angle argument in radians

        Returns
        --------
        S : array_like(float)
            Numerical value of F_int for the given set of parameters
        """
        nmax = len(self.fn)
        hlp1 = np.exp(-self.RV.tau / mu1) * np.log(mu1 / (1. - mu1)) - expi(-self.RV.tau) + np.exp(-self.RV.tau / mu1) * expi(self.RV.tau / mu1 - self.RV.tau)
        S2 = np.array([np.sum(mu1 ** (-k) * (expn(k + 1., self.RV.tau) - np.exp(-self.RV.tau / mu1) / k) for k in range(1, (n + 1) + 1)) for n in range(nmax)])
        fn = np.array([self._get_fn(n, np.arccos(mu1), phi1, np.arccos(mu2), phi2) for n in range(nmax)])

        mu = np.array([mu1 ** (n + 1) for n in range(nmax)])
        S = np.sum(fn * mu * (S2 + hlp1), axis=0)
        return S

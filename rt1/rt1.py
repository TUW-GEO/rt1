"""
Core module for implementation of 1st order scattering model
using arbitrary BRDF and phase functions

References
----------
Quast & Wagner (2016): doi:10.1364/AO.55.005379
"""

from . import log

import numpy as np

from functools import lru_cache

from scipy.special import expi
from scipy.special import expn

import sympy as sp

try:
    # if symengine is available, use it to perform series-expansions
    from symengine import expand, Lambdify

    _init_lambda_backend = "symengine"
except ImportError:
    from sympy import expand

    _init_lambda_backend = "sympy"


class RT1(object):
    """
    Main class to perform RT-simulations

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
           (if geometry is 'mono', theta_ex is automatically set to t_0)

    p_ex : array_like(float)
           array of exit azimuth-angles in radians
           (if geometry is 'mono', phi_ex is automatically set to p_0 + np.pi)

    V : rt1.volume
        random object from rt1.volume class

    SRF : surface
          random object from rt1.surface class

    fn_input : array_like(sympy expression), optional (default = None)
         optional input of pre-calculated array of sympy-expressions
         to speedup calculations where the same fn-coefficients can be used.
         if None, the coefficients will be calculated automatically at the
         initialization of the RT1-object

    _fnevals_input : callable, optional (default = None)
               optional input of pre-compiled function to numerically evaluate
               the fn_coefficients. if None, the function will be compiled
               using the fn-coefficients provided.
               Note that once the _fnevals function is provided, the
               fn-coefficients are no longer needed and have no effect on the
               calculated results!

    geometry : str (default = 'vvvv')
        4 character string specifying which components of the angles should
        be fixed or variable. This is done to significantly speed up the
        evaluation-process of the fn-coefficient generation

        The 4 characters represent in order the properties of:
            t_0, t_ex, p_0, p_ex

        - 'f' indicates that the angle is treated 'fixed'
          (i.e. as a numerical constant)
        - 'v' indicates that the angle is treated 'variable'
          (i.e. as a sympy-variable)
        - Passing  geometry = 'mono'  indicates a monstatic geometry
          (i.e.:  t_ex = t_0, p_ex = p_0 + pi)
          If monostatic geometry is used, the input-values of t_ex and p_ex
          have no effect on the calculations!

        For detailed information on the specification of the
        geometry-parameter, please have a look at the "Evaluation Geometries"
        section of the documentation:
        (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

    bsf : float (default = 0.)
          fraction of bare-soil contribution (no attenuation due to vegetation)

    param_dict : dict (default = {})
                 a dictionary to assign numerical values to sympy.Symbols
                 appearing in the definitions of V and SRF.
    lambda_backend : str (default = 'symengine' if possible, else 'sympy')
                     indicator to select the module that shall be used
                     to compile a function for numerical evaluation of the
                     fn-coefficients.

                     possible values are:
                         - 'sympy' :  sympy.lambdify is used to compile
                           the _fnevals function
                         - 'symengine' : symengine.LambdifyCSE is used to
                           compile the _fnevals function. This results in
                           considerable speedup for long fn-coefficients
    int_Q : bool (default = True)
            indicator whether the interaction-term should be calculated or not
    verbosity : int
            select the verbosity level of the module to get status-reports
                - 0 : print nothing
                - 1 : print some infos during runtime
                - 2 : print more
                - >=3 : print all
    """

    def __init__(
        self,
        I0,
        t_0,
        t_ex,
        p_0,
        p_ex,
        V=None,
        SRF=None,
        fn_input=None,
        _fnevals_input=None,
        geometry="mono",
        bsf=0.0,
        param_dict={},
        lambda_backend=_init_lambda_backend,
        int_Q=True,
        verbosity=1,
    ):

        assert isinstance(geometry, str), (
            "ERROR: geometry must be " + "a 4-character string"
        )
        assert len(geometry) == 4, "ERROR: geometry must be " + "a 4-character string"
        self.geometry = geometry

        self.I0 = I0
        self.param_dict = param_dict
        self.lambda_backend = lambda_backend
        self.int_Q = int_Q

        assert V is not None, "ERROR: needs to provide volume information"
        self.V = V

        assert SRF is not None, "ERROR: needs to provide surface information"
        self.SRF = SRF

        self.fn_input = fn_input
        self._fnevals_input = _fnevals_input

        self._set_t_0(t_0)
        self._set_p_0(p_0)
        if self.geometry != "mono":
            self._set_t_ex(t_ex)
            self._set_p_ex(p_ex)

        self.verbosity = verbosity
        self.bsf = bsf

        # self._set_fn(fn)
        # self._set_fnevals(_fnevals)

        # the asserts for omega & tau are performed inside the RT1-class
        # rather than the Volume-class to allow calling Volume-elements without
        # providing omega & tau which is needed to generate linear-combinations
        # of Volume-elements with unambiguous tau- & omega-specifications

        assert self.V.omega is not None, (
            "Single scattering albedo " + "needs to be provided"
        )
        assert self.V.tau is not None, "Optical depth needs to be provided"

        # self.V.omega[0] must be used instead of self.V.omega since the
        # setter functions for omega, tau and NormBRDF add an additional
        # axis to the given input. Checking for sp.Basic is sufficient
        # to distinguish if the input was given as a sympy equation. For
        # details see: http://docs.sympy.org/latest/guide.html#basics
        if not isinstance(self.V.omega, sp.Basic):
            assert np.any(self.V.omega >= 0.0), (
                "Single scattering albedo " + "must be greater than 0"
            )
        if not isinstance(self.V.tau, sp.Basic):
            assert np.any(self.V.tau >= 0.0), (
                "Optical depth " + "must be greater than 0"
            )
        if not isinstance(self.SRF.NormBRDF, sp.Basic):
            assert np.any(self.SRF.NormBRDF >= 0.0), (
                "NormBRDF " + "must be greater than 0"
            )

    def __getstate__(self):
        # this is required since functions created by
        # symengine are currently not pickleable!
        log.info("dropping fn-coefficients to allow pickling...")
        for delkey in ["_RT1__fn"]:
            if delkey in self.__dict__:
                log.info("removing", delkey, "from __dict__")
                del self.__dict__[delkey]
        for Nonekey in ["_fn_input"]:
            if Nonekey in self.__dict__:
                log.info("setting", Nonekey, "to None")
                self.__dict__[Nonekey] = None

        if self.lambda_backend == "symengine":
            log.info(
                "the resulting dump of the _fnevals functions "
                + "generated by symengine will be platform-dependent!"
            )

        return self.__dict__

    @property
    def _cached_props(self):
        """a list of the names of the properties that are cached"""

        names = [
            "_d_surface_dummy_lambda",
            "_d_surface_dummy_lambda",
            "_mu_0_x",
            "_mu_ex_x",
        ]
        return names

    def _clear_cache(self):
        self._d_surface_dummy_lambda.cache_clear()
        self._d_surface_dummy_lambda.cache_clear()
        RT1._mu_0_x.fget.cache_clear()
        RT1._mu_ex_x.fget.cache_clear()

    def _cache_info(self):
        text = []
        for name in self._cached_props:
            try:
                cinfo = getattr(self, name).cache_info()
                text += [f"{name:<18}:   " + f"{cinfo}"]
            except Exception:
                text += [f"{name:<18}:   " + "???"]

        log.info("\n".join(text))

    def prv(self, v, msg):
        """
        function to set print output based on verbosity level v.
        possible values for v:

            - 0 : print nothing
            - 1 : print some infos during runtime
            - 2 : print more
            - >=3 : print all

        Parameters
        ----------
        v : int
            the verbosity.
        msg : str
            the message to be printed.
        """
        if self.verbosity >= v:
            log.info(msg)

    def _get_fn(self):
        try:
            return self.__fn
        except AttributeError:
            self._set_fn(self.fn_input)
            return self.__fn

    def _set_fn(self, fn):
        # set the fn-coefficients and generate lambdified versions
        # of the fn-coefficients for evaluation
        # only evaluate fn-coefficients if _fnevals funcions are not
        # already available!
        if fn is None and self.int_Q is True:
            self.prv(1, "evaluating fn-coefficients...")

            import timeit

            tic = timeit.default_timer()
            # precalculate the expansiion coefficients for the interaction term
            expr_int = self._calc_interaction_expansion()
            toc = timeit.default_timer()
            self.prv(2, "expansion calculated, it took " + str(toc - tic) + " sec")

            # extract the expansion coefficients
            tic = timeit.default_timer()
            self.__fn = self._extract_coefficients(expr_int)
            toc = timeit.default_timer()
            self.prv(2, "coefficients extracted, it took " + str(toc - tic) + " sec")
        else:
            self.prv(3, "using provided fn-coefficients")
            self.__fn = fn

    fn = property(_get_fn, _set_fn)

    def _get_fnevals(self):
        try:
            return self.__fnevals
        except AttributeError:
            self._set_fnevals(self._fnevals_input)
            return self.__fnevals

        return self.__fnevals

    def _set_fnevals(self, _fnevals):
        if _fnevals is None and self.int_Q is True:
            self.prv(1, "generation of _fnevals functions...")
            import timeit

            tic = timeit.default_timer()

            # define new lambda-functions for each fn-coefficient
            variables = sp.var(
                ("theta_0", "phi_0", "theta_ex", "phi_ex")
                + tuple(map(str, self.param_dict.keys()))
            )

            # use symengine's Lambdify if symengine has been used within
            # the fn-coefficient generation
            if self.lambda_backend == "symengine":
                self.prv(1, "symengine")
                # using symengines own "common subexpression elimination"
                # routine to perform lambdification

                # llvm backend is used to allow pickling of the functions
                # see https://github.com/symengine/symengine.py/issues/294
                self.__fnevals = Lambdify(
                    list(variables),
                    self.fn,
                    order="F",
                    cse=True,
                    backend="llvm",
                )
            elif self.lambda_backend == "sympy":
                # using sympy's lambdify without "common subexpression
                # elimination" to perform lambdification

                self.prv(1, "sympy")

                sympy_fn = list(map(sp.sympify, self.fn))

                self.__fnevals = sp.lambdify(
                    (variables),
                    sp.sympify(sympy_fn),
                    modules=["numpy", "sympy"],
                    dummify=False,
                )

                self.__fnevals.__doc__ = """
                                    A function to numerically evaluate the
                                    fn-coefficients a for given set of
                                    incidence angles and parameter-values
                                    as defined in the param_dict dict.

                                    The call-signature is:
                                        RT1-object._fnevals(theta_0, phi_0, \
                                        theta_ex, phi_ex, *param_dict.values())
                                    """

            else:
                self.prv(
                    1,
                    'lambda_backend "' + self.lambda_backend + '" is not available',
                )

            toc = timeit.default_timer()
            self.prv(
                2,
                "lambdification finished, it took " + str(toc - tic) + " sec",
            )

        else:
            self.prv(3, "using provided _fnevals-functions")
            self.__fnevals = _fnevals

    _fnevals = property(_get_fnevals, _set_fnevals)

    def _get_t_0(self):
        return self.__t_0

    def _set_t_0(self, t_0):
        # if t_0 is given as scalar input, convert it to 1d numpy array
        if np.isscalar(t_0):
            t_0 = np.array([t_0])
        self.__t_0 = t_0

    t_0 = property(_get_t_0, _set_t_0)

    def _get_p_0(self):
        return self.__p_0

    def _set_p_0(self, p_0):
        # if p_o is given as scalar input, convert it to 1d numpy array
        if np.isscalar(p_0):
            p_0 = np.array([p_0])
        self.__p_0 = p_0

    p_0 = property(_get_p_0, _set_p_0)

    def _get_t_ex(self):
        if self.geometry == "mono":
            return self._get_t_0()
        else:
            return self.__t_ex

    def _set_t_ex(self, t_ex):
        # if geometry is mono, set t_ex to t_0
        if self.geometry == "mono":
            log.info('t_ex is always equal to t_0 if geometry is "mono"')
            pass
        else:
            # if t_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(t_ex):
                t_ex = np.array([t_ex])
            self.__t_ex = t_ex

    t_ex = property(_get_t_ex, _set_t_ex)

    def _get_p_ex(self):
        if self.geometry == "mono":
            return self._get_p_0() + np.pi
        else:
            return self.__p_ex

    def _set_p_ex(self, p_ex):
        # if geometry is mono, set p_ex to p_0
        if self.geometry == "mono":
            log.info('p_ex is equal to (p_0 + PI) if geometry is "mono"!')
            pass
        else:
            # if p_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(p_ex):
                p_ex = np.array([p_ex])
            self.__p_ex = p_ex

    p_ex = property(_get_p_ex, _set_p_ex)

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

        This is done by collecting the terms of expr with respect to powers
        of cos(theta_s) and simplifying the gained coefficients by applying
        a simple trigonometric identity.

        Parameters
        ----------
        expr : sympy expression
               prepared sympy-expression to be used for extracting
               the fn-coefficients (output of _calc_interaction_expansion())

        Returns
        -------
        fn : list(sympy expressions)
             A list of sympy expressions that represent the fn-coefficients
             associated with the given input-equation (expr).

        """

        theta_s = sp.Symbol("theta_s")

        N_fn = self.SRF.ncoefs + self.V.ncoefs - 1

        fn = []

        # find f_0 coefficient
        repl0 = dict([[sp.cos(theta_s), 0]])
        fn = fn + [expr.xreplace(repl0)]

        # find f_1 coefficient
        repl1 = dict(
            [[sp.cos(theta_s) ** i, 0] for i in list(range(N_fn, 0, -1)) if i != 1]
            + [[sp.cos(theta_s), 1]]
        )
        fn = fn + [expr.xreplace(repl1) - fn[0]]

        for n in np.arange(2, N_fn, dtype=int):
            repln = dict([[sp.cos(theta_s) ** int(n), 1]])
            fn = fn + [(expr.xreplace(repln)).xreplace(repl0) - fn[0]]

        #        # alternative way of extracting the coefficients:
        #        theta_s = sp.Symbol('theta_s')
        #        # collect terms with equal powers of cos(theta_s)
        #        expr_sort = sp.collect(expr, sp.cos(theta_s), evaluate=False)
        #
        #        # convert generated dictionary to list of coefficients
        #        # the use of  .get() is necessary for getting the dict-values since
        #        # otherwise coefficients that are actually 0. would not appear
        #        #  in the list of fn-coefficients
        #
        #        fn = [expr_sort.get(sp.cos(theta_s) ** n, 0.)
        #              for n in range(self.SRF.ncoefs + self.V.ncoefs - 1)]

        return fn

    def _calc_interaction_expansion(self):
        """
        Evaluation of the polar-integral from the definition of
        the fn-coefficients.
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-fn_coef_definition)

        The approach is as follows:

            1. Expand the Legrende coefficents of the surface and volume
               phase functions
            2. Apply the function _integrate_0_2pi_phis() to evaluate
               the integral
            3. Replace remaining sin(theta_s) terms in the Legrende polynomials
               by cos(theta_s) to prepare for fn-coefficient extraction
            4. Expand again to ensure that a fully expanded expression
               is returned (to be used as input in _extract_coefficients() )

        Returns
        -------
        res : sympy expression
              A fully expanded expression that can be used as
              input for _extract_coefficients()

        """
        # preevaluate expansions for volume and surface phase functions
        # this returns symbolic code to be then further used

        volexp = self.V.legexpansion(
            self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry
        ).doit()

        brdfexp = self.SRF.legexpansion(
            self.t_0, self.t_ex, self.p_0, self.p_ex, self.geometry
        ).doit()

        # preparation of the product of p*BRDF for coefficient retrieval
        # this is the eq.23. and would need to be integrated from 0 to 2pi
        fPoly = expand(2 * sp.pi * volexp * brdfexp)

        # do integration of eq. 23
        expr = self._integrate_0_2pi_phis(fPoly)

        # now we do still simplify the expression to be able to express
        # things as power series of cos(theta_s)
        theta_s = sp.Symbol("theta_s")
        replacements = [
            (
                sp.sin(theta_s) ** i,
                expand((1.0 - sp.cos(theta_s) ** 2) ** sp.Rational(i, 2)),
            )
            for i in range(1, self.SRF.ncoefs + self.V.ncoefs - 1)
            if i % 2 == 0
        ]

        res = expand(expr.xreplace(dict(replacements)))

        return res

    def _cosintegral(self, i):
        """
        Analytical solution to the integral of cos(x)**i
        in the inteVal 0 ... 2*pi

        Parameters
        ----------
        i : scalar(int)
            Power of the cosine function to be integrated, i.e.  cos(x)^i

        Returns
        -------
        - : float
              Numerical value of the integral of cos(x)^i
              in the inteVal 0 ... 2*pi
        """

        if i % 2 == 0:
            return (2 ** (-i)) * sp.binomial(i, i * sp.Rational(1, 2))
        else:
            # for odd exponents result is always zero
            return 0.0

    def _integrate_0_2pi_phis(self, expr):
        """
        Perforn symbolic integration of a pre-expanded power-series
        in sin(phi_s) and cos(phi_s) over the variable phi_s
        in the inteVal 0 ... 2*pi

        The approach is as follows:

            1. Replace all appearing sin(phi_s)^odd with 0 since the
               integral vanishes
            2. Replace all remaining sin(phi_s)^even with their representation
               in terms of cos(phi_s)
            3. Replace all cos(phi_s)^i terms with _cosintegral(i)
            4. Expand the gained solution for further processing

        Parameters
        ----------
        expr : sympy expression
               pre-expanded product of the legendre-expansions of
               V.legexpansion() and SRF.legexpansion()

        Returns
        -------
        res : sympy expression
              resulting symbolic expression that results from integrating
              expr over the variable phi_s in the inteVal 0 ... 2*pi
        """

        phi_s = sp.Symbol("phi_s")

        # replace first all odd powers of sin(phi_s) as these are
        # all zero for the integral
        replacements1 = [
            (sp.sin(phi_s) ** i, 0.0)
            for i in range(1, self.SRF.ncoefs + self.V.ncoefs + 1)
            if i % 2 == 1
        ]

        # then substitute the sine**2 by 1-cos**2
        replacements1 = replacements1 + [
            (
                sp.sin(phi_s) ** i,
                expand((1.0 - sp.cos(phi_s) ** 2) ** sp.Rational(i, 2)),
            )
            for i in range(2, self.SRF.ncoefs + self.V.ncoefs + 1)
            if i % 2 == 0
        ]

        res = expand(expr.xreplace(dict(replacements1)))

        # replacements need to be done simultaneously, otherwise all
        # remaining sin(phi_s)**even will be replaced by 0

        # integrate the cosine terms
        replacements3 = [
            (sp.cos(phi_s) ** i, self._cosintegral(i))
            for i in range(1, self.SRF.ncoefs + self.V.ncoefs + 1)
        ]

        res = expand(res.xreplace(dict(replacements3)))
        return res

    def calc(self):
        """
        Perform actual calculation of bistatic scattering at top of the
        random volume (z=0) for the specified geometry. For details please
        have a look at the documentation:
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

        if isinstance(self.V.tau, (int, float)):
            Isurf = self.surface()
            # differentiation for non-existing canopy, as otherwise NAN values
            if self.V.tau > 0.0:
                Ivol = self.volume()
                if self.int_Q is True:
                    Iint = self.interaction()
                else:
                    Iint = np.array([0.0])
            else:
                Ivol = np.array([0.0])
                Iint = np.array([0.0])
        else:
            Isurf = self.surface()
            Ivol = self.volume()
            # TODO this should be fixed more properly
            # (i.e. for tau=0, no interaction-term should be calculated)
            if self.int_Q is True:
                Iint = self.interaction()
                # check if there are nan-values present that result from
                # (self.V.tau = 0) and replace them with 0
                wherenan = np.isnan(Iint)
                if np.any(wherenan) and np.allclose(
                    *np.broadcast_arrays(wherenan, self.V.tau == 0.0)
                ):
                    self.prv(
                        3,
                        "Warning replacing nan-values caused by tau=0 \
                             in the interaction-term with 0!",
                    )
                    Iint[np.where(wherenan)] = 0.0
                else:
                    pass

        if self.int_Q is True:
            return Isurf + Ivol + Iint, Isurf, Ivol, Iint
        else:
            return Isurf + Ivol, Isurf, Ivol

    def surface(self):
        """
        Numerical evaluation of the surface-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        -------
        - : array_like(float)
            Numerical value of the surface-contribution for the
            given set of parameters
        """
        # bare soil contribution
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex))) * I_bs

        return self.SRF.NormBRDF * ((1.0 - self.bsf) * Isurf + self.bsf * I_bs)

    def surface_slope(self, dB=False, sig0=False):
        """
        Numerical evaluation of the slope (dI_s/dt_0) of the
        (!monostatic!) surface-contribution

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic slope of the
            surface-contribution
        """
        # evaluate the slope of the used brdf
        brdf_slope = self.SRF.brdf_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )
        # evaluate the used brdf
        brdf_val = self.SRF.brdf(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        # vegetated soil contribution
        I_vegs_slope = (
            self.I0
            * np.exp(-(2 * self.V.tau / self._mu_0))
            * (
                self._mu_0 * brdf_slope
                - (2 * self.V.tau / self._mu_0 + 1) * np.sin(self.t_0) * brdf_val
            )
        )

        # bare soil contribution
        I_bs_slope = self.I0 * (self._mu_0 * brdf_slope - np.sin(self.t_0) * brdf_val)

        I_slope = self.SRF.NormBRDF * (
            (1.0 - self.bsf) * I_vegs_slope + self.bsf * I_bs_slope
        )

        if sig0 is False and dB is False:
            return I_slope
        else:
            I_val = self.surface()
            if sig0 is True and dB is False:
                return 4.0 * np.pi * (self._mu_0 * I_slope - np.sin(self.t_0) * I_val)
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * I_slope / I_val
            elif sig0 is True and dB is True:
                return 10.0 / np.log(10) * (I_slope / I_val - np.tan(self.t_0))

    def surface_curv(self, dB=False, sig0=False):
        """
        Numerical evaluation of the curvature (d^2I_s/dt_0^2)
        of the (!monostatic!) surface-contribution

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic curvature of the
            surface-contribution
        """

        # evaluate the slope of the used brdf
        brdf_curv = self.SRF.brdf_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=2,
        )
        # evaluate the slope of the used brdf
        brdf_slope = self.SRF.brdf_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )
        # evaluate the used brdf
        brdf_val = self.SRF.brdf(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        # vegetated soil contribution
        I_vegs_curv = (
            self.I0
            * np.exp(-(2.0 * self.V.tau / self._mu_0))
            * (
                self._mu_0 * brdf_curv
                - 2.0
                * np.sin(self.t_0)
                * brdf_slope
                * (2.0 * self.V.tau / self._mu_0 + 1.0)
                + (
                    4.0 * self.V.tau ** 2 / self._mu_0 ** 3 * np.sin(self.t_0) ** 2
                    - 2.0 * self.V.tau
                    - self._mu_0
                )
                * brdf_val
            )
        )

        # bare soil contribution
        I_bs_curv = self.I0 * (
            self._mu_0 * brdf_curv
            - 2.0 * np.sin(self.t_0) * brdf_slope
            - self._mu_0 * brdf_val
        )

        I_curv = self.SRF.NormBRDF * (
            (1.0 - self.bsf) * I_vegs_curv + self.bsf * I_bs_curv
        )

        if sig0 is False and dB is False:
            return I_curv
        else:
            I_slope = self.surface_slope(dB=False, sig0=False)
            I_val = self.surface()
            if sig0 is True and dB is False:
                return (
                    4.0
                    * np.pi
                    * (
                        self._mu_0 * I_curv
                        - 2.0 * np.sin(self.t_0) * I_slope
                        - self._mu_0 * I_val
                    )
                )
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * (I_curv / I_val - I_slope ** 2 / I_val ** 2)
            elif sig0 is True and dB is True:
                return (
                    10.0
                    / np.log(10)
                    * (I_curv / I_val - I_slope ** 2 / I_val ** 2 - self._mu_0 ** (-2))
                )

    def volume(self):
        """
        Numerical evaluation of the volume-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#volume_contribution)

        Returns
        -------
        - : array_like(float)
            Numerical value of the volume-contribution for the
            given set of parameters
        """
        vol = (
            (self.I0 * self.V.omega * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex)))
            * self.V.p(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        return (1.0 - self.bsf) * vol

    def volume_slope(self, dB=False, sig0=False):
        """
        Numerical evaluation of the slope (dI_v/dt_0) of the
        (!monostatic!) volume-contribution

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic slope of the
            volume-contribution
        """

        # evaluate the slope of the used phase-function
        p_slope = self.V.p_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )

        # evaluate the used phase function
        p_val = self.V.p(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        # volume contribution
        I_slope = (
            (1.0 - self.bsf)
            * self.I0
            * self.V.omega
            / 2.0
            * (
                (
                    np.exp(-(2 * self.V.tau / self._mu_0))
                    * 2
                    * self.V.tau
                    * np.sin(self.t_0)
                    / self._mu_0 ** 2
                )
                * p_val
                + (1.0 - np.exp(-(2 * self.V.tau / self._mu_0))) * p_slope
            )
        )

        if sig0 is False and dB is False:
            return I_slope
        else:
            I_val = self.volume()
            if sig0 is True and dB is False:
                return 4.0 * np.pi * (self._mu_0 * I_slope - np.sin(self.t_0) * I_val)
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * I_slope / I_val
            elif sig0 is True and dB is True:
                return 10.0 / np.log(10) * (I_slope / I_val - np.tan(self.t_0))

    def volume_curv(self, dB=False, sig0=False):
        """
        Numerical evaluation of the curvature (d^2I_s/dt_0^2)
        of the (!monostatic!) volume-contribution

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic curvature of the
            volume-contribution
        """
        # evaluate the slope of the used brdf
        p_curv = self.V.p_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=2,
        )
        # evaluate the slope of the used brdf
        p_slope = self.V.p_theta_diff(
            t_0=self.t_0,
            t_ex=self.t_ex,
            p_0=self.p_0,
            p_ex=self.p_ex,
            geometry="mono",
            param_dict=self.param_dict,
            return_symbolic=False,
            n=1,
        )
        # evaluate the used brdf
        p_val = self.V.p(
            self.t_0,
            self.t_ex,
            self.p_0,
            self.p_ex,
            param_dict=self.param_dict,
        )

        I_curv = (
            (1.0 - self.bsf)
            * self.I0
            * self.V.omega
            / 2.0
            * (
                np.exp(-(2 * self.V.tau / self._mu_0))
                * (2 * self.V.tau / self._mu_0 ** 3)
                * (
                    np.sin(self.t_0) ** 2
                    + 1.0
                    - 2.0 * self.V.tau / self._mu_0 * np.sin(self.t_0) ** 2
                )
                * p_val
                + (
                    np.exp(-(2 * self.V.tau / self._mu_0))
                    * 4.0
                    * self.V.tau
                    / self._mu_0 ** 2
                    * np.sin(self.t_0)
                )
                * p_slope
                + (1 - np.exp(-(2 * self.V.tau / self._mu_0))) * p_curv
            )
        )

        if sig0 is False and dB is False:
            return I_curv
        else:
            I_slope = self.volume_slope(dB=False, sig0=False)
            I_val = self.volume()
            if sig0 is True and dB is False:
                return (
                    4.0
                    * np.pi
                    * (
                        self._mu_0 * I_curv
                        - 2.0 * np.sin(self.t_0) * I_slope
                        - self._mu_0 * I_val
                    )
                )
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * (I_curv / I_val - I_slope ** 2 / I_val ** 2)
            elif sig0 is True and dB is True:
                return (
                    10.0
                    / np.log(10)
                    * (I_curv / I_val - I_slope ** 2 / I_val ** 2 - self._mu_0 ** (-2))
                )

    def tot_slope(self, sig0=False, dB=False):
        """
        numerical value of the (!monostatic!) slope of total
        contribution (surface + volume)

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic slope of the
            total-contribution
        """

        I_slope = self.volume_slope(dB=False, sig0=False) + self.surface_slope(
            dB=False, sig0=False
        )

        if sig0 is False and dB is False:
            return I_slope
        else:
            I_val = self.volume() + self.surface()
            if sig0 is True and dB is False:
                return 4.0 * np.pi * (self._mu_0 * I_slope - np.sin(self.t_0) * I_val)
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * I_slope / I_val
            elif sig0 is True and dB is True:
                return 10.0 / np.log(10) * (I_slope / I_val - np.tan(self.t_0))

    def tot_curv(self, sig0=False, dB=False):
        """
        numerical value of the (!monostatic!) curvature of
        total contribution (surface + volume)

        Parameters
        ----------
        dB : bool (default = False)
             indicator if the derivative is calculated for
             the dB values or for the linear values
        sig0 : bool (default = False)
               indicator if the derivative is calculated for
               the intensity (False) or for
               sigma_0 = 4 * pi * cos(t_0) * intensity (True)

        Returns
        -------
        - : array_like(float)
            Numerical value of the monostatic curvature of the
            total-contribution
        """

        I_curv = self.volume_curv(dB=False, sig0=False) + self.surface_curv(
            dB=False, sig0=False
        )

        if sig0 is False and dB is False:
            return I_curv
        else:
            I_slope = self.volume_slope(dB=False, sig0=False) + self.surface_slope(
                dB=False, sig0=False
            )
            I_val = self.volume() + self.surface()
            if sig0 is True and dB is False:
                return (
                    4.0
                    * np.pi
                    * (
                        self._mu_0 * I_curv
                        - 2.0 * np.sin(self.t_0) * I_slope
                        - self._mu_0 * I_val
                    )
                )
            elif sig0 is False and dB is True:
                return 10.0 / np.log(10) * (I_curv / I_val - I_slope ** 2 / I_val ** 2)
            elif sig0 is True and dB is True:
                return (
                    10.0
                    / np.log(10)
                    * (I_curv / I_val - I_slope ** 2 / I_val ** 2 - self._mu_0 ** (-2))
                )

    def interaction(self):
        """
        Numerical evaluation of the interaction-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#interaction_contribution)

        Returns
        -------
        - : array_like(float)
            Numerical value of the interaction-contribution for
            the given set of parameters
        """

        Fint1 = self._calc_Fint_1()
        Fint2 = self._calc_Fint_2()

        Iint = (
            self.I0
            * self._mu_0
            * self.V.omega
            * (
                np.exp(-self.V.tau / self._mu_ex) * Fint1
                + np.exp(-self.V.tau / self._mu_0) * Fint2
            )
        )

        return self.SRF.NormBRDF * (1.0 - self.bsf) * Iint

    def _calc_Fint_1(self):
        """
        Numerical evaluation of the F_int() function used in the definition
        of the interaction-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#F_int)

        Returns
        -------
        S : array_like(float)
            Numerical value of F_int for the given set of parameters
        """
        mu1, mu2, phi1, phi2 = self._mu_0, self._mu_ex, self.p_0, self.p_ex

        # evaluate fn-coefficients
        if self.lambda_backend == "symengine":
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )

            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(args))
        else:
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )
            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(*args))

        multip = self._mu_0_x * self._S2_mu(mu1, self.V.tau)
        S = np.sum(fn * multip, axis=0)
        return S

    def _calc_Fint_2(self):
        """
        Numerical evaluation of the F_int() function used in the definition
        of the interaction-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#F_int)

        Returns
        -------
        S : array_like(float)
            Numerical value of F_int for the given set of parameters
        """
        mu1, mu2, phi1, phi2 = self._mu_ex, self._mu_0, self.p_ex, self.p_0

        # evaluate fn-coefficients
        if self.lambda_backend == "symengine":
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )

            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(args))
        else:
            args = np.broadcast_arrays(
                np.arccos(mu1),
                phi1,
                np.arccos(mu2),
                phi2,
                *self.param_dict.values(),
            )
            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(*args))

        multip = self._mu_ex_x * self._S2_mu(mu1, self.V.tau)
        S = np.sum(fn * multip, axis=0)
        return S

    @property
    @lru_cache()
    def _mu_0_x(self):
        nmax = self.V.ncoefs + self.SRF.ncoefs - 1
        mux = np.array([self._mu_0 ** (n + 1) for n in range(nmax)])
        return mux

    @property
    @lru_cache()
    def _mu_ex_x(self):
        nmax = self.V.ncoefs + self.SRF.ncoefs - 1
        mux = np.array([self._mu_ex ** (n + 1) for n in range(nmax)])
        return mux

    def _S2_mu(self, mu, tau):
        nmax = self.V.ncoefs + self.SRF.ncoefs - 1

        hlp1 = (
            np.exp(-tau / mu) * np.log(mu / (1.0 - mu))
            - expi(-tau)
            + np.exp(-tau / mu) * expi(tau / mu - tau)
        )

        # cache the results of the expn-evaluations to speed up the S2 loop
        @lru_cache()
        def innerfunc(k):
            return mu ** (-k) * (expn(k + 1.0, tau) - np.exp(-tau / mu) / k)

        S2 = np.array(
            [sum(innerfunc(k) for k in range(1, (n + 1) + 1)) for n in range(nmax)]
        )

        # clear the cache since tau might have changed!
        innerfunc.cache_clear()

        return S2 + hlp1

    def _dvolume_dtau(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to tau
        Returns
        --------
        dvdt : array_like(float)
               Numerical value of dIvol/dtau for the given set of parameters
        """

        dvdt = (
            self.I0
            * self.V.omega
            * (self._mu_0 / (self._mu_0 + self._mu_ex))
            * (
                (1.0 / self._mu_0 + 1.0 / self._mu_ex ** (-1))
                * np.exp(-self.V.tau / self._mu_0 - self.V.tau / self._mu_ex)
            )
            * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        return (1.0 - self.bsf) * dvdt

    def _dvolume_domega(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to omega
        Returns
        --------
        dvdo : array_like(float)
               Numerical value of dIvol/domega for the given set of parameters
        """

        dvdo = (
            (self.I0 * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex)))
            * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        return (1.0 - self.bsf) * dvdo

    def _dvolume_dbsf(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to omega
        Returns
        --------
        dvdo : array_like(float)
               Numerical value of dIvol/domega for the given set of parameters
        """

        vol = (
            (self.I0 * self.V.omega * self._mu_0 / (self._mu_0 + self._mu_ex))
            * (1.0 - np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex)))
            * self.V.p(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        return -vol

    def _dvolume_dR(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to R (the hemispherical reflectance)

        Returns
        -------
        dvdr : array_like(float)
               Numerical value of dIvol/dR for the given set of parameters
        """

        dvdr = 0.0

        return dvdr

    def _dsurface_dtau(self):
        """
        Numerical evaluation of the derivative of the
        surface-contribution with respect to tau
        Returns
        --------
        dsdt : array_like(float)
               Numerical value of dIsurf/dtau for the given set of parameters
        """

        dsdt = (
            self.I0
            * (-1.0 / self._mu_0 - 1.0 / self._mu_ex)
            * np.exp(-self.V.tau / self._mu_0 - self.V.tau / self._mu_ex)
            * self._mu_0
            * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex, self.param_dict)
        )

        # Incorporate BRDF-normalization factor
        dsdt = self.SRF.NormBRDF * (1.0 - self.bsf) * dsdt

        return dsdt

    def _dsurface_domega(self):
        """
        Numerical evaluation of the derivative of the
        surface-contribution with respect to omega
        Returns
        --------
        dsdo : array_like(float)
               Numerical value of dIsurf/domega for the given set of parameters
        """

        dsdo = 0.0

        return dsdo

    def _dsurface_dR(self):
        """
        Numerical evaluation of the derivative of the
        surface-contribution with respect to R (the hemispherical reflectance)

        Returns
        -------
        dsdr : array_like(float)
               Numerical value of dIsurf/dR for the given set of parameters
        """

        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex))) * I_bs

        return (1.0 - self.bsf) * Isurf + self.bsf * I_bs

    def _dsurface_dbsf(self):
        """
        Numerical evaluation of the surface-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        -------
        - : array_like(float)
            Numerical value of the surface-contribution for the
            given set of parameters
        """
        # bare soil contribution
        I_bs = (
            self.I0
            * self._mu_0
            * self.SRF.brdf(
                self.t_0,
                self.t_ex,
                self.p_0,
                self.p_ex,
                param_dict=self.param_dict,
            )
        )

        Isurf = (
            (np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex)))
            * I_bs
            * np.ones_like(self.t_0)
        )

        return self.SRF.NormBRDF * (I_bs - Isurf)

    @lru_cache(20)
    def _d_surface_dummy_lambda(self, key):
        """
        a cached lambda-function for computing
        the derivative of the surface-function
        with respect to a given parameter

        Parameters
        ----------
        key : str
            the parameter to use.

        Returns
        -------
        dummyd : callable
            a function that calculates the derivative with respect to key.

        """
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return sp.lambdify(
            args,
            sp.diff(self.SRF._func, sp.Symbol(key)),
            modules=["numpy", "sympy"],
        )

    @lru_cache(20)
    def _d_volume_dummy_lambda(self, key):
        """same as _d_surface_dummy_lambda but for volume"""

        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(self.param_dict.keys())

        return sp.lambdify(
            args,
            sp.diff(self.V._func, sp.Symbol(key)),
            modules=["numpy", "sympy"],
        )

    def _d_surface_ddummy(self, key):
        """
        Generation of a function that evaluates the derivative of the
        surface-contribution with respect to the provided key

        Parameters
        ----------
        key : string


        Returns
        -------
        - : array_like(float)
            Numerical value of dIsurf/dkey for the given set of parameters
        """

        dI_bs = (
            self.I0
            * self._mu_0
            * self._d_surface_dummy_lambda(key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
        )

        dI_s = (np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex))) * dI_bs

        return self.SRF.NormBRDF * ((1.0 - self.bsf) * dI_s + self.bsf * dI_bs)

    def _d_volume_ddummy(self, key):
        """same as d_surface_ddummy but for volume"""

        dIvol = (
            self.I0
            * self.V.omega
            * self._mu_0
            / (self._mu_0 + self._mu_ex)
            * (1.0 - np.exp(-(self.V.tau / self._mu_0) - (self.V.tau / self._mu_ex)))
            * self._d_volume_dummy_lambda(key)(
                self.t_0, self.t_ex, self.p_0, self.p_ex, **self.param_dict
            )
        )
        return (1.0 - self.bsf) * dIvol

    def jacobian(self, dB=False, sig0=False, param_list=["omega", "tau", "NormBRDF"]):
        """
        Returns the jacobian of the total backscatter with respect
        to the parameters provided in param_list.
        (default: param_list = ['omega', 'tau', 'NormBRDF'])

        The jacobian can be evaluated for measurements in linear or dB
        units, and for either intensity- or sigma_0 values.

        Note:
            The contribution of the interaction-term is currently
            not considered in the calculation of the jacobian!

        Parameters
        ----------
        dB : boolean (default = False)
             Indicator whether linear or dB units are used.
             The applied relation is given by:

             dI_dB(x)/dx =
             10 / [log(10) * I_linear(x)] * dI_linear(x)/dx

        sig0 : boolean (default = False)
               Indicator wheather intensity- or sigma_0-values are used
               The applied relation is given by:

               sig_0 = 4 * pi * cos(inc) * I

               where inc denotes the incident zenith-angle and I is the
               corresponding intensity
        param_list : list
                     a list of strings that correspond to the parameters
                     for which the jacobian should be evaluated.

                     possible values are: 'omega', 'tau' 'NormBRDF' and
                     any string corresponding to a sympy.Symbol used in the
                     definition of V or SRF

        Returns
        -------
        jac : array-like(float)
              The jacobian of the total backscatter with respect to
              omega, tau and NormBRDF
        """

        if sig0 is True and dB is False:
            norm = 4.0 * np.pi * np.cos(self.t_0)
        elif dB is True:
            norm = 10.0 / (np.log(10.0) * (self.surface() + self.volume()))
        else:
            norm = 1.0

        jac = []
        for key in param_list:

            if key == "omega":
                jac += [(self._dsurface_domega() + self._dvolume_domega()) * norm]
            elif key == "tau":
                jac += [(self._dsurface_dtau() + self._dvolume_dtau()) * norm]
            elif key == "NormBRDF":
                jac += [(self._dsurface_dR() + self._dvolume_dR()) * norm]
            elif key == "bsf":
                jac += [(self._dsurface_dbsf() + self._dvolume_dbsf()) * norm]
            elif key in self.param_dict:
                jac += [
                    (self._d_surface_ddummy(key) + self._d_volume_ddummy(key)) * norm
                ]
            else:
                assert False, (
                    "error in jacobian calculation... "
                    + str(key)
                    + " is not in param_dict"
                )

        return jac

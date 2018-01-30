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
# import time

try:
    # if symengine is available, use it to perform series-expansions
    # this try-exept is necessary since symengine does currently not
    # build correctly with conda using a python 2.7 environment
    from symengine import expand
    from symengine import Lambdify as lambdify_seng
except ImportError:
    from sympy import expand
    # print('symengine could not be imported fn-function generation')


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
           (if geometry is 'mono', theta_ex is automatically set to t_0)

    p_ex : array_like(float)
           array of exit azimuth-angles in radians
           (if geometry is 'mono', phi_ex is automatically set to p_0 + np.pi)

    V : rt1.volume
        random object from rt1.volume class

    SRF : surface
          random object from rt1.surface class

    fn : array_like(sympy expression), optional (default = None)
         optional input of pre-calculated array of sympy-expressions
         to speedup calculations where the same fn-coefficients can be used.
         if None, the coefficients will be calculated automatically at the
         initialization of the RT1-object

    _fnevals : callable, optional (default = None)
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
    param_dict : dict (default = {})
                 a dictionary to assign numerical values to sympy.Symbols
                 appearing in the definitions of V and SRF.
    lambda_backend : str (default = 'cse')
                     indicator to select the module that shall be used
                     to compile a function for numerical evaluation of the
                     fn-coefficients.

                     TODO(update this) possible values are:
                         - 'sympy' :  sympy.lambdify is used to compile
                           the _fnevals function
                         - 'symengine' : symengine.LambdifyCSE is used to
                           compile the _fnevals function. This results in
                           considerable speedup for long fn-coefficients
                         - 'cse' : sympy.lambdify is used together with
                           sympy.cse to generate a fast evaluation-function
    int_Q : bool (default = True)
            indicator whether the interaction-term should be calculated or not
    verbosity : int
            select the verbosity level of the module to get status-reports
                - 0 : print nothing
                - 1 : print some infos during runtime
                - 2 : print more
                - >=3 : print all
    """

    def __init__(self, I0, t_0, t_ex, p_0, p_ex,
                 V=None, SRF=None, fn_input=None, _fnevals_input=None,
                 geometry='vvvv', param_dict={},
                 lambda_backend='cse', int_Q=True, verbosity = 1):

        assert isinstance(geometry, str), ('ERROR: geometry must be ' +
                                           'a 4-character string')
        assert len(geometry) == 4, ('ERROR: geometry must be ' +
                                           'a 4-character string')
        self.geometry = geometry

        self.I0 = I0
        self.param_dict = param_dict
        self.lambda_backend = lambda_backend
        self.int_Q = int_Q

        assert V is not None, 'ERROR: needs to provide volume information'
        self.V = V

        assert SRF is not None, 'ERROR: needs to provide surface information'
        self.SRF = SRF

        self.fn_input = fn_input
        self._fnevals_input = _fnevals_input

        self._set_t_0(t_0)
        self._set_t_ex(t_ex)
        self._set_p_0(p_0)
        self._set_p_ex(p_ex)

        self.verbosity = verbosity

        # self._set_fn(fn)
        # self._set_fnevals(_fnevals)

        # the asserts for omega & tau are performed inside the RT1-class
        # rather than the Volume-class to allow calling Volume-elements without
        # providing omega & tau which is needed to generate linear-combinations
        # of Volume-elements with unambiguous tau- & omega-specifications

        assert self.V.omega is not None, ('Single scattering albedo ' +
                                          'needs to be provided')
        assert self.V.tau is not None, 'Optical depth needs to be provided'

        # self.V.omega[0] must be used instead of self.V.omega since the
        # setter functions for omega, tau and NormBRDF add an additional
        # axis to the given input. Checking for sp.Basic is sufficient
        # to distinguish if the input was given as a sympy equation. For
        # details see: http://docs.sympy.org/latest/guide.html#basics
        if not isinstance(self.V.omega[0], sp.Basic):
            assert np.any(self.V.omega >= 0.), ('Single scattering albedo ' +
                                                 'must be greater than 0')
        if not isinstance(self.V.tau[0], sp.Basic):
            assert np.any(self.V.tau >= 0.), ('Optical depth ' +
                                                 'must be greater than 0')
        if not isinstance(self.SRF.NormBRDF[0], sp.Basic):
            assert np.any(self.SRF.NormBRDF >= 0.), ('NormBRDF ' +
                                                 'must be greater than 0')

# TODO  fix asserts to allow symbolic parameters
        # check if all parameters have been provided (and also if no
        # unused parameter has been specified)

#        refset = set(sp.var(('theta_0', 'phi_0', 'theta_ex', 'phi_ex') +
#                            tuple(map(str, self.param_dict.keys()))))
#
#        funcset = self.V._func.free_symbols | self.SRF._func.free_symbols
#
#        if refset <= funcset:
#            errdict = ' in the definition of V and SRF'
#        elif refset >= funcset:
#            errdict = ' in the definition of param_dict'
#        else:
#            errdict = ' in the definition of V, SRF and param_dict'
#
#        assert (funcset == refset), ('false parameter-specification, please ' +
#                                     'check assignment of the parameters '
#                                     + str(refset ^ funcset) + errdict)

    def prv(self, v, msg):
        '''
        function to set print output based on verbosity level v.
        possible values for v:
            - 0 : print nothing
            - 1 : print some infos during runtime
            - 2 : print more
            - >=3 : print all
        '''
        if self.verbosity >= v:
            print(msg)

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
            self.prv(1, 'evaluating fn-coefficients...')

            import timeit
            tic = timeit.default_timer()
            # precalculate the expansiion coefficients for the interaction term
            expr_int = self._calc_interaction_expansion()
            toc = timeit.default_timer()
            self.prv(2,
                     'expansion calculated, it took ' + str(toc-tic) + ' sec')

            # extract the expansion coefficients
            tic = timeit.default_timer()
            self.__fn = self._extract_coefficients(expr_int)
            toc = timeit.default_timer()
            self.prv(2,
                     'coefficients extracted, it took ' +
                     str(toc - tic) + ' sec')
        else:
            self.prv(3, 'using provided fn-coefficients')
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
            self.prv(1, 'generation of _fnevals functions...')
            import timeit
            tic = timeit.default_timer()

            # define new lambda-functions for each fn-coefficient
            variables = sp.var(('theta_0', 'phi_0', 'theta_ex', 'phi_ex') +
                               tuple(map(str, self.param_dict.keys())))

            # use symengine's Lambdify if symengine has been used within
            # the fn-coefficient generation
            if self.lambda_backend == 'symengine':
                self.prv(1,
                         'symengine currently only working with dev-version!!')
                # set lambdify module
                lambdify = lambdify_seng

                self.__fnevals = lambdify(list(variables),
                                          self.fn, order='F')

            elif self.lambda_backend == 'cse':
                self.prv(1, 'cse - sympy')

                # define a generator function to use deferred vectors for cse
                def defgen(name='defvec'):
                    x = sp.DeferredVector(name)
                    n = 0
                    while(True):
                        yield x[n]
                        n += 1

                variables = sp.var(('theta_0', 'phi_0', 'theta_ex', 'phi_ex') +
                                   tuple(map(str, self.param_dict.keys())))

                tic = timeit.default_timer()
                # convert the equations to sympy formulas
                # (needed if symengine has been used to expand the products)
                funs = list(map(sp.sympify, self.fn))
                toc = timeit.default_timer()
                self.prv(2, 'sympifying took ' + str(toc-tic))

                # initialize arrasy that store the cse-functions and variables
                fn_cse_funs = []
                fn_cse_repfuncs = []
                fn_cse_vars = []

                # initialize a deferred vector and an associated generator

                for nf, fncoef in enumerate(funs):
                    defvecgen = defgen('defvec')

                    # evaluate cse functions and variables for the i'th coef.
                    fn_repl, fn_csefun = sp.cse(fncoef, symbols=defvecgen,
                                                order='none')
                    self.prv(2,
                             'fn_repl has ' + str(len(fn_repl)) + ' elements')
                    # store for later use
                    fn_cse_funs = fn_cse_funs + [fn_csefun[0]]

                    fn_funcs = []
                    cse_variables = []  # list(variables)
                    for i, ff in enumerate(fn_repl):
                        # use a deferred vector in lambdify to avoid exceeding
                        # allowed maximum number of arguments
                        defvec = sp.DeferredVector('defvec')
                        symbs = [defvec] + list(variables)
                        fn_funcs.append(sp.lambdify(symbs, ff[1],
                                                    modules='numpy'))
                        cse_variables.append(ff[0])

                    fn_cse_repfuncs = fn_cse_repfuncs + [fn_funcs]
                    fn_cse_vars = fn_cse_vars + [cse_variables]
                    self.prv(2, 'cse of coefficient ' + str(nf) + '/' +
                          str(len(funs)) + ' finished')

                ifuncs = []
                for n in range(len(fn_cse_funs)):
                    # use a deferred vector in lambdify to avoid exceeding
                    # allowed maximum number of arguments
                    defvec = sp.DeferredVector('defvec')
                    symbs = [defvec] + list(variables)
                    ifunc = sp.lambdify(symbs, fn_cse_funs[n],
                                        modules='numpy')
                    ifuncs = ifuncs + [ifunc]

                # define a function that evaluates the i'th fn-coefficient
                def fneval(*variables):
                    sol = []
                    for n in range(len(fn_cse_funs)):
                        xs = []
                        i = 0
                        for f in fn_cse_repfuncs[n]:
                            # broadcast arrays to ensure correct shape
                            xscalc = np.broadcast_arrays(f(xs, *variables),
                                                         variables[0])[0]
                            xs = xs + [xscalc]
                            i = i + 1
                        sol = sol + [ifuncs[n](xs, *variables)]
                    return sol

                self.__fnevals = fneval

            elif self.lambda_backend == 'sympy':
                self.prv(1, 'sympy')
                # set lambdify module
                lambdify = sp.lambdify

                sympy_fn = list(map(sp.sympify, self.fn))

                self.__fnevals = lambdify((variables),
                                          sp.sympify(sympy_fn),
                                          modules=["numpy", "sympy"],
                                          dummify=False)

                self.__fnevals.__doc__ = ('''
                                    A function to numerically evaluate the
                                    fn-coefficients a for given set of
                                    incidence angles and parameter-values
                                    as defined in the param_dict dict.

                                    The call-signature is:
                                        RT1-object._fnevals(theta_0, phi_0, \
                                        theta_ex, phi_ex, *param_dict.values())
                                    ''')

            elif self.lambda_backend == 'cse_symengine_sympy':
                '''
                sympy's cse functionality is used to avoid sympifying the whole
                fn-coefficient array.

                symengines lambdify is used since it turns out to be superior
                in terms of computational speed compared to symengines
                currently available Lambdify function.
                '''

                self.prv(1,
                         'use symengines cse and sympys lambdify with numpy')
                variables = sp.var(('theta_0', 'phi_0', 'theta_ex', 'phi_ex') +
                                   tuple(map(str, self.param_dict.keys())))

                from symengine import cse as cse_seng
                funs = self.fn

                # initialize array that store the cse-functions and variables
                fn_cse_funs = []  # resulting cse-function for each coefficient
                fn_cse_repfuncs = []  # replacement functions for each coef.
                for nf, fncoef in enumerate(funs):
                    # evaluate cse functions and variables for the i'th coef.
                    fn_repl, fn_csefun = cse_seng([fncoef])

                    self.prv(2,
                             'fn_repl has ' + str(len(fn_repl)) + ' elements')

                    # generate lambda-functions for replacements
                    fn_rfuncs = {}
                    defvec_subs = {}
                    for i, repl in enumerate(fn_repl):
                        defvec = sp.DeferredVector('defvec')
                        defvec_subs[sp.sympify(repl[0])] = defvec[i]
                        # replace xn's with deferred vector elements
                        funcreplaced = sp.sympify(
                                repl[1]).xreplace(defvec_subs)
                        symbs = [defvec] + list(variables)
                        fn_rfuncs[defvec[i]] = sp.lambdify(
                            symbs,
                            funcreplaced,
                            modules=['numpy'])

                    fn_cse_repfuncs += [fn_rfuncs]

                    # generate lambda-functions for cse_functions
                    fn_csefun_defvec = sp.sympify(
                            fn_csefun[0]).xreplace(defvec_subs)
                    funargs = [defvec] + list(variables)
                    fn_cse_funs += [sp.lambdify(funargs,
                                                fn_csefun_defvec,
                                                modules=['numpy'])]

                    self.prv(2, 'cse of coefficient ' + str(nf) + '/' +
                             str(len(funs)) + ' finished')

                def fneval(*variables):
                    sol = []
                    for n, fn_cse_fun in enumerate(fn_cse_funs):
                        replvec = []
                        for key in fn_cse_repfuncs[n]:
                            replvec += [fn_cse_repfuncs[n][key](
                                replvec, *variables)]
                        sol += [fn_cse_fun(replvec, *variables)]
                    return sol

                self.__fnevals = fneval

            elif self.lambda_backend == 'cse_seng_sp_newlambdify':
                '''
                sympy's cse functionality is used to avoid sympifying the whole
                fn-coefficient array.

                a customly defined lambdify-function is used since the
                functions generated with sympy's lambdify with the option
                modules = ['numpy'] are not pickleable.
                (necessary for multiprocessing purposes)
                '''

                # define a function for lambdification
                def newlambdify(args, funs):
                    from sympy.printing.lambdarepr import NumPyPrinter
                    from sympy.utilities.lambdify import lambdastr
                    funcstr = lambdastr(args, funs, printer=NumPyPrinter)

                    funcstr = funcstr.replace(
                            'pi', 'np.pi').replace(
                            'sin', 'np.sin').replace(
                            'cos', 'np.cos').replace(
                            'sqrt', 'np.sqrt')

                    return eval(funcstr)

                self.prv(1,
                         'use symengines cse and sympys lambdify with ' +
                         'manually defined lambdify function')
                variables = sp.var(('theta_0', 'phi_0', 'theta_ex', 'phi_ex') +
                                   tuple(map(str, self.param_dict.keys())))

                from symengine import cse as cse_seng
                funs = self.fn

                # initialize array that store the cse-functions and variables
                fn_cse_funs = []  # resulting cse-function for each coefficient
                fn_cse_repfuncs = []  # replacement functions for each coef.
                for nf, fncoef in enumerate(funs):
                    # evaluate cse functions and variables for the i'th coef.
                    fn_repl, fn_csefun = cse_seng([fncoef])

                    self.prv(2,
                             'fn_repl has ' + str(len(fn_repl)) + ' elements')

                    # generate lambda-functions for replacements
                    fn_rfuncs = {}
                    defvec_subs = {}
                    for i, repl in enumerate(fn_repl):
                        defvec = sp.DeferredVector('defvec')
                        defvec_subs[sp.sympify(repl[0])] = defvec[i]
                        # replace xn's with deferred vector elements
                        funcreplaced = sp.sympify(
                                repl[1]).xreplace(defvec_subs)
                        symbs = [defvec] + list(variables)
                        fn_rfuncs[defvec[i]] = newlambdify(
                            symbs,
                            funcreplaced)

                    fn_cse_repfuncs += [fn_rfuncs]

                    # generate lambda-functions for cse_functions
                    fn_csefun_defvec = sp.sympify(
                            fn_csefun[0]).xreplace(defvec_subs)
                    funargs = [defvec] + list(variables)
                    fn_cse_funs += [newlambdify(funargs,
                                                fn_csefun_defvec)]

                    self.prv(2, 'cse of coefficient ' + str(nf) + '/' +
                             str(len(funs)) + ' finished')

                def fneval(*variables):
                    sol = []
                    for n, fn_cse_fun in enumerate(fn_cse_funs):
                        replvec = []
                        for key in fn_cse_repfuncs[n]:
                            replvec += [fn_cse_repfuncs[n][key](
                                replvec, *variables)]
                        sol += [fn_cse_fun(replvec, *variables)]
                    return sol

                self.__fnevals = fneval

            else:
                self.prv(1, 'lambda_backend "' + self.lambda_backend +
                         '" is not available')

            toc = timeit.default_timer()
            self.prv(2,
                     'lambdification finished, it took ' +
                     str(toc - tic) + ' sec')

        else:
            self.prv(3, 'using provided _fnevals-functions')
            self.__fnevals = _fnevals

    _fnevals = property(_get_fnevals, _set_fnevals)

    def _get_t_0(self):
        return self.__t_0

    def _set_t_0(self, t_0):
        # if t_0 is given as scalar input, convert it to 1d numpy array
        if np.isscalar(t_0):
            t_0 = np.array([t_0])
        self.__t_0 = t_0
        # if geometry is mono, set t_ex to t_0
        if self.geometry == 'mono':
            self._set_t_ex(t_0)

    t_0 = property(_get_t_0, _set_t_0)

    def _get_t_ex(self):
        return self.__t_ex

    def _set_t_ex(self, t_ex):
        # if geometry is mono, set t_ex to t_0
        if self.geometry == 'mono':
            t_ex = self._get_t_0()
        else:
            # if t_ex is given as scalar input, convert it to 1d numpy array
            if np.isscalar(t_ex):
                t_ex = np.array([t_ex])
        self.__t_ex = t_ex
    t_ex = property(_get_t_ex, _set_t_ex)

    def _get_p_0(self):
        return self.__p_0

    def _set_p_0(self, p_0):
        # if p_o is given as scalar input, convert it to 1d numpy array
        if np.isscalar(p_0):
            p_0 = np.array([p_0])
        self.__p_0 = p_0
        # if geometry is mono, set p_ex to p_0
        if self.geometry == 'mono':
            self._set_p_ex(p_0)

    p_0 = property(_get_p_0, _set_p_0)

    def _get_p_ex(self):
        return self.__p_ex

    def _set_p_ex(self, p_ex):
        # if geometry is mono, set p_ex to p_0
        if self.geometry == 'mono':
            p_ex = self._get_p_0() + np.pi
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
        --------
        fn : list(sympy expressions)
             A list of sympy expressions that represent the fn-coefficients
             associated with the given input-equation (expr).

        """

        theta_s = sp.Symbol('theta_s')

        N_fn = self.SRF.ncoefs + self.V.ncoefs - 1

        fn = []

        # find f_0 coefficient
        repl0 = dict([[sp.cos(theta_s), 0]])
        fn = fn + [expr.xreplace(repl0)]

        # find f_1 coefficient
        repl1 = dict([[sp.cos(theta_s)**i, 0] for i in list(range(N_fn, 0, -1))
                     if i != 1] + [[sp.cos(theta_s), 1]])
        fn = fn + [expr.xreplace(repl1) - fn[0]]

        for n in np.arange(2, N_fn, dtype=int):
            repln = dict([[sp.cos(theta_s)**int(n), 1]])
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
        --------
        res : sympy expression
              A fully expanded expression that can be used as
              input for _extract_coefficients()

        """
        # preevaluate expansions for volume and surface phase functions
        # this returns symbolic code to be then further used

        volexp = self.V.legexpansion(self.t_0, self.t_ex,
                                      self.p_0, self.p_ex,
                                      self.geometry).doit()

        brdfexp = self.SRF.legexpansion(self.t_0, self.t_ex,
                                        self.p_0, self.p_ex,
                                        self.geometry).doit()

        # preparation of the product of p*BRDF for coefficient retrieval
        # this is the eq.23. and would need to be integrated from 0 to 2pi
        fPoly = expand(2 * sp.pi * volexp * brdfexp)

        # do integration of eq. 23
        expr = self._integrate_0_2pi_phis(fPoly)

        # now we do still simplify the expression to be able to express
        # things as power series of cos(theta_s)
        theta_s = sp.Symbol('theta_s')
        replacements = [(sp.sin(theta_s) ** i,
                         expand((1. - sp.cos(theta_s) ** 2)
                                ** sp.Rational(i, 2)))
                        for i in range(1, self.SRF.ncoefs + self.V.ncoefs - 1)
                        if i % 2 == 0]

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
            return 0.

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

        phi_s = sp.Symbol('phi_s')

        # replace first all odd powers of sin(phi_s) as these are
        # all zero for the integral
        replacements1 = [(sp.sin(phi_s) ** i, 0.)
                         for i in range(1, self.SRF.ncoefs +
                                        self.V.ncoefs + 1) if i % 2 == 1]

        # then substitute the sine**2 by 1-cos**2
        replacements1 = (replacements1 +
                         [(sp.sin(phi_s) ** i,
                           expand((1. -
                                   sp.cos(phi_s) ** 2) ** sp.Rational(i, 2)))
                          for i in range(2, self.SRF.ncoefs +
                                         self.V.ncoefs + 1) if i % 2 == 0])

        res = expand(expr.xreplace(dict(replacements1)))

        # replacements need to be done simultaneously, otherwise all
        # remaining sin(phi_s)**even will be replaced by 0

        # integrate the cosine terms
        replacements3 = [(sp.cos(phi_s) ** i, self._cosintegral(i))
                         for i in range(1, self.SRF.ncoefs +
                                        self.V.ncoefs + 1)]

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

        # the following if query ensures that volume- and interaction-terms
        # are only calculated if tau > 0.
        # (to avoid nan-values from invalid function-evaluations)

        if self.V.tau.shape == (1,):
            Isurf = self.surface()
            # differentiation for non-existing canopy, as otherwise NAN values
            if self.V.tau > 0.:
                Ivol = self.volume()
                if self.int_Q is True:
                    Iint = self.interaction()
                else:
                    Iint = np.array([0.])
            else:
                Ivol = np.array([0.])
                Iint = np.array([0.])
        else:
            # calculate surface-term (valid for any tau-value)
            Isurf = self.surface()

            # store initial parameter-values
            old_t_0 = self.t_0
            old_p_0 = self.p_0
            old_t_ex = self.t_ex
            old_p_ex = self.p_ex

            old_tau = self.V._get_tau()
            old_omega = self.V._get_omega()
            old_NN = self.SRF._get_NormBRDF()

            # set mask for tau > 0.
            mask = old_tau > 0.
            valid_index = np.where(mask)
            inval_index = np.where(~mask)

            # set parameter-values to valid values for calculation
            self.t_0 = old_t_0[valid_index[0]]
            self.p_0 = old_p_0[valid_index[0]]
            self.t_ex = old_t_ex[valid_index[0]]
            self.p_ex = old_p_ex[valid_index[0]]

            # squeezing the arrays is necessary since the setter-function for
            # tau, omega and NormBRDF automatically adds an axis to the arrays!
            self.V.tau = np.squeeze(old_tau[valid_index[0]])
            if np.array(self.V.omega).size != 1:
                self.V.omega = np.squeeze(old_omega[valid_index[0]])
            if np.array(self.SRF.NormBRDF).size != 1:
                self.SRF.NormBRDF = np.squeeze(old_NN[valid_index[0]])

            # calculate volume and interaction term where tau-values are valid
            _Ivol = self.volume()
            if self.int_Q is True:
                _Iint = self.interaction()
            else:
                _Iint = np.full_like(self.t_0, 0.)

            # reset parameter values to old values
            self.t_0 = old_t_0
            self.p_0 = old_p_0
            self.t_ex = old_t_ex
            self.p_ex = old_p_ex

            # squeezing the arrays is necessary since the setter-function for
            # tau, omega and NormBRDF automatically add an axis to the arrays!
            self.V.tau = np.squeeze(old_tau)
            self.V.omega = np.squeeze(old_omega)
            self.SRF.NormBRDF = np.squeeze(old_NN)

            # combine calculated volume-contributions for valid tau-values
            # with zero-arrays for invalid tau-values
            Ivol = np.ones_like(self.t_0)
            Ivol[valid_index[0]] = _Ivol
            Ivol[inval_index[0]] = np.ones_like(Ivol[inval_index[0]]) * 0.

            # combine calculated interaction-contributions for valid tau-values
            # with zero-arrays for invalid tau-values
            if self.int_Q is True:
                Iint = np.ones_like(self.t_0)
                Iint[valid_index[0]] = _Iint
                Iint[inval_index[0]] = np.ones_like(Iint[inval_index[0]]) * 0.
            else:
                Iint = np.full_like(self.t_0, 0.)

        return Isurf + Ivol + Iint, Isurf, Ivol, Iint

    def surface(self):
        """
        Numerical evaluation of the surface-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#surface_contribution)

        Returns
        --------
        - : array_like(float)
            Numerical value of the surface-contribution for the
            given set of parameters
        """

        Isurf = (self.I0 * np.exp(-(self.V.tau / self._mu_0) -
                                  (self.V.tau / self._mu_ex)) * self._mu_0
                 * self.SRF.brdf(self.t_0, self.t_ex,
                                 self.p_0, self.p_ex,
                                 param_dict=self.param_dict))

        return self.SRF.NormBRDF * Isurf

    def volume(self):
        """
        Numerical evaluation of the volume-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#volume_contribution)

        Returns
        --------
        - : array_like(float)
            Numerical value of the volume-contribution for the
            given set of parameters
        """
        return ((self.I0 * self.V.omega *
                 self._mu_0 / (self._mu_0 + self._mu_ex))
                * (1. - np.exp(-(self.V.tau / self._mu_0) -
                               (self.V.tau / self._mu_ex)))
                * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex,
                            param_dict=self.param_dict))

    def interaction(self):
        """
        Numerical evaluation of the interaction-contribution
        (http://rt1.readthedocs.io/en/latest/theory.html#interaction_contribution)

        Returns
        --------
        - : array_like(float)
            Numerical value of the interaction-contribution for
            the given set of parameters
        """

        Fint1 = self._calc_Fint(self._mu_0, self._mu_ex, self.p_0, self.p_ex)
        Fint2 = self._calc_Fint(self._mu_ex, self._mu_0, self.p_ex, self.p_0)

        Iint = (self.I0 * self._mu_0 * self.V.omega *
                (np.exp(-self.V.tau / self._mu_ex) * Fint1 +
                 np.exp(-self.V.tau / self._mu_0) * Fint2))

        return self.SRF.NormBRDF * Iint

    def _calc_Fint(self, mu1, mu2, phi1, phi2):
        """
        Numerical evaluation of the F_int() function used in the definition
        of the interaction-contribution
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

        # evaluate fn-coefficients
        if self.lambda_backend == 'symengine':
            args = np.broadcast_arrays(np.arccos(mu1), phi1, np.arccos(mu2),
                                       phi2, *self.param_dict.values())
            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(args))
        else:
            args = np.broadcast_arrays(np.arccos(mu1), phi1, np.arccos(mu2),
                                       phi2, *self.param_dict.values())
            # to correct for 0 dimensional arrays if a fn-coefficient
            # is identical to 0 (in a symbolic manner)
            fn = np.broadcast_arrays(*self._fnevals(*args))

        nmax = len(fn)

        hlp1 = (np.exp(-self.V.tau / mu1) * np.log(mu1 / (1. - mu1))
                - expi(-self.V.tau) + np.exp(-self.V.tau / mu1)
                * expi(self.V.tau / mu1 - self.V.tau))

        S2 = np.array([np.sum(mu1 ** (-k) * (expn(k + 1., self.V.tau) -
                                             np.exp(-self.V.tau / mu1) / k)
                              for k in range(1, (n + 1) + 1))
                       for n in range(nmax)])

        mu = np.array([mu1 ** (n + 1) for n in range(nmax)])

        S = np.sum(fn * mu * (S2 + hlp1), axis=0)

        return S

    def _dvolume_dtau(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to tau
        Returns
        --------
        dvdt : array_like(float)
               Numerical value of dIvol/dtau for the given set of parameters
        """

        dvdt = (self.I0 * self.V.omega
                * (self._mu_0 / (self._mu_0 + self._mu_ex))
                * ((1. / self._mu_0 + 1. / self._mu_ex**(-1))
                   * np.exp(- self.V.tau / self._mu_0 -
                            self.V.tau / self._mu_ex))
                * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex,
                            self.param_dict))

        return dvdt

    def _dvolume_domega(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to omega
        Returns
        --------
        dvdo : array_like(float)
               Numerical value of dIvol/domega for the given set of parameters
        """

        dvdo = ((self.I0 * self._mu_0 / (self._mu_0 + self._mu_ex)) *
                (
                1. - np.exp(-(self.V.tau / self._mu_0) -
                            (self.V.tau / self._mu_ex))
                ) * self.V.p(self.t_0, self.t_ex, self.p_0, self.p_ex,
                              self.param_dict))

        return dvdo

    def _dvolume_dR(self):
        """
        Numerical evaluation of the derivative of the
        volume-contribution with respect to R (the hemispherical reflectance)
        Returns
        --------
        dvdr : array_like(float)
               Numerical value of dIvol/dR for the given set of parameters
        """

        dvdr = 0.

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

        dsdt = (self.I0
                * (- 1. / self._mu_0 - 1. / self._mu_ex)
                * np.exp(- self.V.tau / self._mu_0
                         - self.V.tau / self._mu_ex)
                * self._mu_0
                * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex,
                                self.param_dict))

        # Incorporate BRDF-normalization factor
        dsdt = self.SRF.NormBRDF * dsdt

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

        dsdo = 0.

        return dsdo

    def _dsurface_dR(self):
        """
        Numerical evaluation of the derivative of the
        surface-contribution with respect to R (the hemispherical reflectance)
        Returns
        --------
        dsdr : array_like(float)
               Numerical value of dIsurf/dR for the given set of parameters
        """

        dsdr = (self.I0
                * np.exp(-(self.V.tau / self._mu_0)
                         - (self.V.tau / self._mu_ex))
                * self._mu_0
                * self.SRF.brdf(self.t_0, self.t_ex, self.p_0, self.p_ex,
                                self.param_dict))

        return dsdr

    # define functions that evaluate the derivatives with
    # respect to the defined parameters
    def _d_surface_ddummy(self, key):
        '''
        Generation of a function that evaluates the derivative of the
        surface-contribution with respect to the provided key

        Parameters:
        ------------
        key : string


        Returns:
        --------
        - : array_like(float)
            Numerical value of dIsurf/dkey for the given set of parameters
        '''
        theta_0 = sp.Symbol('theta_0')
        theta_ex = sp.Symbol('theta_ex')
        phi_0 = sp.Symbol('phi_0')
        phi_ex = sp.Symbol('phi_ex')

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(
            self.param_dict.keys())

        dummyd = sp.lambdify(args,
                             sp.diff(self.SRF._func, sp.Symbol(key)),
                             modules=["numpy", "sympy"])

        dIsurf = (self.I0 *
                  np.exp(-(self.V.tau / self._mu_0) -
                         (self.V.tau / self._mu_ex))
                  * self._mu_0
                  * dummyd(self.t_0, self.t_ex, self.p_0, self.p_ex,
                           **self.param_dict)
                  )

        return self.SRF.NormBRDF * dIsurf

    def _d_volume_ddummy(self, key):
        theta_0 = sp.Symbol('theta_0')
        theta_ex = sp.Symbol('theta_ex')
        phi_0 = sp.Symbol('phi_0')
        phi_ex = sp.Symbol('phi_ex')

        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(
            self.param_dict.keys())

        dummyd = sp.lambdify(args,
                             sp.diff(self.V._func, sp.Symbol(key)),
                             modules=["numpy", "sympy"])

        dIvol = (self.I0 * self.V.omega
                 * self._mu_0 / (self._mu_0 + self._mu_ex)
                 * (1. - np.exp(-(self.V.tau / self._mu_0) -
                                (self.V.tau / self._mu_ex)))
                 * dummyd(self.t_0, self.t_ex, self.p_0, self.p_ex,
                          **self.param_dict))
        return dIvol

    def jacobian(self, dB=False, sig0=False,
                 param_list=['omega', 'tau', 'NormBRDF']):
        '''
        Returns the jacobian of the total backscatter with respect
        to the parameters provided in param_list.
        (default: param_list = ['omega', 'tau', 'NormBRDF'])

        The jacobian can be evaluated for measurements in linear or dB
        units, and for either intensity- or sigma_0 values.

        Note:
            The contribution of the interaction-term is currently
            not considered in the calculation of the jacobian!

        Parameters:
        -------------
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

        Returns:
        ---------
        jac : array-like(float)
              The jacobian of the total backscatter with respect to
              omega, tau and NormBRDF
        '''

        jacdict = {}
        if 'omega' in param_list:
            jacdict['omega'] = (self._dsurface_domega() +
                                self._dvolume_domega())
        if 'tau' in param_list:
            jacdict['tau'] = (self._dsurface_dtau() +
                              self._dvolume_dtau())

        if 'NormBRDF' in param_list:
            jacdict['NormBRDF'] = (self._dsurface_dR() +
                                   self._dvolume_dR())

        for key in self.param_dict:
            if key in param_list:
                jacdict[key] = (self._d_surface_ddummy(key) +
                                self._d_volume_ddummy(key))

        if sig0 is True and dB is False:
            norm = 4. * np.pi * np.cos(self.t_0)
        if dB is True:
            norm = 10. / (np.log(10.) * (self.surface()
                                         + self.volume()))
        else:
            norm = 1.

        # transform jacobian to the desired shape
        jac = [jacdict[str(key)] * norm for key in param_list]

        # ----- this is removed due to memory-overflow issues for large arrays
        # (converting them to a block_diag matrix yields too large arrays)
        # from scipy.linalg import block_diag
        # transform jacobian to the desired shape
        # if len(self.surface().shape) == 1:
        #     jac = [jacdict[str(key)] * norm for key in param_list]
        # else:
        #     jac = [block_diag(*(jacdict[str(key)] * norm))
        #            for key in param_list]

        return jac

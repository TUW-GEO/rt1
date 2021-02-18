"""Definition of volume phase scattering functions"""

import numpy as np
import sympy as sp
from functools import partial, update_wrapper
from .scatter import Scatter
from .rtplots import polarplot


class Volume(Scatter):
    """basic volume class"""

    def __init__(self, **kwargs):
        self.omega = kwargs.pop("omega", None)
        self.tau = kwargs.pop("tau", None)

        # set scattering angle generalization-matrix to [-1,1,1] if it is not
        # explicitly provided by the chosen class this results in a peak in
        # forward-direction which is suitable for describing volume-scattering
        # phase-functions
        self.a = getattr(self, "a", [-1.0, 1.0, 1.0])

        # add a quick way for visualizing the functions as polarplot
        self.polarplot = partial(polarplot, X=self)
        update_wrapper(self.polarplot, polarplot)

    def p(self, t_0, t_ex, p_0, p_ex, param_dict={}):
        """
        Calculate numerical value of the volume-scattering phase-function
        for chosen incidence- and exit angles.

        Parameters
        ----------
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
            Numerical value of the volume-scattering phase-function
        """
        # define sympy objects
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        # replace arguments and evaluate expression
        # sp.lambdify is used to allow array-inputs
        # for python > 3.5 unpacking could be used, i.e.:
        # pfunc = sp.lambdify((theta_0, theta_ex, phi_0, phi_ex,
        #                     *param_dict.keys()),
        #                    self._func, modules=["numpy", "sympy"])
        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(param_dict.keys())
        pfunc = sp.lambdify(args, self._func, modules=["numpy", "sympy"])

        # in case _func is a constant, lambdify will produce a function with
        # scalar output which is not suitable for further processing
        # (this happens e.g. for the Isotropic brdf).
        # The following query is implemented to ensure correct array-output:
        # TODO this is not a proper test !
        if not isinstance(
            pfunc(
                np.array([0.1, 0.2, 0.3]),
                0.1,
                0.1,
                0.1,
                **{key: 0.12 for key in param_dict.keys()}
            ),
            np.ndarray,
        ):
            pfunc = np.vectorize(pfunc)

        return pfunc(t_0, t_ex, p_0, p_ex, **param_dict)

    def p_theta_diff(
        self,
        t_0,
        t_ex,
        p_0,
        p_ex,
        geometry,
        param_dict={},
        return_symbolic=False,
        n=1,
    ):
        """
        Calculation of the derivative of p with respect to
        the scattering-angles t_ex

        Parameters
        ----------
        t_0 : array_like(float)
              array of incident zenith-angles in radians

        p_0 : array_like(float)
              array of incident azimuth-angles in radians

        t_ex : array_like(float)
               array of exit zenith-angles in radians

        p_ex : array_like(float)
               array of exit azimuth-angles in radians

        geometry : str
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
            geometry-parameter, please have a look at the
            "Evaluation Geometries" section of the documentation
            (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

        return_symbolic : bool (default = False)
                          indicator if symbolic result
                          should be returned
        n : int (default = 1)
            order of derivatives (d^n / d_theta^n)

        Returns
        -------
        sympy - expression
            The derivative of the BRDF with espect to the excident angle
            t_ex for the chosen geometry

        """

        # define sympy variables based on chosen geometry
        if geometry == "mono":
            assert len(np.unique(p_0)) == 1, (
                "p_0 must contain only a "
                + "single unique value for monostatic geometry"
            )

            theta_0 = sp.Symbol("theta_0")
            theta_ex = theta_0
            phi_0 = np.unique(p_0)[0]
            phi_ex = np.unique(p_0)[0] + sp.pi

            t_ex = t_0
            p_ex = p_0 + np.pi
        else:
            if geometry[0] == "v":
                theta_0 = sp.Symbol("theta_0")
            elif geometry[0] == "f":
                assert len(np.unique(t_0)) == 1, (
                    "t_0 must contain only a "
                    + "single unique value for geometry[0] == f"
                )

                theta_0 = np.unique(t_0)[0]
            else:
                raise AssertionError("wrong choice of theta_0 geometry")

            if geometry[1] == "v":
                theta_ex = sp.Symbol("theta_ex")
            elif geometry[1] == "f":
                assert len(np.unique(t_ex)) == 1, (
                    "t_ex must contain only"
                    + " a single unique value for geometry[1] == f"
                )

                theta_ex = np.unique(t_ex)[0]
            else:
                raise AssertionError("wrong choice of theta_ex geometry")

            if geometry[2] == "v":
                phi_0 = sp.Symbol("phi_0")
            elif geometry[2] == "f":
                assert len(np.unique(p_0)) == 1, (
                    "p_0 must contain only"
                    + " a single unique value for geometry[2] == f"
                )

                phi_0 = np.unique(p_0)[0]
            else:
                raise AssertionError("wrong choice of phi_0 geometry")

            if geometry[3] == "v":
                phi_ex = sp.Symbol("phi_ex")
            elif geometry[3] == "f":
                assert len(np.unique(p_0)) == 1, (
                    "p_ex must contain only"
                    + " a single unique value for geometry[3] == f"
                )

                phi_ex = np.unique(p_ex)[0]
            else:
                raise AssertionError("wrong choice of phi_ex geometry")

        if geometry[1] == "f":
            dfunc_dtheta_0 = 0.0
        else:
            func = self._func.xreplace(
                {
                    sp.Symbol("theta_0"): theta_0,
                    sp.Symbol("theta_ex"): theta_ex,
                    sp.Symbol("phi_0"): phi_0,
                    sp.Symbol("phi_ex"): phi_ex,
                }
            )

            dfunc_dtheta_0 = sp.diff(func, theta_ex, n)

        if return_symbolic is True:
            return dfunc_dtheta_0
        else:
            args = (
                sp.Symbol("theta_0"),
                sp.Symbol("theta_ex"),
                sp.Symbol("phi_0"),
                sp.Symbol("phi_ex"),
            ) + tuple(param_dict.keys())

            pfunc = sp.lambdify(args, dfunc_dtheta_0, modules=["numpy", "sympy"])

            # in case _func is a constant, lambdify will produce a function
            # with scalar output which is not suitable for further processing
            # (this happens e.g. for the Isotropic brdf).
            # The following query is implemented to ensure correct array-output
            # TODO this is not a proper test !
            if not isinstance(
                pfunc(
                    np.array([0.1, 0.2, 0.3]),
                    0.1,
                    0.1,
                    0.1,
                    **{key: 0.12 for key in param_dict.keys()}
                ),
                np.ndarray,
            ):
                pfunc = np.vectorize(pfunc)

            return pfunc(t_0, t_ex, p_0, p_ex, **param_dict)

    def legexpansion(self, t_0, t_ex, p_0, p_ex, geometry):
        """
        Definition of the legendre-expansion of the
        volume-scattering phase-function

        .. note::
            The output represents the legendre-expansion as needed to compute
            the fn-coefficients for the chosen geometry!
            (http://rt1.readthedocs.io/en/latest/theory.html#equation-fn_coef_definition)

            The incidence-angle argument of the legexpansion() is different to
            the documentation due to the direct definition of the argument as
            the zenith-angle (t_0) instead of the incidence-angle defined in a
            spherical coordinate system (t_i).
            They are related via: t_i = pi - t_0


        Parameters
        ----------
        t_0 : array_like(float)
              array of incident zenith-angles in radians

        p_0 : array_like(float)
              array of incident azimuth-angles in radians

        t_ex : array_like(float)
               array of exit zenith-angles in radians

        p_ex : array_like(float)
               array of exit azimuth-angles in radians

        geometry : str
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
            geometry-parameter, please have a look at the
            "Evaluation Geometries" section of the documentation
            (http://rt1.readthedocs.io/en/latest/model_specification.html#evaluation-geometries)

        Returns
        -------
        sympy-expression
            The legendre-expansion of the volume-scattering phase-function
            for the chosen geometry

        """
        assert self.ncoefs > 0

        theta_s = sp.Symbol("theta_s")
        phi_s = sp.Symbol("phi_s")

        NP = self.ncoefs
        n = sp.Symbol("n")

        # define sympy variables based on chosen geometry
        if geometry == "mono":

            assert len(np.unique(p_0)) == 1, (
                "p_0 must contain only a "
                + "single unique value for monostatic geometry"
            )

            theta_0 = sp.Symbol("theta_0")
            theta_ex = theta_0
            phi_0 = np.unique(p_0)[0]
            phi_ex = np.unique(p_0)[0] + sp.pi
        else:
            if geometry[0] == "v":
                theta_0 = sp.Symbol("theta_0")
            elif geometry[0] == "f":
                assert len(np.unique(t_0)) == 1, (
                    "t_0 must contain only a "
                    + "single unique value for geometry[0] == f"
                )

                theta_0 = np.unique(t_0)[0]
            else:
                raise AssertionError("wrong choice of theta_i geometry")

            if geometry[1] == "v":
                theta_ex = sp.Symbol("theta_ex")
            elif geometry[1] == "f":
                assert len(np.unique(t_ex)) == 1, (
                    "t_ex must contain only"
                    + " a single unique value for geometry[1] == f"
                )

                theta_ex = np.unique(t_ex)[0]
            else:
                raise AssertionError("wrong choice of theta_ex geometry")

            if geometry[2] == "v":
                phi_0 = sp.Symbol("phi_0")
            elif geometry[2] == "f":
                assert len(np.unique(p_0)) == 1, (
                    "p_0 must contain only"
                    + " a single unique value for geometry[2] == f"
                )

                phi_0 = np.unique(p_0)[0]
            else:
                raise AssertionError("wrong choice of phi_i geometry")

            if geometry[3] == "v":
                phi_ex = sp.Symbol("phi_ex")
            elif geometry[3] == "f":
                assert len(np.unique(p_0)) == 1, (
                    "p_ex must contain only"
                    + " a single unique value for geometry[3] == f"
                )

                phi_ex = np.unique(p_ex)[0]
            else:
                raise AssertionError("wrong choice of phi_ex geometry")

        # correct for backscattering
        return sp.Sum(
            self.legcoefs
            * sp.legendre(
                n,
                self.scat_angle(sp.pi - theta_0, theta_s, phi_0, phi_s, self.a),
            ),
            (n, 0, NP - 1),
        )


class LinCombV(Volume):
    """
    Class to generate linear-combinations of volume-class elements

    For details please look at the documentation
    (http://rt1.readthedocs.io/en/latest/model_specification.html#linear-combination-of-scattering-distributions)

    .. note::
        Since the normalization of a volume-scattering phase-function is fixed,
        the weighting-factors must equate to 1!

    Parameters
    ----------
    tau : scalar(float)
          Optical depth of the combined phase-function

          ATTENTION: tau-values provided within the Vchoices-list will not
          be considered!

    omega : scalar(float)
            Single scattering albedo of the combined phase-function

            ATTENTION: omega-values provided within the Vchoices-list will not
            be considered!

    Vchoices : [ [float, Volume]  ,  [float, Volume]  ,  ...]
               a list that contains the the individual phase-functions
               (Volume-objects) and the associated weighting-factors
               (floats) of the linear-combination.
    """

    def __init__(self, Vchoices=None, **kwargs):
        super(LinCombV, self).__init__(**kwargs)

        self.Vchoices = Vchoices
        self._set_function()
        self._set_legexpansion()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""

        self._func = self._Vcombiner()._func

    def _set_legexpansion(self):
        """set legexpansion to the combined legexpansion"""

        self.ncoefs = self._Vcombiner().ncoefs
        self.legexpansion = self._Vcombiner().legexpansion

    def _Vcombiner(self):
        """
        Returns a Volume-class element based on an input-array of Volume-class
        elements. The array must be shaped in the form:
            Vchoices = [  [ weighting-factor   ,   Volume-class element ]  ,
                          [ weighting-factor   ,   Volume-class element ]  ,
                          ...]

        In order to keep the normalization of the phase-functions correct,
        the sum of the weighting factors must equate to 1!

        Attention
        ---------
            the .legexpansion()-function of the combined volume-class
            element is no longer related to its legcoefs (which are set to 0.)
            since the individual legexpansions of the combined volume-class
            elements are possibly evaluated with a different a-parameter
            of the generalized scattering angle! This does not affect any
            calculations, since the evaluation is exclusively based on the
            use of the .legexpansion()-function.
        """

        class Phasefunction(Volume):
            """
            dummy-Volume-class object used to generate linear-combinations
            of volume-phase-functions
            """

            def __init__(self, **kwargs):
                super(Phasefunction, self).__init__(**kwargs)
                self._set_function()
                self._set_legcoefficients()

            def _set_function(self):
                """def phase function as sympy object for later evaluation"""

                self._func = 0.0

            def _set_legcoefficients(self):
                """
                set Legrende coefficients
                needs to be a function that can be later evaluated by
                subsituting 'n'
                """

                self.legcoefs = 0.0

        # find phase functions with equal a parameters
        equals = [
            np.where(
                (np.array([VV[1].a for VV in self.Vchoices]) == tuple(V[1].a)).all(
                    axis=1
                )
            )[0]
            for V in self.Vchoices
        ]

        # evaluate index of phase-functions that have equal a parameter
        equal_a = list({tuple(row) for row in equals})

        # initialize a combined phase-function class element

        # set tau and omega to the values for the combined phase-function
        Vcomb = Phasefunction(tau=self.tau, omega=self.omega)
        # set ncoefs of the combined volume-class element to the maximum
        Vcomb.ncoefs = max([V[1].ncoefs for V in self.Vchoices])
        #   number of coefficients within the chosen functions.
        #   (this is necessary for correct evaluation of fn-coefficients)

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0, len(equal_a)):

            Vdummy = Phasefunction()
            # select V choices where a parameter is equal
            Vequal = np.take(self.Vchoices, equal_a[i], axis=0)
            # set ncoefs to the maximum number within the choices with
            # equal a-parameter
            Vdummy.ncoefs = max([V[1].ncoefs for V in Vequal])
            # loop over phase-functions with equal a-parameter
            for V in Vequal:

                # set parameters based on chosen phase-functions and evaluate
                # combined legendre-expansion
                Vdummy.a = V[1].a
                Vdummy._func = Vdummy._func + V[1]._func * V[0]
                Vdummy.legcoefs = Vdummy.legcoefs + V[1].legcoefs * V[0]

            dummylegexpansion = dummylegexpansion + [Vdummy.legexpansion]

        # combine legendre-expansions for each a-parameter based on given
        # combined legendre-coefficients
        Vcomb.legexpansion = lambda t_0, t_ex, p_0, p_ex, geometry: np.sum(
            [lexp(t_0, t_ex, p_0, p_ex, geometry) for lexp in dummylegexpansion]
        )

        for V in self.Vchoices:
            # set parameters based on chosen classes to define analytic
            # function representation
            Vcomb._func = Vcomb._func + V[1]._func * V[0]

        return Vcomb


class Rayleigh(Volume):
    """
    Define a Rayleigh scattering function

    Parameters
    ----------
    tau : scalar(float)
          Optical depth

    omega : scalar(float)
            Single scattering albedo

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    def __init__(self, **kwargs):
        super(Rayleigh, self).__init__(**kwargs)
        self._set_function()
        self._set_legcoefficients()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        self._func = 3.0 / (16.0 * sp.pi) * (1.0 + x ** 2.0)

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        # only 3 coefficients are needed to correctly represent
        # the Rayleigh scattering function
        self.ncoefs = 3
        n = sp.Symbol("n")
        self.legcoefs = (
            (3.0 / (16.0 * sp.pi))
            * (
                (4.0 / 3.0) * sp.KroneckerDelta(0, n)
                + (2.0 / 3.0) * sp.KroneckerDelta(2, n)
            )
        ).expand()


class HenyeyGreenstein(Volume):
    """
    Define a HenyeyGreenstein scattering function

    Parameters
    ----------
    tau : scalar(float)
          Optical depth

    omega : scalar(float)
            Single scattering albedo

    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein phase function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    def __init__(self, t=None, ncoefs=None, a=[-1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeffs needs to be specified"
        super(HenyeyGreenstein, self).__init__(**kwargs)
        self.t = t
        self.a = a
        assert isinstance(self.a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )
        self.ncoefs = ncoefs
        assert self.ncoefs > 0
        self._set_function()
        self._set_legcoefficients()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        self._func = (1.0 - self.t ** 2.0) / (
            (4.0 * sp.pi) * (1.0 + self.t ** 2.0 - 2.0 * self.t * x) ** 1.5
        )

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        n = sp.Symbol("n")
        self.legcoefs = (1.0 / (4.0 * sp.pi)) * (2.0 * n + 1) * self.t ** n


class HGRayleigh(Volume):
    """
    Define a HenyeyGreenstein-Rayleigh scattering function as proposed in:

        'Quanhua Liu and Fuzhong Weng: Combined henyey-greenstein and
        rayleigh phase function,
        Appl. Opt., 45(28):7475-7479, Oct 2006. doi: 10.1364/AO.45.'

    Parameters
    ----------
    tau : scalar(float)
          Optical depth

    omega : scalar(float)
            Single scattering albedo

    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein-Rayleigh phase function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [-1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)
    """

    def __init__(self, t=None, ncoefs=None, a=[-1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeffs needs to be specified"
        super(HGRayleigh, self).__init__(**kwargs)
        self.t = t
        self.a = a
        assert isinstance(self.a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )
        assert all(type(x) == float for x in a), (
            "Error: Generalization-"
            + "parameter array must contain only floating-point values!"
        )
        self.ncoefs = ncoefs
        assert self.ncoefs > 0
        self._set_function()
        self._set_legcoefficients()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, self.a)
        self._func = (
            3.0
            / (8.0 * sp.pi)
            * (
                1.0
                / (2.0 + self.t ** 2)
                * (1 + x ** 2)
                * (1.0 - self.t ** 2.0)
                / ((1.0 + self.t ** 2.0 - 2.0 * self.t * x) ** 1.5)
            )
        )

    def _set_legcoefficients(self):
        """
        set Legrende coefficients
        needs to be a function that can be later evaluated by subsituting 'n'
        """
        n = sp.Symbol("n")
        self.legcoefs = sp.Piecewise(
            (
                3.0
                / (8.0 * sp.pi)
                * 1.0
                / (2.0 + self.t ** 2)
                * (
                    (n + 2.0) * (n + 1.0) / (2.0 * n + 3) * self.t ** (n + 2.0)
                    + (n + 1.0) ** 2.0 / (2.0 * n + 3.0) * self.t ** n
                    + (5.0 * n ** 2.0 - 1.0) / (2.0 * n - 1.0) * self.t ** n
                ),
                n < 2,
            ),
            (
                3.0
                / (8.0 * sp.pi)
                * 1.0
                / (2.0 + self.t ** 2)
                * (
                    n * (n - 1.0) / (2.0 * n - 1.0) * self.t ** (n - 2.0)
                    + (n + 2.0) * (n + 1.0) / (2.0 * n + 3) * self.t ** (n + 2.0)
                    + (n + 1.0) ** 2.0 / (2.0 * n + 3.0) * self.t ** n
                    + (5.0 * n ** 2.0 - 1.0) / (2.0 * n - 1.0) * self.t ** n
                ),
                True,
            ),
        )

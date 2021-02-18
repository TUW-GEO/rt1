"""Definition of BRDF functions"""

import numpy as np
import sympy as sp
from functools import partial, update_wrapper

from .scatter import Scatter
from .rtplots import polarplot, hemreflect


class Surface(Scatter):
    """basic surface class"""

    def __init__(self, **kwargs):
        # set scattering angle generalization-matrix to [1,1,1] if it is not
        # explicitly provided by the chosen class.
        # this results in a peak in specular-direction which is suitable
        # for describing surface BRDF's
        self.a = getattr(self, "a", [1.0, 1.0, 1.0])

        self.NormBRDF = kwargs.pop("NormBRDF", 1.0)
        # quick way for visualizing the functions as polarplot
        self.polarplot = partial(polarplot, X=self)
        update_wrapper(self.polarplot, polarplot)
        # quick way for visualizing the associated hemispherical reflectance
        self.hemreflect = partial(hemreflect, SRF=self)
        update_wrapper(self.hemreflect, hemreflect)

    def brdf(self, t_0, t_ex, p_0, p_ex, param_dict={}):
        """
        Calculate numerical value of the BRDF for chosen
        incidence- and exit angles.

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
                          Numerical value of the BRDF
        """

        # define sympy objects
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        # replace arguments and evaluate expression
        # sp.lambdify is used to allow array-inputs
        # for python >3.5 unpacking could be used, i.e.:
        # brdffunc = sp.lambdify((theta_0, theta_ex, phi_0, phi_ex,
        #                        *param_dict.keys()),
        #                       self._func, modules=["numpy", "sympy"])
        args = (theta_0, theta_ex, phi_0, phi_ex) + tuple(param_dict.keys())
        brdffunc = sp.lambdify(args, self._func, modules=["numpy", "sympy"])

        # in case _func is a constant, lambdify will produce a function with
        # scalar output which is not suitable for further processing
        # (this happens e.g. for the Isotropic brdf).
        # The following query is implemented to ensure correct array-output:
        # TODO this is not a proper test !
        if not isinstance(
            brdffunc(
                np.array([0.1, 0.2, 0.3]),
                0.1,
                0.1,
                0.1,
                **{key: 0.12 for key in param_dict.keys()}
            ),
            np.ndarray,
        ):
            brdffunc = np.vectorize(brdffunc)

        return brdffunc(t_0, t_ex, p_0, p_ex, **param_dict)

    def legexpansion(self, t_0, t_ex, p_0, p_ex, geometry):
        """
        Definition of the legendre-expansion of the BRDF

        .. note::
            The output represents the legendre-expansion as needed to
            compute the fn-coefficients for the chosen geometry!
            (http://rt1.readthedocs.io/en/latest/theory.html#equation-fn_coef_definition)

            The incidence-angle argument of the legexpansion() is different
            to the documentation due to the direct definition of the argument
            as the zenith-angle (t_0) instead of the incidence-angle
            defined in a spherical coordinate system (t_i).
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
        sympy - expression
            The legendre - expansion of the BRDF for the chosen geometry

        """
        assert self.ncoefs > 0

        theta_s = sp.Symbol("theta_s")
        phi_s = sp.Symbol("phi_s")

        NBRDF = self.ncoefs
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

        return sp.Sum(
            self.legcoefs
            * sp.legendre(n, self.scat_angle(theta_s, theta_ex, phi_s, phi_ex, self.a)),
            (n, 0, NBRDF - 1),
        )

    def brdf_theta_diff(
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
        Calculation of the derivative of the BRDF with respect to
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

            brdffunc = sp.lambdify(args, dfunc_dtheta_0, modules=["numpy", "sympy"])

            # in case _func is a constant, lambdify will produce a function
            # with scalar output which is not suitable for further processing
            # (this happens e.g. for the Isotropic brdf).
            # The following query is implemented to ensure correct array-output
            # TODO this is not a proper test !
            if not isinstance(
                brdffunc(
                    np.array([0.1, 0.2, 0.3]),
                    0.1,
                    0.1,
                    0.1,
                    **{key: 0.12 for key in param_dict.keys()}
                ),
                np.ndarray,
            ):
                brdffunc = np.vectorize(brdffunc)

            return brdffunc(t_0, t_ex, p_0, p_ex, **param_dict)


class LinCombSRF(Surface):
    """
    Class to generate linear-combinations of volume-class elements

    For details please look at the documentation
    (http://rt1.readthedocs.io/en/latest/model_specification.html#linear-combination-of-scattering-distributions)

    Parameters
    ----------
    SRFchoices : [ [float, Surface]  ,  [float, Surface]  ,  ...]
                 A list that contains the the individual BRDF's
                 (Surface-objects) and the associated weighting-factors
                 (floats) for the linear-combination.

    NormBRDf : scalar(float)
               Hemispherical reflectance of the combined BRDF

               ATTENTION: NormBRDF-values provided within the SRFchoices-list
               will not be considered!
    """

    def __init__(self, SRFchoices=None, **kwargs):

        super(LinCombSRF, self).__init__(**kwargs)

        self.SRFchoices = SRFchoices
        self._set_function()
        self._set_legexpansion()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""

        self._func = self._SRFcombiner()._func

    def _set_legexpansion(self):
        """set legexpansion to the combined legexpansion"""

        self.ncoefs = self._SRFcombiner().ncoefs
        self.legexpansion = self._SRFcombiner().legexpansion

    def _SRFcombiner(self):
        """
        Returns a Surface-class element based on an input-array of
        Surface-class elements.
        The array must be shaped in the form:
            SRFchoices = [  [ weighting-factor   ,   Surface-class element ],
                            [ weighting-factor   ,   Surface-class element ],
                        ...]

        ATTENTION: the .legexpansion()-function of the combined surface-class
        element is no longer related to its legcoefs (which are set to 0.)
                   since the individual legexpansions of the combined surface-
                   class elements are possibly evaluated with a different
                   a-parameter of the generalized scattering angle! This does
                   not affect any calculations, since the evaluation is
                   only based on the use of the .legexpansion()-function.
        """

        class BRDFfunction(Surface):
            """
            dummy-Surface-class object used to generate
            linear-combinations of BRDF-functions
            """

            def __init__(self, **kwargs):
                super(BRDFfunction, self).__init__(**kwargs)
                self._set_function()
                self._set_legcoefficients()

            def _set_function(self):
                """def phase function as sympy object for later evaluation"""
                self._func = 0.0

            def _set_legcoefficients(self):
                self.legcoefs = 0.0

        # initialize a combined phase-function class element
        SRFcomb = BRDFfunction(NormBRDf=self.NormBRDF)
        # set ncoefs of the combined volume-class element to the maximum
        SRFcomb.ncoefs = max([SRF[1].ncoefs for SRF in self.SRFchoices])
        #   number of coefficients within the chosen functions.
        #   (this is necessary for correct evaluation of fn-coefficients)

        # find BRDF functions with equal a parameters
        equals = [
            np.where(
                (np.array([VV[1].a for VV in self.SRFchoices]) == tuple(V[1].a)).all(
                    axis=1
                )
            )[0]
            for V in self.SRFchoices
        ]

        # evaluate index of BRDF-functions that have equal a parameter

        # find phase functions where a-parameter is equal
        equal_a = list({tuple(row) for row in equals})

        # evaluation of combined expansion in legendre-polynomials
        dummylegexpansion = []
        for i in range(0, len(equal_a)):

            SRFdummy = BRDFfunction()
            # select SRF choices where a parameter is equal
            SRFequal = np.take(self.SRFchoices, equal_a[i], axis=0)
            # set ncoefs to the maximum number within the choices
            # with equal a-parameter
            SRFdummy.ncoefs = max([SRF[1].ncoefs for SRF in SRFequal])
            # loop over phase-functions with equal a-parameter
            for SRF in SRFequal:

                # set parameters based on chosen phase-functions and evaluate
                # combined legendre-expansion
                SRFdummy.a = SRF[1].a
                SRFdummy.NormBRDF = SRF[1].NormBRDF
                SRFdummy._func = SRFdummy._func + SRF[1]._func * SRF[0]
                SRFdummy.legcoefs += SRF[1].legcoefs * SRF[0]

            dummylegexpansion = dummylegexpansion + [SRFdummy.legexpansion]

        # combine legendre-expansions for each a-parameter based on given
        # combined legendre-coefficients
        SRFcomb.legexpansion = lambda t_0, t_ex, p_0, p_ex, geometry: np.sum(
            [lexp(t_0, t_ex, p_0, p_ex, geometry) for lexp in dummylegexpansion]
        )

        for SRF in self.SRFchoices:
            # set parameters based on chosen classes to define analytic
            # function representation
            SRFcomb._func = SRFcomb._func + SRF[1]._func * SRF[0]
        return SRFcomb


class Isotropic(Surface):
    """
    Define an isotropic surface brdf

    Parameters
    ----------
    NormBRDF : float, optional (default = 1.)
               Normalization-factor used to scale the BRDF,
               i.e.  BRDF = NormBRDF * f(t_0,p_0,t_ex,p_ex)
    """

    def __init__(self, **kwargs):
        super(Isotropic, self).__init__(**kwargs)
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        self.ncoefs = 1
        n = sp.Symbol("n")
        self.legcoefs = (1.0 / sp.pi) * sp.KroneckerDelta(0, n)

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        self._func = 1.0 / sp.pi


class CosineLobe(Surface):
    """
    Define a (possibly generalized) cosine-lobe of power i.

    Parameters
    ----------
    i : scalar(int)
        Power of the cosine lobe, i.e. cos(x)^i

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)

    NormBRDF : float, optional (default = 1.)
               Normalization-factor used to scale the BRDF,
               i.e.  BRDF = NormBRDF * f(t_0,p_0,t_ex,p_ex)
    """

    def __init__(self, ncoefs=None, i=None, a=[1.0, 1.0, 1.0], **kwargs):
        assert ncoefs is not None, (
            "Error: number of coefficients " + "needs to be provided!"
        )
        assert i is not None, "Error: Cosine lobe power needs to be specified!"
        super(CosineLobe, self).__init__(**kwargs)
        assert ncoefs > 0
        self.i = i
        assert isinstance(self.i, int), (
            "Error: Cosine lobe power needs " + "to be an integer!"
        )
        assert i >= 0, "ERROR: Power of Cosine-Lobe needs to be greater than 0"
        self.a = a
        assert isinstance(self.a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )
        assert all(type(x) == float for x in a), (
            "Error: Generalization-"
            + "parameter array must "
            + "contain only floating-"
            + "point values!"
        )
        self.ncoefs = int(ncoefs)
        self._set_function()
        self._set_legcoefficients()

    def _set_legcoefficients(self):
        n = sp.Symbol("n")
        # A13   The Rational(is needed as otherwise a Gamma function
        # Pole error is issued)
        self.legcoefs = (
            1.0
            / sp.pi
            * (
                (
                    2 ** (-2 - self.i)
                    * (1 + 2 * n)
                    * sp.sqrt(sp.pi)
                    * sp.gamma(1 + self.i)
                )
                / (
                    sp.gamma((2 - n + self.i) * sp.Rational(1, 2))
                    * sp.gamma((3 + n + self.i) * sp.Rational(1, 2))
                )
            )
        )

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        # self._func = sp.Max(self.scat_angle(theta_i,
        #                                    theta_s,
        #                                    phi_i,
        #                                    phi_s,
        #                                    a=self.a), 0.)**self.i  # eq. A13

        # alternative formulation avoiding the use of sp.Max()
        #     (this is done because   sp.lambdify('x',sp.Max(x), "numpy")
        #      generates a function that can not interpret array inputs.)
        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, a=self.a)
        self._func = 1.0 / sp.pi * (x * (1.0 + sp.sign(x)) / 2.0) ** self.i


class HenyeyGreenstein(Surface):
    """
    Define a HenyeyGreenstein scattering function for use as BRDF
    approximation function.

    Parameters
    ----------
    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)

    NormBRDF : float, optional (default = 1.)
               Normalization-factor used to scale the BRDF,
               i.e.  BRDF = NormBRDF * f(t_0,p_0,t_ex,p_ex)
    """

    def __init__(self, t=None, ncoefs=None, a=[1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeff. needs to be specified"
        super(HenyeyGreenstein, self).__init__(**kwargs)
        self.t = t
        self.ncoefs = ncoefs
        assert self.ncoefs > 0

        self.a = a
        assert isinstance(self.a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )
        self._set_function()
        self._set_legcoefficients()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, a=self.a)

        self._func = (
            1.0
            * (1.0 - self.t ** 2.0)
            / ((sp.pi) * (1.0 + self.t ** 2.0 - 2.0 * self.t * x) ** 1.5)
        )

    def _set_legcoefficients(self):
        n = sp.Symbol("n")
        self.legcoefs = 1.0 * (1.0 / (sp.pi)) * (2.0 * n + 1) * self.t ** n


class HG_nadirnorm(Surface):
    """
    Define a HenyeyGreenstein scattering function for use as BRDF
    approximation function.

    Parameters
    ----------
    t : scalar(float)
        Asymmetry parameter of the Henyey-Greenstein function

    ncoefs : scalar(int)
             Number of coefficients used within the Legendre-approximation

    a : [ float , float , float ] , optional (default = [1.,1.,1.])
        generalized scattering angle parameters used for defining the
        scat_angle() of the BRDF
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)

    NormBRDF : float, optional (default = 1.)
               Normalization-factor used to scale the BRDF,
               i.e.  BRDF = NormBRDF * f(t_0,p_0,t_ex,p_ex)
    """

    def __init__(self, t=None, ncoefs=None, a=[1.0, 1.0, 1.0], **kwargs):
        assert t is not None, "t parameter needs to be provided!"
        assert ncoefs is not None, "Number of coeffs needs to be specified"
        super(HG_nadirnorm, self).__init__(**kwargs)
        self.t = t
        self.ncoefs = ncoefs
        assert self.ncoefs > 0

        self.a = a
        assert isinstance(self.a, list), (
            "Error: Generalization-parameter " + "needs to be a list"
        )
        assert len(a) == 3, (
            "Error: Generalization-parameter list must " + "contain 3 values"
        )
        self._set_function()
        self._set_legcoefficients()

    def _set_function(self):
        """define phase function as sympy object for later evaluation"""
        theta_0 = sp.Symbol("theta_0")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_ex = sp.Symbol("phi_ex")

        x = self.scat_angle(theta_0, theta_ex, phi_0, phi_ex, a=self.a)

        nadir_hemreflect = 4 * (
            (1.0 - self.t ** 2.0)
            * (
                1.0
                - self.t * (-self.t + self.a[0])
                - sp.sqrt(
                    (1 + self.t ** 2 - 2 * self.a[0] * self.t) * (1 + self.t ** 2)
                )
            )
            / (
                2.0
                * self.a[0] ** 2.0
                * self.t ** 2.0
                * sp.sqrt(1.0 + self.t ** 2.0 - 2.0 * self.a[0] * self.t)
            )
        )

        self._func = (1.0 / nadir_hemreflect) * (
            (1.0 - self.t ** 2.0)
            / ((sp.pi) * (1.0 + self.t ** 2.0 - 2.0 * self.t * x) ** 1.5)
        )

    def _set_legcoefficients(self):
        nadir_hemreflect = 4 * (
            (1.0 - self.t ** 2.0)
            * (
                1.0
                - self.t * (-self.t + self.a[0])
                - sp.sqrt(
                    (1 + self.t ** 2 - 2 * self.a[0] * self.t) * (1 + self.t ** 2)
                )
            )
            / (
                2.0
                * self.a[0] ** 2.0
                * self.t ** 2.0
                * sp.sqrt(1.0 + self.t ** 2.0 - 2.0 * self.a[0] * self.t)
            )
        )

        n = sp.Symbol("n")
        self.legcoefs = (1.0 / nadir_hemreflect) * (
            (1.0 / (sp.pi)) * (2.0 * n + 1) * self.t ** n
        )

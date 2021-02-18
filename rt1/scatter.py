"""Define general object for scattering calculations"""

# general other imports
import sympy as sp


class Scatter(object):
    """The basis object for any Surface and Volume objects"""

    def __init__(self):
        pass

    def scat_angle(self, t_0, t_ex, p_0, p_ex, a):
        """
        Function to return the generalized scattering angle with respect to the
        given zenith-angles
        (http://rt1.readthedocs.io/en/latest/theory.html#equation-general_scat_angle)

        Standard-choices assigned in the volume- and surface class:

        - Surface: a=[ 1,1,1] ... pi-shifted scattering angle
          cos[t_0]*cos[t_ex] + sin[t_0]*sin[t_ex]*cos[p_0 - p_ex]
        - Volume:  a=[-1,1,1] ... ordinary scattering angle
          -cos[t_0]*cos[t_ex] + sin[t_0]*sin[t_ex]*cos[p_0 - p_ex]

        .. note::
            The scattering angle is defined with respect to the incident
            zenith-angle t_0, and not with respect to the incidence-angle in
            a spherical coordinate system (t_i)! The two definitions are
            related via t_i = pi - t_0

        Parameters
        ----------
        t_0 : array_like(float)
              incident zenith-angle
        p_0 : array_like(float)
              incident azimuth-angle
        t_ex : array_like(float)
               exit zenith-angle
        p_ex : array_like(float)
               exit azimuth-angle
        a : [ float , float , float ]
            generalized scattering angle parameters

        Returns
        -------
        float
              the generalized scattering angle

        """
        return (
            a[0] * sp.cos(t_0) * sp.cos(t_ex)
            + a[1] * sp.sin(t_0) * sp.sin(t_ex) * sp.cos(p_0) * sp.cos(p_ex)
            + a[2] * sp.sin(t_0) * sp.sin(t_ex) * sp.sin(p_0) * sp.sin(p_ex)
        )

    def _get_legcoef(self, n0):
        """
        function to evaluate legendre coefficients
        used mainly for testing purposes
        the actual coefficient are used in the symbolic expansion
        """
        n = sp.Symbol("n")
        return self.legcoefs.xreplace({n: int(n0)}).evalf()

    def _eval_legpoly(self, t_0, t_s, p_0, p_s, geometry=None):
        """
        function to evaluate legendre coefficients based on expansion
        used mainly for testing purposes
        the actual coefficient are used in the symbolic expansion
        """

        assert geometry is not None, "Geometry needs to be specified!"

        theta_0 = sp.Symbol("theta_0")
        theta_s = sp.Symbol("theta_s")
        theta_ex = sp.Symbol("theta_ex")
        phi_0 = sp.Symbol("phi_0")
        phi_s = sp.Symbol("phi_s")
        phi_ex = sp.Symbol("phi_ex")

        res = self.legexpansion(t_0, t_s, p_0, p_s, geometry).xreplace(
            {
                theta_0: t_0,
                theta_s: t_s,
                phi_0: p_0,
                phi_s: p_s,
                theta_ex: t_s,
                phi_ex: p_s,
            }
        )
        return res.evalf()

"""
Core module for 1st order scattering simulations using arbitrary BRDF and phase functions

References
----------
Quast & Wagner (2016): doi:10.1364/AO.55.005379

"""

###todo implement unittests

import numpy as np
from surface import Isotropic
from scipy.special import expi
from scipy.special import expn



class RT1(object):
    """
    main class to perform RT simulations
    """
    def __init__(self, I0, mu_0, mu_ex, phi_0, phi_ex, RV=None, SRF=None, nmax=10, Fn=None):
        """
        Parameters
        ----------
        I0 : float
            incidence radiation
        RV : Volume
            random volume object
        """
        self.I0 = I0
        self.mu_0 = mu_0
        self.mu_ex = mu_ex
        self.phi_0 = phi_0
        self.phi_ex = phi_ex

        self.RV = RV
        assert self.RV is not None, 'ERROR: needs to provide volume information'

        self.SRF = SRF
        assert self.SRF is not None, 'ERROR: needs to provide surface information'

        self._nmax = nmax
        self.Fn = Fn
        assert self.Fn is not None, 'ERROR: an object handling the coefficients needs to be provided'

    def cos_theta(self, mu_i, mu_s, phi_i, phi_s):
        """
        A14

        Parameters
        ----------
        mu_i : float
            cosine of incidence angle
        mu_s : float
            cosine of scattering angle
        phi_i : float
            incident azimuth angle [rad]
        phi_s : float
            scattering azimuth angle [rad]
        """
        ctheta = mu_i*mu_s + np.sin(np.arccos(mu_i))*np.sin(np.arccos(mu_s))*np.cos(phi_i-phi_s)
        return ctheta

    def cos_theta_prime(self, mu_i, mu_s, phi_i, phi_s):
        """
        A15
        """
        ctheta_prime = -mu_i*mu_s + np.sin(np.arccos(mu_i))*np.sin(np.arccos(mu_s))*np.cos(phi_i-phi_s)
        return ctheta_prime


    def calc(self):
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
        ctheta = self.cos_theta(-self.mu_0, self.mu_ex, self.phi_0, self.phi_ex)
        return self.I0 * np.exp(-(self.RV.tau / self.mu_0) - (self.RV.tau/self.mu_ex)) * self.mu_0 * self.SRF.brdf(ctheta)

    def volume(self):
        """
        (18)
        """
        # todo ctheta Winkel Definition ???
        ctheta = self.cos_theta(-self.mu_0, self.mu_ex, self.phi_0, self.phi_ex)
        return (self.I0*self.RV.omega*self.mu_0/(self.mu_0+self.mu_ex)) * (1.-np.exp(-(self.RV.tau/self.mu_0)-(self.RV.tau/self.mu_ex))) * self.RV.p(ctheta)

    def interaction(self):
        """
        (19)
        """
        Fint1 = self._calc_Fint(self.mu_0, self.mu_ex)
        Fint2 = self._calc_Fint(self.mu_ex, self.mu_0)
        return self.I0 * self.mu_0 * self.RV.omega * (np.exp(-self.RV.tau/self.mu_ex) * Fint1 + np.exp(-self.RV.tau/self.mu_0)*Fint2 )

    def _calc_Fint(self, mu1, mu2):
        """
        (37)
        """

        S = 0.
        fn, nmax_trunc = self.Fn.fn(mu1, self._nmax)

        # set maximum number of itterations either to default value or to the value were all subsequent coefficeints are only zero
        nmax = min(nmax_trunc, self._nmax)

        for n in xrange(nmax):
            S2 = 0.
            for k in xrange(1,(n+1)+1):
                E_k1 = expn(k+1., self.RV.tau)
                S2 += mu1**(-k) * (E_k1 - np.exp(-self.RV.tau/mu1)/k)


            # final sum
            # todo check once more the function
            S += fn[n] * mu1**(n+1) * (np.exp(-self.RV.tau/mu1)*np.log(mu1/(1.-mu1)) - expi(-self.RV.tau) + np.exp(-self.RV.tau/mu1)*expi(self.RV.tau/mu1-self.RV.tau) + S2)

        return S







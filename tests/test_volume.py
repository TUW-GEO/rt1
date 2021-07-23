# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.volume import Volume
from rt1.volume import Rayleigh, HenyeyGreenstein, HGRayleigh, LinCombV
import matplotlib.pyplot as plt
from scipy.integrate import simps
from itertools import product


class TestVolume(unittest.TestCase):

    def test_init(self):
        V = Volume(omega=0.2, tau=1.7)
        init_dict = V.init_dict

        self.assertEqual(V.omega, 0.2)
        self.assertEqual(V.tau, 1.7)

    def test_rayleigh(self):
        V = Rayleigh(omega=0.2, tau=1.7)
        t_0 = np.pi/2.
        t_ex = 0.234234
        p_0 = np.pi/2.
        p_ex = 0.

        p = V.p(t_0, t_ex, p_0, p_ex)   # cos=0

        self.assertEqual(p, (3./(16.*np.pi)))

    def test_HenyeyGreenstein(self):
        V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        init_dict = V.init_dict

        self.assertEqual(V.t, 0.7)
        t_0 = np.pi/2.
        t_ex = 0.234234
        p_0 = np.pi/2.
        p_ex = 0.

        # --> cos(THETA) = 0
        p = V.p(t_0, t_ex, p_0, p_ex)
        self.assertAlmostEqual(
            p, (1. - 0.7 * 0.7) / (4. * np.pi * (1. + 0.7 * 0.7) ** 1.5), 10
            )

    def test_HGRayleigh(self):
        V = HGRayleigh(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        init_dict = V.init_dict
        self.assertEqual(V.t, 0.7)
        t_0 = np.pi/2.
        t_ex = 0.234234
        p_0 = np.pi/2.
        p_ex = 0.

        Vr = Rayleigh(omega=0.2, tau=1.7)
        init_dict = Vr.init_dict
        Vhg = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        init_dict = Vhg.init_dict

        p = V.p(t_0, t_ex, p_0, p_ex)
        phg = Vhg.p(t_0, t_ex, p_0, p_ex)
        pr = Vr.p(t_0, t_ex, p_0, p_ex)

        self.assertAlmostEqual(p,
                               4. * np.pi * (4 / (2 + .7 ** 2)) * 1 / 2 * phg * pr,
                               10
                               )

    def test_HenyeyGreenstein_coefficients(self):
        V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        init_dict = V.init_dict
        self.assertEqual(V._get_legcoef(0), 1./(4.*np.pi))
        self.assertEqual(V._get_legcoef(1), 3. * 0.7 / (4. * np.pi))

    def test_polarplot_V(self):
        V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        init_dict = V.init_dict
        pl = V.polarplot()
        plt.close(pl)

    def test_V_normalization(self):

        V1 = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        V2 = HGRayleigh(omega=0.2, tau=1.7, t=0.7, ncoefs=20)
        V3 = Rayleigh(omega=0.2, tau=1.7)

        V4 = LinCombV([[.5, Rayleigh()],
                      [.5, HenyeyGreenstein(t=0.7, ncoefs=20)]],
                      omega=0.2, tau=1.7)

        for V in [V1, V2, V3, V4]:
            init_dict = V.init_dict
            # set incident (zenith-angle) directions for which the integral
            # should be evaluated!
            incnum = np.linspace(0, np.pi, 5)
            pincnum = np.linspace(0, 2*np.pi, 5)

            # define grid for integration
            x = np.linspace(0., np.pi, 500)
            y = np.linspace(0., 2 * np.pi, 500)

            # initialize array for solutions
            sol = []
            # ---- evaluation of Integral
            # adapted from
            # (http://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy)
            for theta_0, phi_0 in product(incnum, pincnum):
                # define the function that has to be integrated
                # (i.e. Eq.20 in the paper)
                # notice the additional  np.sin(thetas)  which oritinates from
                # integrating over theta_s instead of mu_s
                def integfunkt(theta_s, phi_s):
                    return np.sin(theta_s) * V.p(theta_0, theta_s,
                                                 phi_0, phi_s)
                # evaluate the integral using Simpson's Rule twice
                z = integfunkt(x[:, None], y)
                sol += [simps(simps(z, y), x)]

                self.assertTrue(np.allclose(sol, 1.))

    def test_LinCombV(self):
        Vr = Rayleigh(omega=0.2, tau=1.7)
        Vhg = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7,ncoefs=20)

        V = LinCombV([[.5, Vr],
                      [.5, Vhg]], omega=0.2, tau=1.7)

        init_dict = V.init_dict

        Vr = Rayleigh(omega=0.2, tau=1.7)
        Vhg = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7, ncoefs=20)

        incnum = np.linspace(0, np.pi, 3)
        pincnum = np.linspace(0, 2*np.pi, 3)

        for t_0, t_ex, p_0, p_ex in product(incnum, pincnum,
                                            incnum, pincnum):

            p = V.p(t_0, t_ex, p_0, p_ex)
            phg = Vhg.p(t_0, t_ex, p_0, p_ex)
            pr = Vr.p(t_0, t_ex, p_0, p_ex)

            self.assertAlmostEqual(p, .5 * (phg + pr), 10)


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.volume import Volume
from rt1.volume import Rayleigh,HenyeyGreenstein



class TestVolume(unittest.TestCase):

    def test_init(self):
        V = Volume(omega=0.2, tau=1.7)
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
        V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7,ncoefs=20)
        self.assertEqual(V.t,0.7)
        t_0 = np.pi/2.
        t_ex = 0.234234
        p_0 = np.pi/2.
        p_ex = 0.

        #--> cos(THETA) = 0
        p = V.p(t_0, t_ex, p_0, p_ex)
        self.assertAlmostEqual(p, (1.-0.7*0.7)/(4.*np.pi*(1.+0.7*0.7)**1.5),10)

    def test_HenyeyGreenstein_coefficients(self):
        V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7,ncoefs=20)
        self.assertEqual(V._get_legcoef(0),1./(4.*np.pi))
        self.assertEqual(V._get_legcoef(1),3.*0.7/(4.*np.pi))









if __name__ == "__main__":
    unittest.main()



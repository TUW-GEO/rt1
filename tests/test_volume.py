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
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        p = V.p(theta_i, theta_s, phi_i, phi_s)   # cos=0

        self.assertEqual(p, (3./(16.*np.pi)))

    def test_HenyeyGreenstein(self):
        V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7,ncoefs=20)
        self.assertEqual(V.t,0.7)
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        #--> cos(THETA) = 0
        p = V.p(theta_i, theta_s, phi_i, phi_s)
        self.assertAlmostEqual(p, (1.-0.7*0.7)/(4.*np.pi*(1.+0.7*0.7)**1.5),10)








if __name__ == "__main__":
    unittest.main()



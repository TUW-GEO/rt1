# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.volume import Volume
from rt1.volume import Rayleigh



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

        #~ p = V.p(theta_i, theta_s, phi_i, phi_s)  # cos = 1
        #~ self.assertEqual(p, (3./(8.*np.pi)))

if __name__ == "__main__":
    unittest.main()



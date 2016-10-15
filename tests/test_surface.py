# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.surface import Isotropic


class TestSurface(unittest.TestCase):

    def test_isotropic(self):
        S = Isotropic()
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        self.assertEqual(S.brdf(theta_i, theta_s, phi_i, phi_s), 1./np.pi)
        self.assertEqual(S.brdf(theta_i, theta_s, phi_i, phi_s), 1./np.pi)
        self.assertEqual(S.brdf(theta_i, theta_s, phi_i, phi_s), 1./np.pi)


if __name__ == "__main__":
    unittest.main()



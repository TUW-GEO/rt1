# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.coefficients import RayleighIsotropic


class TestAR1(unittest.TestCase):

    def test_RayleighIsotropic(self):
        C = RayleighIsotropic()

        mu_i = 0.
        n = 10
        self.assertEqual(C.fn(mu_i, n)[0], 9./(16.*np.pi))
        self.assertEqual(C.fn(mu_i, n)[1], 0.)
        self.assertEqual(C.fn(mu_i, n)[2], -3./(16.*np.pi))
        self.assertEqual(C.fn(mu_i, n)[3], 0.)
        self.assertEqual(C.fn(mu_i, n)[8], 0.)

        mu_i = 1.
        n = 10
        self.assertEqual(C.fn(mu_i, n)[0], 3./(8.*np.pi))
        self.assertEqual(C.fn(mu_i, n)[1], 0.)
        self.assertEqual(C.fn(mu_i, n)[2], 3./(8.*np.pi))
        self.assertEqual(C.fn(mu_i, n)[3], 0.)
        self.assertEqual(C.fn(mu_i, n)[8], 0.)


if __name__ == "__main__":
    unittest.main()



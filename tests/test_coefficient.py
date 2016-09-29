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


        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[0], 9./(16.*np.pi))
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[1], 0.)
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[2], -3./(16.*np.pi))
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[3], 0.)
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[8], 0.)

        mu_i = 1.
        n = 10
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[0], 3./(8.*np.pi))
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[1], 0.)
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[2], 3./(8.*np.pi))
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[3], 0.)
        res, nmax = C.fn(mu_i, n)
        self.assertEqual(res[8], 0.)


    def test_nmax(self):
        C = RayleighIsotropic()

        x = np.asarray([1.,2.,3.,4.]).astype('float')
        n = C._get_nmax(x)
        self.assertEqual(n, 3)

        x = np.asarray([0.,1.,0.,0.]).astype('float')
        n = C._get_nmax(x)
        self.assertEqual(n, 1)

        x = np.asarray([0.,0.,0.,4.]).astype('float')
        n = C._get_nmax(x)
        self.assertEqual(n, 3)

        x = np.asarray([1.,0.,0.,0.]).astype('float')
        n = C._get_nmax(x)
        self.assertEqual(n, 0)




if __name__ == "__main__":
    unittest.main()



# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.surface import Isotropic


class TestAR1(unittest.TestCase):

    def test_isotropic(self):
        S = Isotropic()
        self.assertEqual(S.brdf(10.), 1./np.pi)
        self.assertEqual(S.brdf(0.), 1./np.pi)
        self.assertEqual(S.brdf(1.), 1./np.pi)


if __name__ == "__main__":
    unittest.main()



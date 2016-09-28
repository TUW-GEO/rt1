# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.volume import Volume
from rt1.volume import Rayleigh



class TestAR1(unittest.TestCase):

    def test_init(self):
        V = Volume(omega=0.2, tau=1.7)
        self.assertEqual(V.omega, 0.2)
        self.assertEqual(V.tau, 1.7)

    def test_rayleigh(self):
        V = Rayleigh(omega=0.2, tau=1.7)
        p = V.p(0.)
        self.assertEqual(p, (3./(16.*np.pi)))

        p = V.p(1.)
        self.assertEqual(p, (3./(8.*np.pi)))

if __name__ == "__main__":
    unittest.main()



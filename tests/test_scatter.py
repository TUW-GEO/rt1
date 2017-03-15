# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.scatter import Scatter

class TestScatter(unittest.TestCase):


    def test_p(self):
        S = Scatter()
        theta_0 = np.pi/2.
        theta_s = 0.234234
        phi_0 = np.pi/2.
        phi_s = 0.

        p = S.scat_angle(theta_0, theta_s, phi_0, phi_s, a=[-1.,1.,1.])   # cos(theta)=0
        self.assertAlmostEqual(p, 0.,10)

        theta_0 = 0.
        theta_s = 0.
        phi_0 = np.pi/2.
        phi_s = 0.12345
        p = S.scat_angle(theta_0, theta_s, phi_0, phi_s, a=[-.7,1.,1.])   # cos(theta)=-1
        self.assertAlmostEqual(p, -.7,10)


if __name__ == "__main__":
    unittest.main()




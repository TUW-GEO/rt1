# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.scatter import Scatter
from rt1.volume import Rayleigh



class TestScatter(unittest.TestCase):


    def test_p(self):
        S = Scatter()
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        p = S.scat_angle(theta_i, theta_s, phi_i, phi_s, a=[-1.,1.,1.])   # cos(theta)=0
        self.assertAlmostEqual(p, 0.,10)

        theta_i = 0.
        theta_s = 0.
        phi_i = np.pi/2.
        phi_s = 0.12345
        p = S.scat_angle(theta_i, theta_s, phi_i, phi_s, a=[-.7,1.,1.])   # cos(theta)=-1
        self.assertAlmostEqual(p, -.7,10)





if __name__ == "__main__":
    unittest.main()




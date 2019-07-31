    # -*- coding: utf-8 -*-

"""
test examples given in paper by comparison against reference solution.

the actual comparison is done by checking the equality of the interaction-term
with a numerical solution (generated with numerical_evaluation.py)

"""

import unittest
import numpy as np

import os
import sys

#sys.path.append(os.path.join('..', 'mintrend'))
sys.path.append('..')
from rt1.rt1 import RT1
from rt1.volume import Rayleigh, HenyeyGreenstein
from rt1.surface import CosineLobe


class TestExamples(unittest.TestCase):

    def setUp(self):
        # read reference solutions for backscattering case
        fname1 = os.path.dirname(__file__) + os.sep + 'example1_int.csv'
        x1 = np.loadtxt(fname1, delimiter=',', skiprows=0)
        self.inc1 = x1[:, 0]
        self.int_num_1 = x1[:, 1]

        fname2 = os.path.dirname(__file__) + os.sep + 'example2_int.csv'
        x2 = np.loadtxt(fname2, delimiter=',', skiprows=0)
        self.inc2 = x2[:, 0]
        self.int_num_2 = x2[:, 1]

    def test_example_1_int(self):
        print('Testing Example 1 ...')

        inc = self.inc1

        # ---- evaluation of first example
        V = Rayleigh(tau=np.array([0.7]), omega=np.array([0.3]))
        # 11 instead of 10 coefficients used to assure 7 digit precision
        SRF = CosineLobe(ncoefs=11, i=5, NormBRDF=np.array([np.pi]))

        R = RT1(1., np.deg2rad(inc), np.deg2rad(inc),
                np.zeros_like(inc), np.full_like(inc, np.pi),
                V=V, SRF=SRF, geometry='mono')

        self.assertTrue(np.allclose(self.int_num_1, R.calc()[3]))

    def test_example_2_int(self):
        print('Testing Example 2 ...')

        inc = self.inc2

        # ---- evaluation of second example
        V = HenyeyGreenstein(tau=np.array([0.7]), omega=np.array([0.3]),
                             t=0.7, ncoefs=20)
        SRF = CosineLobe(ncoefs=10, i=5, NormBRDF=np.array([np.pi]))

        R = RT1(1., np.deg2rad(inc), np.deg2rad(inc),
                np.zeros_like(inc), np.full_like(inc, np.pi),
                V=V, SRF=SRF, geometry='mono')

        self.assertTrue(np.allclose(self.int_num_2, R.calc()[3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

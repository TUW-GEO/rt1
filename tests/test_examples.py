# -*- coding: utf-8 -*-

"""
test examples given in paper by comparison against reference solution
"""

import unittest
import numpy as np

import sys

#sys.path.append(os.path.join('..', 'mintrend'))
sys.path.append('..')
from rt1.rt1 import RT1
from rt1.volume import Rayleigh
from rt1.surface import Isotropic, CosineLobe

import json


class TestExamples(unittest.TestCase):

    def setUp(self):
        pass

    def test_example1(self):

        S = CosineLobe(ncoefs=10)
        V = Rayleigh(tau=0.7, omega=0.3)

        fname = './example1.json'
        x = json.load(open(fname,'r'))
        inc = x['inc']

        fn = None
        I0 = 1.
        phi_0 = x['phii']
        phi_ex = x['phiex']
        step = 10
        for i in xrange(0,len(inc),step):
            t0 = inc[i]
            mu_0 = np.cos(t0)
            mu_ex = mu_0*1. # test is for backscattering case
            RT = RT1(I0, mu_0, mu_ex, phi_0, phi_ex, RV=V, SRF=S, fn=fn)
            Itot, Isurf, Ivol, Iint = RT.calc()

            self.assertAlmostEqual(Isurf,x['surf'][i],10)
            self.assertAlmostEqual(Ivol,x['vol'][i],10)








if __name__ == "__main__":
    unittest.main()



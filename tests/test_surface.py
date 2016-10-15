# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.surface import Isotropic, CosineLobe
from scipy import special as sc


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


    def test_cosine(self):
        S = CosineLobe(ncoefs=10)
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        self.assertAlmostEqual(S.thetaBRDF(theta_i,theta_s, phi_i, phi_s),0.,15) #--> 0
        self.assertAlmostEqual(S.brdf(theta_i, theta_s, phi_i, phi_s), 0., 20)   #cos(THeTa)=0 --> 0.

        theta_i = 0.
        theta_s = np.deg2rad(60.)
        phi_i = 0.
        phi_s = 0.
        self.assertAlmostEqual(S.thetaBRDF(theta_i,theta_s, phi_i, phi_s),0.5,15) #--> 0.5
        self.assertAlmostEqual(S.brdf(theta_i, theta_s, phi_i, phi_s), 0.5**5., 10)


    def test_cosine_coeff(self):
        S = CosineLobe(ncoefs=10)
        self.assertAlmostEqual(S._get_legcoef(0), 15.*np.sqrt(np.pi)/(16.*sc.gamma(3.5)*sc.gamma(4.)))
        self.assertAlmostEqual(S._get_legcoef(2), 75.*np.sqrt(np.pi)/(16.*sc.gamma(2.5)*sc.gamma(5.)))

    def test_expansion(self):
        S = CosineLobe(ncoefs=10)
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        # test that it works in general
        res = S.legexpansion()
        #~ print res
        #~ assert False

        # reference solution based on first 2 Legrende polynomials
        S = CosineLobe(ncoefs=4)
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.
        r = S._eval_legpoly(theta_i,theta_s,phi_i,phi_s) # P0=1 --> result should be independen of angles; --> result should be similar to coefficients



        #~ S._get_legcoef(8)
        #~ S._get_legcoef(10)
        #~ for i in xrange(8):
            #~ print i
            #~ print S._get_legcoef(i), S._get_legcoef(i)


        ref = S._get_legcoef(0)*1. + S._get_legcoef(1)*0. + S._get_legcoef(2)*(-0.5)  + S._get_legcoef(4)*(-1.5) #+ S._get_legcoef(6) * (-5./16.) + S._get_legcoef(8) * (35./128.) + S._get_legcoef(10)*(-63./256.)
        self.assertAlmostEqual(r, ref, 15)





if __name__ == "__main__":
    unittest.main()



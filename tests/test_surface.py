# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
from rt1.surface import Isotropic, CosineLobe
from scipy import special as sc
import sympy as sp


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
        S = CosineLobe(ncoefs=10, i=5)
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
        # test legcoefs for example in paper
        n=10
        S = CosineLobe(ncoefs=n, i=5)
        for i in xrange(n):
            z1 = 120.*np.sqrt(np.pi)*(1./128.+i/64.)
            z2 = sc.gamma((7.-i)*0.5)*sc.gamma((8.+i)*0.5)
            self.assertAlmostEqual(S._get_legcoef(i), z1/z2)

        self.assertAlmostEqual(S._get_legcoef(0), 15.*np.sqrt(np.pi)/(16.*sc.gamma(3.5)*sc.gamma(4.)))
        self.assertAlmostEqual(S._get_legcoef(2), 75.*np.sqrt(np.pi)/(16.*sc.gamma(2.5)*sc.gamma(5.)))

    def test_expansion_cosine_lobe(self):
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        # reference solution based on first N Legrende polynomials
        ncoefs = 10
        S = CosineLobe(ncoefs=ncoefs, i=5)   # means coefficients 0...9; i=5 is for the example in the paper

        # input parameters are set in a way that COS_THETA = 1
        # and therefore only the legendre coefficients should be returned
        # which is always 1 in case that cos_THETA=1
        theta_i = 0.1234
        theta_s = np.pi/2.
        phi_i = 0.3456
        phi_s = 0.

        r = S._eval_legpoly(theta_i,theta_s,phi_i,phi_s, geometry='ffff')
        #ref = S._get_legcoef(0)*1. + S._get_legcoef(1)*1. #+ S._get_legcoef(2)*(-0.5)  #+ S._get_legcoef(4)*(3./8.) + S._get_legcoef(6)*(-5./16.) + S._get_legcoef(8) * (35./128.) #+ S._get_legcoef(10)*(-63./256.)

        # calculate reference solution
        refs = []
        for k in xrange(ncoefs):
            refs.append(S._get_legcoef(k)*1.)
        ref = np.array(refs).sum()
        self.assertAlmostEqual(r, ref, 15)



if __name__ == "__main__":
    unittest.main()



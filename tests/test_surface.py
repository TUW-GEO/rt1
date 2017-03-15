# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append('..')
# from rt1.volume import Rayleigh
# from rt1.rt1 import RT1

from rt1.surface import Isotropic, CosineLobe, HenyeyGreenstein
from scipy import special as sc
# import sympy as sp


class TestSurface(unittest.TestCase):

    def test_isotropic(self):
        S = Isotropic()
        t_0 = np.pi/2.
        t_ex = 0.234234
        p_0 = np.pi/2.
        p_ex = 0.

        self.assertEqual(S.brdf(t_0, t_ex, p_0, p_ex), 1./np.pi)
        self.assertEqual(S.brdf(t_0, t_ex, p_0, p_ex), 1./np.pi)
        self.assertEqual(S.brdf(t_0, t_ex, p_0, p_ex), 1./np.pi)

    def test_cosine(self):
        S = CosineLobe(ncoefs=10, i=5, NormBRDF = np.pi)
        t_0 = np.pi/2.
        t_ex = 0.234234
        p_0 = np.pi/2.
        p_ex = 0.

        self.assertAlmostEqual(S.scat_angle(t_0,t_ex, p_0, p_ex, S.a),0.,15) #--> 0
        self.assertAlmostEqual(S.brdf(t_0, t_ex, p_0, p_ex), 0., 20)   #cos(THeTa)=0 --> 0.

        t_0 = 0.
        t_ex = np.deg2rad(60.)
        p_0 = 0.
        p_ex = 0.
        self.assertAlmostEqual(S.scat_angle(t_0,t_ex, p_0, p_ex, S.a),0.5,15) #--> 0.5
        self.assertAlmostEqual(S.brdf(t_0, t_ex, p_0, p_ex), 0.5**5., 10)

    def test_cosine_coeff(self):
        # test legcoefs for example in paper
        n=10
        S = CosineLobe(ncoefs=n, i=5, NormBRDF = np.pi)
        for i in range(n):
            z1 = 120.*np.sqrt(np.pi)*(1./128.+i/64.)
            z2 = sc.gamma((7.-i)*0.5)*sc.gamma((8.+i)*0.5)
            self.assertAlmostEqual(S._get_legcoef(i), z1/z2)

        self.assertAlmostEqual(S._get_legcoef(0), 15.*np.sqrt(np.pi)/(16.*sc.gamma(3.5)*sc.gamma(4.)))
        self.assertAlmostEqual(S._get_legcoef(2), 75.*np.sqrt(np.pi)/(16.*sc.gamma(2.5)*sc.gamma(5.)))

    def test_expansion_cosine_lobe(self):
        # theta_i = np.pi/2.
        # theta_s = 0.234234
        # phi_i = np.pi/2.
        # phi_s = 0.

        # reference solution based on first N Legrende polynomials
        ncoefs = 10
        S = CosineLobe(ncoefs=ncoefs, i=5, NormBRDF = np.pi)   # means coefficients 0...9; i=5 is for the example in the paper

        # input parameters are set in a way that COS_THETA = 1
        # and therefore only the legendre coefficients should be returned
        # which is always 1 in case that cos_THETA=1
        t_0 = 0.1234
        t_ex = np.pi/2.
        p_0 = 0.3456
        p_ex = 0.

        r = S._eval_legpoly(t_0,t_ex,p_0,p_ex, geometry='ffff')
        #ref = S._get_legcoef(0)*1. + S._get_legcoef(1)*1. #+ S._get_legcoef(2)*(-0.5)  #+ S._get_legcoef(4)*(3./8.) + S._get_legcoef(6)*(-5./16.) + S._get_legcoef(8) * (35./128.) #+ S._get_legcoef(10)*(-63./256.)

        # calculate reference solution
        refs = []
        for k in range(ncoefs):
            refs.append(S._get_legcoef(k)*1.)
        ref = np.array(refs).sum()
        self.assertAlmostEqual(r, ref, 15)


    def test_normalization(self):
        # test normalization results for the extreme case of isotropic scattering.
        # this is done by testing for different geometries the following cases:
        # Isotropic := 1/pi = CosineLobe(i=0) = HenyeyGreenstein(t=0)

        N = 20
        t_0 = np.random.random(N)*np.pi
        t_ex = np.random.random(N)*np.pi
        p_0 = np.pi/4.
        p_ex = np.pi/4.

        for i in range(N):
            I = Isotropic()
            self.assertEqual(I.brdf(t_0[i], t_ex[i], p_0, p_ex), 1./np.pi)

            ncoefs = 10
            C = CosineLobe(ncoefs=ncoefs, i=0)
            self.assertEqual(C.brdf(t_0[i], t_ex[i], p_0, p_ex), 1./np.pi)

            H = HenyeyGreenstein(t=0, ncoefs=5)
            self.assertEqual(H.brdf(t_0[i], t_ex[i], p_0, p_ex), 1./np.pi)







if __name__ == "__main__":
    unittest.main()



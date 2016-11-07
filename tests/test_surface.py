# -*- coding: utf-8 -*-

import unittest
import numpy as np

from nose.tools import nottest

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

    def test_expansion_cosine_lobe(self):
        theta_i = np.pi/2.
        theta_s = 0.234234
        phi_i = np.pi/2.
        phi_s = 0.

        # reference solution based on first N Legrende polynomials
        S = CosineLobe(ncoefs=10)   # means coefficients 0...9

        # input parameters are set in a way that COS_THETA = 0
        # and therefore only the legendre coefficients should be returned
        theta_i = 0.1234
        theta_s = np.pi/2.
        phi_i = 0.3456
        phi_s = np.pi/2.

        r = S._eval_legpoly(theta_i,theta_s,phi_i,phi_s, geometry='ffff')
        #~ ref = S._get_legcoef(0)*1. + S._get_legcoef(1)*0. + S._get_legcoef(2)*(-0.5)  + S._get_legcoef(4)*(3./8.) + S._get_legcoef(6)*(-5./16.) + S._get_legcoef(8) * (35./128.) #+ S._get_legcoef(10)*(-63./256.)

        refs = []
        refs.append(S._get_legcoef(0)*1.)
        refs.append(S._get_legcoef(2)*(-0.5))
        refs.append(S._get_legcoef(4)*(3./8.))
        refs.append(S._get_legcoef(6) * (-5./16.))
        refs.append(S._get_legcoef(8) * (35./128.))
        #~ refs.append(S._get_legcoef(10) * (-63./256.))
        ref = np.array(refs).sum()

        self.assertAlmostEqual(r, ref, 15)

    @nottest  # does not work properly yet!
    def test_eval_legpoly(self):
        S = CosineLobe(ncoefs=2)

        t0 = np.pi/2.
        ts = 0.234234
        p0 = np.pi/2.
        ps = 0.

        theta_i = sp.Symbol('theta_i')
        theta_s = sp.Symbol('theta_s')
        phi_i = sp.Symbol('phi_i')
        phi_s = sp.Symbol('phi_s')
        n = sp.Symbol('n')

        r1 = S._eval_legpoly(t0,ts,p0,ps, geometry='ffff')  ## sum goes until NCOEFS+1; here 11
        #~ print 'Legcoefs: ', S.legcoefs
        refs = []
        for i in xrange(S.ncoefs):
            #~ print ''
            #~ print ''
            #~ print 'i: ', i
            # compare individual references against those of a scatterer with reduced coefficients
            ref = S._get_legcoef(i)*sp.legendre(i,S.thetaBRDF(theta_i,theta_s,phi_i,phi_s)).xreplace({n:i, theta_i:t0,theta_s:ts,phi_i:p0,phi_s:ps})

            ST = CosineLobe(ncoefs=i+1)
            r = ST._eval_legpoly(t0,ts,p0,ps, geometry='ffff')
            refs.append(ref)

            print ''
            print 'i: ', i
            print 'Refs: ', refs
            print 'r: ', r
            self.assertAlmostEqual(r, np.array(refs).sum())

        # check overall results
        refs = np.array(refs)
        self.assertAlmostEqual(r1, refs.sum())






if __name__ == "__main__":
    unittest.main()



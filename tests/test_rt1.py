# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys

sys.path.append("..")
from rt1.rt1 import RT1
from rt1.volume import Rayleigh, HenyeyGreenstein

# from rt1.coefficients import RayleighIsotropic
from rt1.surface import Isotropic, CosineLobe

from scipy.special import gamma


class TestRT1(unittest.TestCase):
    def setUp(self):
        self.I0 = 1.0
        self.t_0 = np.deg2rad(60.0)
        self.t_ex = np.deg2rad(60.0)
        self.p_0 = np.ones_like(self.t_0) * 0.0
        self.p_ex = np.ones_like(self.t_0) * 0.0
        self.V = Rayleigh(tau=np.array([0.7]), omega=np.array([0.3]))
        self.S = Isotropic()
        # self.C = RayleighIsotropic()

    def test_init(self):
        RT = RT1(
            self.I0, self.t_0, self.t_ex, self.p_0, self.p_ex, V=self.V, SRF=self.S
        )
        self.assertTrue(RT.t_0 == self.t_0)

    def test_calc(self):
        # just try to get it running simply without further testing
        RT = RT1(
            self.I0, self.t_0, self.t_ex, self.p_0, self.p_ex, V=self.V, SRF=self.S
        )
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertTrue(np.allclose(Itot, Isurf + Ivol + Iint))

        V = Rayleigh(tau=np.array([0.0]), omega=np.array([0.0]))
        RT = RT1(self.I0, self.t_0, self.t_ex, self.p_0, self.p_ex, V=V, SRF=self.S)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertEqual(Ivol, 0.0)
        self.assertEqual(Iint, 0.0)  # todo gives nan
        self.assertEqual(Itot, Isurf)
        self.assertTrue(Isurf > 0.0)

    def test_surface(self):
        V = Rayleigh(tau=np.array([0.0]), omega=np.array([0.0]))
        t_0 = np.deg2rad(60.0)
        RT = RT1(4.0, t_0, self.t_ex, self.p_0, self.p_ex, V=V, SRF=self.S)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertTrue(np.allclose(Isurf, 2.0 / np.pi, 15))

    def test_volume(self):
        t_0 = np.deg2rad(60.0)
        t_ex = np.deg2rad(60.0)
        V = Rayleigh(tau=np.array([0.0]), omega=np.array([0.0]))
        RT = RT1(self.I0, t_0, t_ex, self.p_0, self.p_ex, V=V, SRF=self.S)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertEqual(Ivol, 0.0)

    def test_fn_coefficients_RayIso(self):
        # test if calculation of fn coefficients is correct
        # this is done by comparing the obtained coefficients
        # against the analytical solution using a Rayleigh volume
        # and isotropic surface scattering phase function
        S = Isotropic()
        V = Rayleigh(tau=np.array([0.7]), omega=np.array([0.3]))
        t_0 = np.deg2rad(60.0)
        t_ex = np.deg2rad(60.0)
        p_0 = 0.0
        RT = RT1(self.I0, t_0, t_ex, p_0, self.p_ex, V=V, SRF=S)

        # the reference solutions should be (see rayleighisocoefficients.pdf)
        f0 = 3.0 / (16.0 * np.pi) * (3.0 - np.cos(t_0) ** 2.0)
        f1 = 0.0
        f2 = 3.0 / (16.0 * np.pi) * (3.0 * np.cos(t_0) ** 2.0 - 1.0)
        # and all others are 0.

        self.assertTrue(
            np.allclose([f0, f1, f2], RT._fnevals(t_0, p_0, t_ex, self.p_ex))
        )

    def test_fn_coefficients_RayCosine(self):
        # test if calculation of fn coefficients is correct
        # for a cosine lobe with reduced number of coefficients
        # this is done by comparing the obtained coefficients
        # against the analytical solution using a Rayleigh volume
        # and isotropic surface scattering phase function
        S = CosineLobe(ncoefs=1, i=5)
        V = Rayleigh(tau=np.array([0.7]), omega=np.array([0.3]))
        # --> cosTHETA = 0.

        # tests are using full Volume phase function, but only
        # ncoef times the coefficients from the surface

        t_0 = np.pi / 2.0
        t_ex = 0.234234
        p_0 = np.pi / 2.0
        p_ex = 0.0

        RT = RT1(self.I0, t_0, t_ex, p_0, p_ex, V=V, SRF=S, geometry="vvvv")
        res = RT._fnevals(t_0, p_0, t_ex, p_ex)

        # ncoefs = 1
        # analtytical solution for ncoefs = 1 --> n=0

        a0 = (3.0 / (16.0 * np.pi)) * (4.0 / 3.0)
        a2 = (3.0 / (16.0 * np.pi)) * (2.0 / 3.0)
        b0 = (15.0 * np.sqrt(np.pi)) / (16.0 * gamma(3.5) * gamma(4.0))

        # print 'a0:', a0, V._get_legcoef(0)
        # print 'a2:', a2, V._get_legcoef(2)
        # print 'b0: ', b0, S._get_legcoef(0)

        ref0 = 1.0 / 4.0 * b0 * (8.0 * a0 - a2 - 3.0 * a2 * np.cos(2.0 * t_0))
        ref2 = 3.0 / 4.0 * a2 * b0 * (1.0 + 3.0 * np.cos(2.0 * t_0))

        self.assertTrue(np.allclose([ref0, ref2], [res[0], res[2]]))

        # ncoefs = 2
        # first and third coef should be the same as for ncoefs=1
        S = CosineLobe(ncoefs=2, i=5, NormBRDF=np.array([np.pi]))
        RT = RT1(self.I0, t_0, t_ex, p_0, p_ex, V=V, SRF=S, geometry="ffff")
        res2 = RT._fnevals(t_0, p_0, t_ex, p_ex)

        self.assertTrue(np.allclose([ref0, ref2], [res2[0], res2[2]]))

    def test_fn_coefficients_HGIso(self):
        # test if calculation of fn coefficients is correct
        # this is done by comparing the obtained coefficients
        # against the analytical solution using a Rayleigh volume
        # and isotropic surface scattering phase function
        S = Isotropic()

        t_0 = 0.0
        t_ex = 0.0
        p_0 = 0.0
        p_ex = np.pi

        V = HenyeyGreenstein(
            omega=np.array([0.2]), tau=np.array([1.7]), t=0.7, ncoefs=1
        )

        RT = RT1(self.I0, t_0, t_ex, p_0, p_ex, V=V, SRF=S, geometry="ffff")

        r = RT._fnevals(t_0, p_0, t_ex, p_ex)

        self.assertTrue(np.allclose(r[0], 1.0 / (2.0 * np.pi)))

        # V = HenyeyGreenstein(omega=0.2, tau=1.7, t=0.7,ncoefs=2)
        # RT = RT1(self.I0, mu_0, mu_ex, phi_0, phi_ex, V=V, SRF=S)
        # r = RT._get_fn(0, RT.theta_0, RT.phi_0)
        # self.assertEqual(r,1./(2.*np.pi))


# todo test for tau-omgea zero order


# tests with I0=0 and extreme angles
# zero order solution ???

# def test_x(self):
# A = AR1()
# x = A.x(self.N, self.std, self.phi, method='slow')
# y = A.x(self.N, self.std, self.phi, mean=0.5, method='slow')
#
# self.assertEqual(len(x), self.N)
# self.assertTrue(np.abs(x.mean()) < 1.e-9)
# self.assertTrue(np.abs(np.abs(y.mean())-0.5) < 1.e-9)
#
# def test_methods(self):
# seed = 12345
# A = AR1()
# x = A.x(self.N, self.std, self.phi, method='slow', seed=seed)
# y = A.x(self.N, self.std, self.phi, seed=seed)
# z = A.x(self.N, self.std, self.phi, method='fast', seed=seed)
# self.assertTrue(np.array_equal(y,x))
# print x / y
# self.assertTrue(np.allclose(x/y, np.ones(self.N).astype('float'),
#                               rtol=1.E-9))
#
# def test_acorr(self):
# A = AR1()
# x = A.x(self.N, self.std, self.phi, method='slow')
# c = acorr_1_lag1(x)
# self.assertTrue(np.abs(1.-c[0]/1.)<0.000000001)  # autocorrelation
# print c
# self.assertTrue(c[1] < 1.)
# self.assertEqual(c[1], A.rho1_1d(x))
#
# for phi in [0.2, 0.5, 0.75, 1.]:
# # with noise
# x = A.x(self.N, self.std, phi, method='slow')
# c = acorr_1_lag1(x)
# self.assertAlmostEqual(c[1], phi,places=1)
#
# # without noise
# x = A.x(self.N, 0.0000000000001, phi, method='slow')
# c = acorr_1_lag1(x)
# self.assertAlmostEqual(c[1], phi,places=1)
#
# def test_fast(self):
# # test that fast method produces the same results
# N = 1000
# sigma = 0.1
# phi = 0.5
# seed = 12345
#
# A = AR1()
# B = AR1()
#
# a = A.x(N, sigma, phi, seed=seed, method='slow')
# b = B.x(N, sigma, phi, seed=seed, method='fast')
#
# self.assertTrue(np.array_equal(a,b))
#
#
#
#
# def test_fit(self):
# # test fit with low noise
# A = AR1()
# x = A.x(self.N, 0.01, self.phi, method='slow')
#
# # retrieve parameters
# phi, std_r = A.fit(x)
# print 'phi, self.phi: ', phi, self.phi
# print 'std, self.std: ', std_r, self.std
# #self.assertAlmostEqual(phi, self.phi, places=1)
# todo: more robust testing of correct results!
# # we only check for PHI, as an error in PHI will automatically
# # propagate into an error in SIGMA
# #self.assertAlmostEqual(std_r, self.std,places=1)
#
# self.assertEqual(phi, A.rho1_1d(x))
# self.assertEqual(std_r, (np.sqrt(x.var()*(1.-phi**2.))))
#
# # check for masked arrays
# y = np.hstack((x,[np.nan,np.nan]))
# y = np.ma.array(y, mask=np.isnan(y))
# phiy, std_ry = A.fit(y)
# ratio = np.abs(1.-phi/phiy)
# self.assertTrue(ratio < 0.1/100.)  # better than 0.1%s
# ratio = np.abs(1.-std_r/std_ry)
# self.assertTrue(ratio < 0.1/100.)  # better than 0.1%s
#
# # test for 2D data (singular0 processing)
# ny = 5
# nx = 3
# X = np.ones((self.N,ny,nx))*np.nan
# for i in xrange(ny):
# for j in xrange(nx):
# X[:,i,j] = x[:]
#
#
# PHI, STD = A.fit(X)
#
# self.assertTrue(np.all(PHI==phi))
# self.assertTrue(np.all(np.abs(1.-STD/std_r) < 0.001)) #0.1% accuracy

# parallel processing
# PHI1, STD1 = A.fit(X, nproc=4)
# self.assertTrue(np.all(PHI1==phi))
# self.assertTrue(np.all(np.abs(1.-STD1/std_r) < 0.001)) #0.1% accuracy


if __name__ == "__main__":
    unittest.main()

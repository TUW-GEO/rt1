# -*- coding: utf-8 -*-

import unittest
import numpy as np

import sys
#import os
#import tempfile

#sys.path.append(os.path.join('..', 'mintrend'))
sys.path.append('..')
from rt1.rt1 import RT1
from rt1.volume import Rayleigh
from rt1.coefficients import RayleighIsotropic
from rt1.surface import Isotropic



class TestRT1(unittest.TestCase):

    def setUp(self):
        self.I0 = 1.
        self.mu_0 = 0.5
        self.mu_ex = 0.5
        self.phi_0 = 0.
        self.phi_ex = 0.
        self.V = Rayleigh(tau=0.7, omega=0.3)
        self.S = Isotropic()
        self.C = RayleighIsotropic()

    def test_init(self):
        RT = RT1(self.I0, self.mu_0, self.mu_ex, self.phi_0, self.phi_ex, RV=self.V, SRF=self.S, Fn=self.C)

    def test_calc(self):
        # just try to get it running simply without further testing
        RT = RT1(self.I0, self.mu_0, self.mu_ex, self.phi_0, self.phi_ex, RV=self.V, SRF=self.S, Fn=self.C)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertEqual(Itot, Isurf+Ivol+Iint)

        V = Rayleigh(tau=0., omega=0.)
        RT = RT1(self.I0, self.mu_0, self.mu_ex, self.phi_0, self.phi_ex, RV=V, SRF=self.S, Fn=self.C)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertEqual(Ivol, 0.)
        self.assertEqual(Iint, 0.)  # todo gives nan
        self.assertEqual(Itot, Isurf)
        self.assertTrue(Isurf>0.)

    def test_surface(self):
        V = Rayleigh(tau=0., omega=0.)
        mu_0 = 0.5
        RT = RT1(4., mu_0, self.mu_ex, self.phi_0, self.phi_ex, RV=V, SRF=self.S, Fn=self.C)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertEqual(Isurf, 2./np.pi)

    def test_volume(self):
        mu_0 = 0.5
        mu_ex = 0.5
        V = Rayleigh(tau=0., omega=0.2)
        RT = RT1(self.I0, mu_0, mu_ex, self.phi_0, self.phi_ex, RV=V, SRF=self.S, Fn=self.C)
        Itot, Isurf, Ivol, Iint = RT.calc()
        self.assertEqual(Ivol, 0.)


    def test_interaction(self):
        pass

    #~ def test_fint(self):
        #~ RT = RT1(self.I0, self.mu_0, self.mu_ex, RV=self.V, SRF=self.S, Fn=self.C)
        #~ mu1 = 0.5
        #~ mu2 = 0.5
        #~ F = RT._calc_Fint(mu1, mu2)
        #~ self.assertEqual(F, 0.)



# todo test for tau-omgea zero order








#~ tests with I0=0 and extreme angles
#~ zero order solution ???






    #~ def test_x(self):
        #~ A = AR1()
        #~ x = A.x(self.N, self.std, self.phi, method='slow')
        #~ y = A.x(self.N, self.std, self.phi, mean=0.5, method='slow')
#~
        #~ self.assertEqual(len(x), self.N)
        #~ self.assertTrue(np.abs(x.mean()) < 1.e-9)
        #~ self.assertTrue(np.abs(np.abs(y.mean())-0.5) < 1.e-9)
#~
    #~ def test_methods(self):
        #~ seed = 12345
        #~ A = AR1()
        #~ x = A.x(self.N, self.std, self.phi, method='slow', seed=seed)
        #~ y = A.x(self.N, self.std, self.phi, seed=seed)
        #~ z = A.x(self.N, self.std, self.phi, method='fast', seed=seed)
        #~ self.assertTrue(np.array_equal(y,x))
        #~ print x / y
        #~ self.assertTrue(np.allclose(x/y, np.ones(self.N).astype('float'), rtol=1.E-9))
#~
    #~ def test_acorr(self):
        #~ A = AR1()
        #~ x = A.x(self.N, self.std, self.phi, method='slow')
        #~ c = acorr_1_lag1(x)
        #~ self.assertTrue(np.abs(1.-c[0]/1.)<0.000000001)  # autocorrelation
        #~ print c
        #~ self.assertTrue(c[1] < 1.)
        #~ self.assertEqual(c[1], A.rho1_1d(x))
#~
        #~ for phi in [0.2, 0.5, 0.75, 1.]:
            #~ # with noise
            #~ x = A.x(self.N, self.std, phi, method='slow')
            #~ c = acorr_1_lag1(x)
            #~ self.assertAlmostEqual(c[1], phi,places=1)
#~
            #~ # without noise
            #~ x = A.x(self.N, 0.0000000000001, phi, method='slow')
            #~ c = acorr_1_lag1(x)
            #~ self.assertAlmostEqual(c[1], phi,places=1)
#~
    #~ def test_fast(self):
        #~ # test that fast method produces the same results
        #~ N = 1000
        #~ sigma = 0.1
        #~ phi = 0.5
        #~ seed = 12345
#~
        #~ A = AR1()
        #~ B = AR1()
#~
        #~ a = A.x(N, sigma, phi, seed=seed, method='slow')
        #~ b = B.x(N, sigma, phi, seed=seed, method='fast')
#~
        #~ self.assertTrue(np.array_equal(a,b))
#~
#~
#~
#~
    #~ def test_fit(self):
        #~ # test fit with low noise
        #~ A = AR1()
        #~ x = A.x(self.N, 0.01, self.phi, method='slow')
#~
        #~ # retrieve parameters
        #~ phi, std_r = A.fit(x)
        #~ print 'phi, self.phi: ', phi, self.phi
        #~ print 'std, self.std: ', std_r, self.std
        #~ #self.assertAlmostEqual(phi, self.phi, places=1)  # todo: more robust testing of correct results!
        #~ # we only check for PHI, as an error in PHI will automatically
        #~ # propagate into an error in SIGMA
        #~ #self.assertAlmostEqual(std_r, self.std,places=1)
#~
        #~ self.assertEqual(phi, A.rho1_1d(x))
        #~ self.assertEqual(std_r, (np.sqrt(x.var()*(1.-phi**2.))))
#~
        #~ # check for masked arrays
        #~ y = np.hstack((x,[np.nan,np.nan]))
        #~ y = np.ma.array(y, mask=np.isnan(y))
        #~ phiy, std_ry = A.fit(y)
        #~ ratio = np.abs(1.-phi/phiy)
        #~ self.assertTrue(ratio < 0.1/100.)  # better than 0.1%s
        #~ ratio = np.abs(1.-std_r/std_ry)
        #~ self.assertTrue(ratio < 0.1/100.)  # better than 0.1%s
#~
        #~ # test for 2D data (singular0 processing)
        #~ ny = 5
        #~ nx = 3
        #~ X = np.ones((self.N,ny,nx))*np.nan
        #~ for i in xrange(ny):
            #~ for j in xrange(nx):
                #~ X[:,i,j] = x[:]
#~
#~
        #~ PHI, STD = A.fit(X)
#~
        #~ self.assertTrue(np.all(PHI==phi))
        #~ self.assertTrue(np.all(np.abs(1.-STD/std_r) < 0.001)) # 0.1% accuracy

        # parallel processing
        #PHI1, STD1 = A.fit(X, nproc=4)
        #self.assertTrue(np.all(PHI1==phi))
        #self.assertTrue(np.all(np.abs(1.-STD1/std_r) < 0.001)) # 0.1% accuracy



if __name__ == "__main__":
    unittest.main()



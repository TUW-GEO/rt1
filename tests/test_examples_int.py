# -*- coding: utf-8 -*-

"""
test examples given in paper by comparison against reference solution.

the actual comparison is done by checking the equality of the interaction-term
with a numerical solution (generated with numerical_evaluation.py)

"""

import unittest
import numpy as np

import sys

#sys.path.append(os.path.join('..', 'mintrend'))
sys.path.append('..')
from rt1.rt1 import RT1
from rt1.volume import Rayleigh, HenyeyGreenstein
from rt1.surface import Isotropic, CosineLobe


class TestExamples(unittest.TestCase):

    def setUp(self):
        # read reference solutions for backscattering case
        fname1 = 'example1_int.csv'
        x1 = np.loadtxt('tests/' + fname1, delimiter=',',skiprows=0)
        self.inc1 = x1[:,0]
        self.int_num_1 = x1[:,1]

        fname2 = 'example2_int.csv'
        x2 = np.loadtxt('tests/' + fname2, delimiter=',',skiprows=0)
        self.inc2 = x2[:,0]
        self.int_num_2 = x2[:,1]

    def test_example_1_int(self):
        print('Testing Example 1 ...')
       
    
        inc = self.inc1
        
        # initialize output fields for faster processing
        Itot = np.ones_like(inc)*np.nan
        Isurf = np.ones_like(inc)*np.nan
        Ivol = np.ones_like(inc)*np.nan
        Iint = np.ones_like(inc)*np.nan
        
        
        
        # ---- evaluation of first example
        V = Rayleigh(tau=0.7, omega=0.3)
        SRF = CosineLobe(ncoefs=11, i=5, NormBRDF = np.pi) # 11 instead of 10 coefficients used to assure 7 digit precision
        
        fn = None
        for i in xrange(len(inc)):
            # set geometries
            mu_0 = np.cos(np.deg2rad(inc[i]))
            phi_0 = np.deg2rad(0.)
            phi_ex = phi_0 + np.pi
        
        
            R = RT1(1., mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
            fn = R.fn  # store coefficients for faster itteration
            Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()
    

        #for i in range(0,len(self.inc1)):
            self.assertAlmostEqual(self.int_num_1[i],Iint[i])
        

    def test_example_2_int(self):
        print('Testing Example 2 ...')

    
        inc = self.inc2
        
        # initialize output fields for faster processing
        Itot = np.ones_like(inc)*np.nan
        Isurf = np.ones_like(inc)*np.nan
        Ivol = np.ones_like(inc)*np.nan
        Iint = np.ones_like(inc)*np.nan
        
        
        # ---- evaluation of second example
        V = HenyeyGreenstein(tau=0.7, omega=0.3, t=0.7, ncoefs=20)
        SRF = CosineLobe(ncoefs=11, i=5, NormBRDF = np.pi) # 11 instead of 10 coefficients used to assure 7 digit precision
        
        
        fn = None
        for i in xrange(len(inc)):
            # set geometries
            mu_0 = np.cos(np.deg2rad(inc[i]))
            phi_0 = np.deg2rad(0.)
            phi_ex = phi_0 + np.pi
        
        
            R = RT1(1., mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
            fn = R.fn  # store coefficients for faster itteration
            Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()


        #for i in range(0,len(self.inc1)):
            self.assertAlmostEqual(self.int_num_2[i],Iint[i])



if __name__ == "__main__":
    unittest.main()


    
    
    

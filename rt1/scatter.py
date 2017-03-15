"""
Define general object for scattering calculations
This is the basis object for any Surface and Vollume objects
"""

# python2/3 compatability
from __future__ import print_function

# general other imports
import sympy as sp

class Scatter(object):
    def __init__(self):
        pass


    def scat_angle(self, t_0, t_ex, p_0, p_ex, a):
        """
        Function to return the generalized scattering angle with respect to the given zenith-angles

        standard-choices assigned in the volume- and surface class:
        Surface: a=[ 1,1,1] ... ordinary scattering angle      cos[i]*cos[s] + sin[i]*sin[s]*cos[phii - phis]
        Volume:  a=[-1,1,1] ... pi-shifted scattering angle   -cos[i]*cos[s] + sin[i]*sin[s]*cos[phii - phis]
        ----------
        
        Parameters:

        t_0 : float
             incident zenith-angle
        p_0 : float
             incident azimuth-angle
        t_ex : float
              exit zenith-angle
        p_ex : float
              exit azimuth-angle
        a : [ float , float , float ] 
           generalized scattering angle parameters
        
        """
        return a[0]*sp.cos(t_0)*sp.cos(t_ex)+a[1]*sp.sin(t_0)*sp.sin(t_ex)*sp.cos(p_0)*sp.cos(p_ex)+a[2]*sp.sin(t_0)*sp.sin(t_ex)*sp.sin(p_0)*sp.sin(p_ex)


    def _get_legcoef(self, n0):
        """
        function to evaluate legendre coefficients
        used mainly for testing purposes
        the actual coefficient are used in the symbolic expansion
        """
        n = sp.Symbol('n')
        return self.legcoefs.xreplace({n:int(n0)}).evalf()


    def _eval_legpoly(self, t_0,t_s,p_0,p_s, geometry=None):
        """
        function to evaluate legendre coefficients based on expansion
        used mainly for testing purposes
        the actual coefficient are used in the symbolic expansion
        geometry
            'mono'
            'vvvv'
            'ffff'
        """

        assert geometry is not None, 'Geometry needs to be specified!'

        theta_0 = sp.Symbol('theta_0')
        theta_s = sp.Symbol('theta_s')
        theta_ex = sp.Symbol('theta_ex')
        phi_0 = sp.Symbol('phi_0')
        phi_s = sp.Symbol('phi_s')
        phi_ex = sp.Symbol('phi_ex')

        #mu_0 = np.cos(t_0)
        #mu_ex = np.cos(t_s)

        ###self.RV.legexpansion(self.mu_0,self.mu_ex,self.phi_0,self.phi_ex,self.geometry).do
        res = self.legexpansion(t_0, t_s, p_0, p_s, geometry).xreplace({theta_0:t_0, theta_s:t_s, phi_0:p_0,phi_s:p_s, theta_ex:t_s, phi_ex:p_s})
        print('THETA: ', self.scat_angle(theta_s,theta_ex,phi_s,phi_ex, a=[1.,1.,1.]).xreplace({theta_0:t_0, theta_s:t_s, phi_0:p_0,phi_s:p_s, theta_ex:t_s, phi_ex:p_s}).evalf())
        return res.evalf()  #.xreplace({theta_i:t0, theta_s:ts, phi_i:p0,phi_s:ps}).evalf()




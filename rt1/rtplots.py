"""
Class for quick visualization of results and used phasefunctions

polarplot() ... plot p and the BRDF as polar-plot

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scatter import Scatter

class Plots(Scatter):
    """
    Parameters:
        R ... RT1 class object
    """
    def __init__(self, R, **kwargs):
        self.R = R
    
    

    def polarplot(self, incp = [15.,35.,55.,75.], incBRDF = [15.,35.,55.,75.], pmultip = 3., BRDFmultip = 1.):
        """
        generate a polar-plot of both the volume- and the surface scattering phase function.
        
        optional parameters:
            incp        ... incidence-angles [deg] at which the volume-scattering phase-function will be plotted 
            incBRDF     ... incidence-angles [deg] at which the BRDF will be plotted
        
            
            pmultip     ... multiplicator of max-plotrange for p     (in case the chosen plotranges are not satisfying)
            BRDFmultip  ... multiplicator of max-plotrange for BRDF
        
            max-plotrange for p is given by the max. value of p in forward-direction (for the chosen incp)
            and for the BRDF by the max. value of the BRDF in specular direction (for the chosen incBRDF)
        """
        
       
        assert isinstance(incp,list), 'Error: incidence-angles for polarplot of p must be a list'
        assert isinstance(incBRDF,list), 'Error: incidence-angles for polarplot of the BRDF must be a list'
        for i in incBRDF: assert i<90, 'ERROR: the incidence-angle of the BRDF in polarplot must be < 90'
        
        assert isinstance(pmultip,float), 'Error: plotrange-multiplyer for polarplot of p must be a floating-point number'
        assert isinstance(BRDFmultip,float), 'Error: plotrange-multiplyer for polarplot of the BRDF must be a floating-point number'


        # define functions for plotting that evaluate the analytic form of the phasefunctions
        brdffunkt = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), self.R.SRF._func,"numpy") 
        phasefunkt = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), self.R.RV._func,"numpy") 
        
        
        # define functions for plotting that evaluate the used approximations in terms of legendre-polynomials
        n = sp.Symbol('n')     
        
        phasefunktapprox = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), sp.Sum(self.R.RV.legcoefs*sp.legendre(n,self.thetap('theta_i','theta_s','phi_i','phi_s', self.R.RV.a)),(n,0,self.R.RV.ncoefs-1)).doit(),"numpy") 
        brdffunktapprox = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), sp.Sum(self.R.SRF.legcoefs*sp.legendre(n,self.thetaBRDF('theta_i','theta_s','phi_i','phi_s', self.R.SRF.a)),(n,0,self.R.SRF.ncoefs-1)).doit(),"numpy") 



 
      
        # open new figure
        polarfig = plt.figure(figsize=(14,7))
        
        # plot of volume-scattering phase-function
        polartest = polarfig.add_subplot(121,projection='polar')
        plottis=np.deg2rad(incp)
        colors = ['r','g','b', 'k','c','m','y']*(len(plottis)/7+1)
        i=0
        
        pmax = pmultip*max(phasefunkt(plottis, np.pi-plottis, 0., 0.))
        
        for ti in plottis:
            color = colors[i]
            i=i+1
            thetass = np.arange(0.,2.*np.pi,.01)
            rad=phasefunkt(ti, thetass, 0., 0.)
            radapprox=phasefunktapprox(ti, thetass, 0., 0.)
            
            polartest.set_theta_direction(-1)   # set theta direction to clockwise
            polartest.set_theta_offset(np.pi/2.) # set theta to start at z-axis
            
            polartest.plot(thetass,rad, color)
            polartest.plot(thetass,radapprox, color+'--')
            polartest.arrow(-ti,pmax*1.2  ,0.,-pmax*0.8, head_width = .0, head_length=.0, fc = color, ec = color, lw=1, alpha=0.3)
            polartest.fill_between(thetass,rad,alpha=0.2, color=color)
            polartest.set_xticklabels(['$0^\circ$','$45^\circ$','$90^\circ$','$135^\circ$','$180^\circ$'])
            polartest.set_yticklabels([])
            polartest.set_rmax(pmax*1.2)
            polartest.set_title('Volume-Scattering Phase Function \n')
        
        
        
        
        # plot of BRDF
        polartest = polarfig.add_subplot(122,projection='polar')
        plottis=np.deg2rad(incBRDF)
        colors = ['r','g','b', 'k','c','m','y']*(len(plottis)/7+1)
        i=0
        
        brdfmax = BRDFmultip*max(brdffunkt(plottis, plottis, 0., 0.))
        
        for ti in plottis:
            color = colors[i]
            i=i+1
            thetass = np.arange(-np.pi/2.,np.pi/2.,.01)
            rad=brdffunkt(ti, thetass, 0., 0.)
            radapprox = brdffunktapprox(ti, thetass, 0., 0.)
            
            polartest.set_theta_direction(-1)   # set theta direction to clockwise
            polartest.set_theta_offset(np.pi/2.) # set theta to start at z-axis
            
            polartest.plot(thetass,rad, color)
            polartest.plot(thetass,radapprox, color + '--')
            polartest.fill(np.arange(np.pi/2.,3.*np.pi/2.,.01),np.ones_like(np.arange(np.pi/2.,3.*np.pi/2.,.01))*brdfmax*1.2,'k')
        
        
            polartest.arrow(-ti,brdfmax*1.2  ,0.,-brdfmax*0.8, head_width =.0, head_length=.0, fc = color, ec = color, lw=1, alpha=0.3)
            polartest.fill_between(thetass,rad,alpha=0.2, color=color)
            polartest.set_xticklabels(['$0^\circ$','$45^\circ$','$90^\circ$'])
            polartest.set_yticklabels([])
            polartest.set_rmax(brdfmax*1.2)
            polartest.set_title('Surface-BRDF\n')
        
        
        
        plt.show()

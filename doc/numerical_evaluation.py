# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:22:59 2016

@author: rquast
"""

import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import simps

"""
 this is the file that was used to generate the numerical reference-solutions
 for the tests, i.e. the files:
     example1_int.csv
     example2_int.csv



 the method used for numerical evaluation has been adapted from:
 http://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy
"""

# phase-function definitions in terms of spherical coordinate system angles (NOT zenith-angles)
def HG(thetai,thetas,phii,phis,t):
    return 1./(4.*np.pi)*(1.-t**2)/(1.+t**2 - 2.*t*(np.cos(thetai)*np.cos(thetas)+np.sin(thetai)*np.sin(thetas)*np.cos(phii-phis)))**(3./2.)

def RAYLEIGH(thetai,thetas,phii,phis):
    return (3./(16.*np.pi)*(1.+(np.cos(thetai)*np.cos(thetas)+np.sin(thetai)*np.sin(thetas)*np.cos(phii-phis))**2))

def COSLOBE(thetai,thetas,phii,phis,t):
    asdf=np.maximum((-np.cos(thetai)*np.cos(thetas)+np.sin(thetai)*np.sin(thetas)*np.cos(phii-phis))**t,0.)
    return asdf

            
# parameters for example 1 and 2 of the paper:
            
CosP=5.                     # set the power of the cosine-lobe     
HGt=0.7                     # set the asymmetry parameter of the Henyey Greenstein Function
phii = 0.                   # set the incident direction to phi_i = 0.

omega = 0.3
tau = 0.7



# set incident directions for which the integral should be evaluated
# notice that this is equal to the zenit-angle theta_0 since the arguments in the paper are theta_0 !
inc = np.arange(5.,90.,10.)

# define grid for integration 
x=np.linspace(0.,np.pi/2.,3000)
y=np.linspace(0.,2*np.pi,3000)

# initialize array for solutions
solCosRay = []
solCosHG = []


# ---- evaluation of first example
print('start of evaluation of first example')
for thetai in np.deg2rad(inc):
    # define the function that has to be integrated (i.e. Eq.20 in the paper)
    # notice the additional  np.sin(thetas)  which oritinates from integrating over theta_s instead of mu_s
    mu0 = np.cos(thetai)
    def integfunkt(thetas,phis):
        return np.array(np.sin(thetas)*2.*omega*np.exp(-tau/mu0)*mu0*np.cos(thetas)/(mu0-np.cos(thetas)) * (np.exp(-tau/mu0)-np.exp(-tau/np.cos(thetas))) *  RAYLEIGH(thetai,thetas,phii,phis)*COSLOBE(thetai,np.pi-thetas,np.pi,phis,CosP))

    # evaluate the integral using Simpson's Rule twice
    z=integfunkt(x[:,None],y)
    solCosRay = solCosRay + [simps(simps(z,y),x)]

solCosRay = np.array(solCosRay)



# ---- evaluation of second example
print('start of evaluation of second example')
for thetai in np.deg2rad(inc):
    # define the function that has to be integrated (i.e. Eq.20 in the paper)
    # notice the additional  np.sin(thetas)  which oritinates from integrating over theta_s instead of mu_s
    mu0 = np.cos(thetai)
    def integfunkt(thetas,phis):
        return np.array(np.sin(thetas)*2.*omega*np.exp(-tau/mu0)*mu0*np.cos(thetas)/(mu0-np.cos(thetas)) * (np.exp(-tau/mu0)-np.exp(-tau/np.cos(thetas))) *  HG(thetai,thetas,phii,phis,HGt)*COSLOBE(thetai,np.pi-thetas,np.pi,phis,CosP))

    # evaluate the integral using Simpson's Rule twice
    z=integfunkt(x[:,None],y)
    solCosHG = solCosHG + [simps(simps(z,y),x)]

solCosHG = np.array(solCosHG)


np.savetxt("../tests/example1_int.csv", [[i,j] for i,j in zip(inc,solCosRay)], delimiter = ",")
np.savetxt("../tests/example2_int.csv", [[i,j] for i,j in zip(inc,solCosHG)], delimiter = ",")




#  - - - - - - - - OPTIONAL GRAPHICAL EVALUATION OF TEST

# ----- evaluation of model
#from rt1.rt1 import RT1
#from rt1.volume import Rayleigh
#from rt1.volume import HenyeyGreenstein
#from rt1.surface import CosineLobe
#
#
## initialize output fields for faster processing
#Itot = np.ones_like(inc)*np.nan
#Isurf = np.ones_like(inc)*np.nan
#Ivol = np.ones_like(inc)*np.nan
#
#Iint1 = np.ones_like(inc)*np.nan
#Iint2 = np.ones_like(inc)*np.nan
#
#
#
## ---- evaluation of first example
#V = Rayleigh(tau=0.7, omega=0.3)
#SRF = CosineLobe(ncoefs=10, i=5)
#label = 'Example 1'
#
#fn = None
#for i in xrange(len(inc)):
#    # set geometries
#    mu_0 = np.cos(np.deg2rad(inc[i]))
#    mu_ex = mu_0*1.
#    phi_0 = np.deg2rad(0.)
#    phi_ex = phi_0 + np.pi
#
#
#    R = RT1(1., mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
#    fn = R.fn  # store coefficients for faster itteration
#    Itot[i], Isurf[i], Ivol[i], Iint1[i] = R.calc()
#
#
#
#
## ---- evaluation of second example
#V = HenyeyGreenstein(tau=0.7, omega=0.3, t=0.7, ncoefs=20)
#SRF = CosineLobe(ncoefs=10, i=5)
#label = 'Example 2'
#
#
#fn = None
#for i in xrange(len(inc)):
#    # set geometries
#    mu_0 = np.cos(np.deg2rad(inc[i]))
#    mu_ex = mu_0*1.
#    phi_0 = np.deg2rad(0.)
#    phi_ex = phi_0 + np.pi
#
#
#    R = RT1(1., mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
#    fn = R.fn  # store coefficients for faster itteration
#    Itot[i], Isurf[i], Ivol[i], Iint2[i] = R.calc()
#
#
#
#
#
#
#fig = plt.figure()
#axnum = fig.add_subplot(1,1,1)
#
## plot of numerical results
#axnum.plot(inc,solCosRay, 'bo', label = "Example 1")
#axnum.plot(inc,solCosHG, 'go', label = "Example 2")
#
## plot of result from model calculation 
##  !!!!!! examples.py needs to be run first to assign inc and Iint
#axnum.plot(inc,Iint1, label = "Example 1")
#axnum.plot(inc,Iint2, label = "Example 2")
#
#
#axnum.legend()
#
#plt.show()







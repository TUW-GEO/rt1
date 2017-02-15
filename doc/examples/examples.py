"""
Reproduce examples like given in the paper
"""

import sys
sys.path.append('../..')


import matplotlib.pyplot as plt
import numpy as np
import timeit

from rt1.rt1 import RT1

from rt1.rtplots import Plots


from rt1.volume import Rayleigh
from rt1.volume import HenyeyGreenstein
from rt1.volume import LinCombV

from rt1.surface import CosineLobe
from rt1.surface import HenyeyGreenstein as HGsurface
from rt1.surface import Isotropic
from rt1.surface import LinCombSRF

from rt1.rt1 import RT1




# start timer
tic = timeit.default_timer()

# set incident intensity to 1.
I0=1.
# define incidence-angle range for generation of backscatter-plots
inc = np.arange(0.,90.,2.)

# EVALUATION OF BISTATIC PLOTS
# if set to true, 3dplots will be generated,
# otherwise the code is stopped after monostatic evaluation
bistaticplot = False


'''
possible choices for example = ?

1 : example 1 from the paper
2 : example 2 from the paper
3 : example using a linear-combination for p and the BRDF
'''
example = 3



# ----------- definition of examples -----------
if example==1:
    # Example 1
    V = Rayleigh(tau=0.7, omega=0.3)
    SRF = CosineLobe(ncoefs=10, i=5)
    label = 'Example 1'
elif example == 2:
    V = HenyeyGreenstein(tau=0.7, omega=0.3, t=0.7, ncoefs=20)
    SRF = CosineLobe(ncoefs=10, i=5)
    label = 'Example 2'
elif example == 3:
    # list of volume-scattering phase-functions to be combined
    phasechoices = [HenyeyGreenstein(t=  0.5, ncoefs = 10, a=[-1.,1.,1.]),  # forward-scattering-peak
                    HenyeyGreenstein(t= -0.2, ncoefs = 10, a=[-1.,1.,1.]),  # backscattering-peak
                    HenyeyGreenstein(t= -0.5, ncoefs = 10, a=[ 1.,1.,1.]),  # downward-specular peak
                    HenyeyGreenstein(t=  0.2, ncoefs = 10, a=[ 1.,1.,1.]),  # upward-specular peak
                   ]
    # weighting-factors for the individual phase-functions
    Vweights = [.3,.3,.2,.2]


    # list of surface-BRDF-functions to be combined
    BRDFchoices = [HGsurface(ncoefs=10, t=-.4, a=[-.8,1.,1.], NormBRDF=.1),         # backscattering peak
                   HGsurface(ncoefs=10, t= .5, a=[ .8,1.,1.], NormBRDF=.1),         # specular peak
                   Isotropic(NormBRDF=.1),        # isotropic scattering contribution
                  ]
    # weighting-factors for the individual BRDF's
    BRDFweights = [.3,.4,.3]

    # generate correctly shaped arrays of the phase-functions and their corresponding weighting-factors:
    Vchoices = map(list,zip(Vweights, phasechoices))
    SRFchoices = map(list,zip(BRDFweights, BRDFchoices))


    V = LinCombV(tau=0.5, omega=0.4, Vchoices=Vchoices)

    SRF = LinCombSRF(SRFchoices=SRFchoices)
else:
    assert False, 'Choose an existing example-number or specify V and SRF explicitly'





# initialize output fields for faster processing
Itot = np.ones_like(inc)*np.nan
Isurf = np.ones_like(inc)*np.nan
Iint = np.ones_like(inc)*np.nan
Ivol = np.ones_like(inc)*np.nan


fn = None
for i in xrange(len(inc)):
    # set geometries
    mu_0 = np.cos(np.deg2rad(inc[i]))
    mu_ex = mu_0*1.
    phi_0 = np.deg2rad(0.)
    phi_ex = phi_0 + np.pi


    R = RT1(I0, mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
    fn = R.fn  # store coefficients for faster itteration
    #Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()

toc = timeit.default_timer()
print('time for coefficient evaluation: ' + str(toc-tic))


# todo this separation in two loops is actually not needed and only for debugging
tic = timeit.default_timer()
for i in xrange(len(inc)):
    mu_0 = np.cos(np.deg2rad(inc[i]))
    mu_ex = mu_0*1.
    #phi_0 = 10.
    #phi_ex = np.pi   # todo ???

    #print inc[i], mu_0, mu_ex, phi_0, phi_ex

    R = RT1(I0, mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
    Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()

toc = timeit.default_timer()
print('evaluating print-values took ' + str(toc-tic))




# ---------------- GENERATION OF POLAR-PLOTS ----------------
#       plot both p and the BRDF in a single plot
plot1 = Plots().polarplot(R,incp = list(np.linspace(0,120,5)), incBRDF = list(np.linspace(0,90,5)) , pmultip = 1.5)

#       plot only the BRDF
#Plots().polarplot(SRF = R.SRF)

#       plot only p
#Plots().polarplot(V = R.RV)


#       plot more than one phase-function simultaneously
#       example: henyey-greenstein phase function for various choices of asymmetry parameter
#hg = Plots().polarplot(V = [HenyeyGreenstein(tau=0.7, omega=0.3, t=tt, ncoefs=10) for tt in [.1,.2,.3,.4,.5,.6]], pmultip = 1., incp = [45], plabel = 'Henyey Greenstein Phase Function', paprox = False)


#       example: cosine-lobes for various choices of i and it's approximation with 10 legendre-coefficients
#cl = Plots().polarplot(SRF = [CosineLobe(ncoefs=10, i=ii) for ii in [1,5,10,15]], BRDFlabel = 'Cosine-Lobe BRDF', incBRDF = [45], BRDFaprox = True)

#       example: lafortune-lobes
#ll = Plots().polarplot(SRF = [CosineLobe(ncoefs=10, i=5, a=[.8,1.,1.])], BRDFlabel = 'Lafortune-Lobe BRDF', BRDFaprox = False)

#       example: isotropic phase function
#iso = Plots().polarplot(SRF = SRFisotropic(), BRDFlabel = 'Isotropic BRDF', BRDFaprox = False, incBRDF = [45], BRDFmultip = 2.)



# ---------------- GENERATION OF BACKSCATTER PLOTS ----------------
#       plot backscattered intensity and fractional contributions
plot2 = Plots().logmono(inc, Itot = Itot, Isurf = Isurf, Ivol = Ivol, Iint = Iint, sig0=True, noint=True)#, ylim=[-25,0])

#       plot only backscattering coefficient without fractions
#Plots().logmono(inc, Itot = Itot, Isurf = Isurf, Ivol = Ivol, Iint = Iint, sig0=True, fractions = False)

#       plot only Itot and Ivol
#Plots().logmono(inc,Itot = Itot, Ivol = Ivol, fractions = False)




# ---------------- GENERATION OF 3D PLOTS ----------------

assert bistaticplot, 'Manual stop before bistatic evaluation'


# definition of incidence-angles
thetainc= 55.
phiinc = 0.

# define plot-range
theta,phi = np.linspace(np.deg2rad(1.),np.deg2rad(89.),25),np.linspace(0.,np.pi,25)


# definition of function to evaluate bistatic contributions
# since this step might take a while an estimation of the calculation-time was included

def Rad(theta, phi, thetainc, phiinc):
    '''
    theta,phi         = [...]      plotrange-arrays
    thetainc, phiinc, = float      incidence-angles for which the model is evaluated
    '''
    # initialize output arrays
    tot3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]
    surf3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]
    vol3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]
    int3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]

    # pre-evaluation of fn-coefficients
    testfn = RT1(1., np.cos(np.deg2rad(thetainc)), np.cos(np.deg2rad(45)), phiinc, np.pi, RV=V, SRF=SRF, fn=None, geometry='fvfv').fn

    # evaluation of model
    for i in range(0,len(theta)):
        if i == 0: tic = timeit.default_timer()
        if i == 0: print('... estimating evaluation-time...')
        for j in range(0,len(phi)):
            R3d = RT1(1., np.cos(np.deg2rad(thetainc)), np.cos(theta[i]), phiinc, phi[j], RV=V, SRF=SRF, fn=testfn, geometry='ffff')
            tot3d[j][i],surf3d[j][i],vol3d[j][i],int3d[j][i] = R3d.calc()
        if i == 0: toc = timeit.default_timer()
        if i == 0: print('evaluation of 3dplot will take approximately ' + str(round((toc-tic)*len(theta)/60.,2)) + ' minutes')
    return np.array(tot3d,dtype=float) ,  np.array(surf3d,dtype=float),  np.array(vol3d,dtype=float),  np.array(int3d,dtype=float)

# evaluation of bistatic contributions
tic = timeit.default_timer()
tot3dplot, surf3dplot, vol3dplot, int3dplot = Rad(theta,phi, thetainc, phiinc)
toc = timeit.default_timer()
print('evaluation finished, it took ' + str(round((toc-tic)/60.,2)) + ' minutes')


#       dplot of total, volume, surface and interaction - contribution
Plots().linplot3d(theta, phi,  Itot3d = tot3dplot, Isurf3d = surf3dplot, Ivol3d = vol3dplot, Iint3d = int3dplot,  zoom = 1., surfmultip = 1.)

#       plot only volume contribution
#Plots().linplot3d(theta, phi,  Ivol3d = vol3dplot,  zoom = 1.2, surfmultip = 1.)
#       plot only surface contribution
#Plots().linplot3d(theta, phi,  Isurf3d = surf3dplot,  zoom = 1.2, surfmultip = 1.)
#       plot only interaction contribution
#Plots().linplot3d(theta, phi,  Iint3d = int3dplot,  zoom = 1.2, surfmultip = 1.)







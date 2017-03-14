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
inc = np.arange(1.,89.,1.)

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





tic = timeit.default_timer()
fn = None
# IMPORTANT: fn-coefficients must be evaluated with single values for incident- and exit angles !
R = RT1(I0, 0., 0., 0., 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
fn = R.fn  # store coefficients for faster iteration
toc = timeit.default_timer()
print('time for coefficient evaluation: ' + str(toc-tic))


# specification of measurement-geometry
t_0 = np.deg2rad(inc)
t_ex = t_0*1.
p_0 = np.ones_like(inc)*0.
p_ex = np.ones_like(inc)*0. + np.pi

tic = timeit.default_timer()
R = RT1(I0, t_0, t_ex, p_0, p_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
Itot, Isurf, Ivol, Iint = R.calc()
toc = timeit.default_timer()
print('evaluating print-values took ' + str(toc-tic))



# ---------------- EVALUATION OF HEMISPHERICAL REFLECTANCE ----------------

# plot the hemispherical reflectance associated with the chosen BRDF
#hemr = Plots().hemreflect(R=R, t_0_step = 5., simps_N = 500)
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
plot2 = Plots().logmono(inc, Itot = Itot, Isurf = Isurf, Ivol = Ivol, Iint = Iint, sig0=True, noint=True, label='Backscattering Coefficient'+'\n $\\omega$ = ' + str(R.RV.omega) + '$ \quad \\tau$ = ' + str(R.RV.tau))#, ylim=[-25,0])

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
t_ex,p_ex = np.linspace(np.deg2rad(1.),np.deg2rad(89.),25),np.linspace(0.,np.pi,25)
theta,phi = np.meshgrid(t_ex,p_ex)
tinc = np.ones_like(theta)*np.deg2rad(thetainc)
pinc = np.ones_like(phi)*np.deg2rad(phiinc)


# definition of function to evaluate bistatic contributions
# since this step might take a while an estimation of the calculation-time was included


def Rad(theta, phi, thetainc, phiinc):
    '''
    thetainc : float
               azimuth incidence angle
    phiinc : float
             polar incidence angle
    theta : numpy array
            azimuth exit angles    
    phi : numpy array
          polar exit angles    
    '''
    # pre-evaluation of fn-coefficients

    print('start of 3d coefficient evaluation')
    tic = timeit.default_timer()
    testfn = RT1(1., np.deg2rad(thetainc), np.deg2rad(45), phiinc, np.pi, RV=V, SRF=SRF, fn=None, geometry='fvfv').fn
    toc = timeit.default_timer()
    print('evaluation of 3d coefficients took ' + str(round((toc-tic)/60.,2)) + ' minutes')


    R3d = RT1(1., tinc, theta, pinc, phi, RV=V, SRF=SRF, fn=testfn, geometry='ffff')

    tic = timeit.default_timer()
    Itot, Isurf, Ivol, Iint = R3d.calc()
    toc = timeit.default_timer()
    print('evaluation of 3dplot print-values took ' + str(round((toc-tic)/60.,2)) + ' minutes')

    return Itot, Isurf, Ivol, Iint




# evaluation of bistatic contributions
tot3dplot, surf3dplot, vol3dplot, int3dplot = Rad(theta,phi, thetainc, phiinc)

#       dplot of total, volume, surface and interaction - contribution
plot3d = Plots().linplot3d(theta, phi,  Itot = tot3dplot, Isurf = surf3dplot, Ivol = vol3dplot, Iint = int3dplot,  zoom = 1., surfmultip = 1.)

#       plot only volume contribution
#Plots().linplot3d(theta, phi,  Ivol3d = vol3dplot,  zoom = 1.2, surfmultip = 1.)
#       plot only surface contribution
#Plots().linplot3d(theta, phi,  Isurf3d = surf3dplot,  zoom = 1.2, surfmultip = 1.)
#       plot only interaction contribution
#Plots().linplot3d(theta, phi,  Iint3d = int3dplot,  zoom = 1.2, surfmultip = 1.)
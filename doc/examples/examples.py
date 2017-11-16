"""
Reproduce examples like given in the paper
"""
# import sys
# sys.path.append('../..')

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

# start timer
tic = timeit.default_timer()

# set incident intensity to 1.
I0 = 1.

# EVALUATION OF BISTATIC PLOTS
# if set to true, 3dplots will be generated,
# otherwise the code is stopped after monostatic evaluation
bistaticplot = True

'''
possible choices for example = ?

1 : example 1 from the paper
2 : example 2 from the paper
3 : example using a linear-combination for p and the BRDF
'''
example = 3

# ----------- definition of examples -----------
if example == 1:
    # define incidence-angle range for generation of backscatter-plots
    inc = np.arange(1., 89., 1.)

    # Example 1
    V = Rayleigh(tau=0.55, omega=0.24)
    SRF = CosineLobe(ncoefs=15, i=6, a=[.28, 1., 1.])
    label = 'Example 1'
elif example == 2:
    # define incidence-angle range for generation of backscatter-plots
    inc = np.arange(1., 89., 1.)

    V = HenyeyGreenstein(tau=0.7, omega=0.3, t=0.7, ncoefs=20)
    SRF = CosineLobe(ncoefs=10, i=5)
    label = 'Example 2'
elif example == 3:
    # define incidence-angle range for generation of backscatter-plots
    inc = np.arange(1., 89., 1.)
    # list of volume-scattering phase-functions to be combined
    phasechoices = [
        # forward-scattering-peak
        HenyeyGreenstein(t=0.5, ncoefs=10, a=[-1., 1., 1.]),
        # backscattering-peak
        HenyeyGreenstein(t=-0.2, ncoefs=10, a=[-1., 1., 1.]),
        # downward-specular peak
        HenyeyGreenstein(t=-0.5, ncoefs=10, a=[1., 1., 1.]),
        # upward-specular peak
        HenyeyGreenstein(t=0.2, ncoefs=10, a=[1., 1., 1.]),
    ]
    # weighting-factors for the individual phase-functions
    Vweights = [.3, .3, .2, .2]

    # list of surface-BRDF-functions to be combined
    BRDFchoices = [  # backscattering peak
        HGsurface(ncoefs=10, t=-.4, a=[-.8, 1., 1.]),
        # specular peak
        HGsurface(ncoefs=10, t=.5, a=[.8, 1., 1.]),
        # isotropic scattering contribution
        Isotropic(NormBRDF=.1),
    ]
    # weighting-factors for the individual BRDF's
    BRDFweights = [.3, .4, .3]

    # generate correctly shaped arrays of the phase-functions and their
    # corresponding weighting-factors:
    Vchoices = [[Vweights[i],
                 phasechoices[i]] for i in range(len(phasechoices))]
    SRFchoices = [[BRDFweights[i],
                   BRDFchoices[i]] for i in range(len(BRDFchoices))]

    V = LinCombV(tau=0.6, omega=0.4, Vchoices=Vchoices)
    SRF = LinCombSRF(SRFchoices=SRFchoices, NormBRDF=0.1)

    label = 'Example 3'
elif example == 4:
    # functions are defined as in example 3, but now multiple measurements
    # are calculated simultaneously!

    # define incidence-angle range for generation of backscatter-plots

    # since the incidence-angle arrays do not have equal size, nan-values
    # are added to generate a rectangular incidence-angle array
    inc1 = np.arange(1., 89., 1.)
    inc2 = np.full_like(inc1, np.nan)

    for i in range(49):
        inc2[i] = np.arange(1., 50., 1.)[i]

    inc = [inc1, inc2]

    # list of volume-scattering phase-functions to be combined
    phasechoices = [
        # forward-scattering-peak
        HenyeyGreenstein(t=0.5, ncoefs=10, a=[-1., 1., 1.]),
        # backscattering-peak
        HenyeyGreenstein(t=-0.2, ncoefs=10, a=[-1., 1., 1.]),
        # downward-specular peak
        HenyeyGreenstein(t=-0.5, ncoefs=10, a=[1., 1., 1.]),
        # upward-specular peak
        HenyeyGreenstein(t=0.2, ncoefs=10, a=[1., 1., 1.]),
    ]
    # weighting-factors for the individual phase-functions
    Vweights = [.3, .3, .2, .2]

    # list of surface-BRDF-functions to be combined
    BRDFchoices = [  # backscattering peak
        HGsurface(ncoefs=10, t=-.4, a=[-.8, 1., 1.]),
        # specular peak
        HGsurface(ncoefs=10, t=.5, a=[.8, 1., 1.]),
        # isotropic scattering contribution
        Isotropic(NormBRDF=.1),
    ]
    # weighting-factors for the individual BRDF's
    BRDFweights = [.3, .4, .3]

    # generate correctly shaped arrays of the phase-functions and their
    # corresponding weighting-factors:
    Vchoices = [[Vweights[i],
                 phasechoices[i]] for i in range(len(phasechoices))]
    SRFchoices = [[BRDFweights[i],
                   BRDFchoices[i]] for i in range(len(BRDFchoices))]

    # define input-parameter arrays
    tau = np.array([.4, .5])
    omega = np.array([.4, .24])
    NormBRDF = np.array([.2, .3])

    V = LinCombV(tau=tau, omega=omega, Vchoices=Vchoices)
    SRF = LinCombSRF(SRFchoices=SRFchoices, NormBRDF=NormBRDF)

    # choose the number of the result to be plotted if multiple results have
    # been evaluated. alternatively choose   Nres = 'all'  to plot all results
    Nres = 'all'

    label = 'Example 4'
else:
    assert False, 'Choose an existing example or specify V and SRF explicitly'

#%%


# specification of measurement-geometry
t_0 = np.deg2rad(inc)
t_ex = t_0 * 1.
p_0 = np.ones_like(t_0) * 0.
p_ex = np.ones_like(t_0) * 0. + np.pi

tic = timeit.default_timer()
R = RT1(I0, t_0, t_ex, p_0, p_ex, V=V, SRF=SRF,
        geometry='mono', lambda_backend='cse_symengine_sympy')

fn = R.fn  # evaluate and store coefficients for faster iteration
_fnevals = R._fnevals  # evaluate and  store coefficients for faster iteration


Itot, Isurf, Ivol, Iint = R.calc()
toc = timeit.default_timer()
print('evaluating print-values took ' + str(toc - tic))


if len(Itot.shape) > 1:
    if Nres == 'all':
        for Nres in range(len(Itot)):
            incp = inc[Nres]
            Itotp = Itot[Nres]
            Isurfp = Isurf[Nres]
            Ivolp = Ivol[Nres]
            Iintp = Iint[Nres]
            label = ('Backscattering Coefficient' + '\n $\\omega$ = ' +
                     str(R.V.omega[Nres][0]) + '$ \quad \\tau$ = ' +
                     str(R.V.tau[Nres][0]))
            plot2 = Plots().logmono(incp, Itot=Itotp, Isurf=Isurfp,
                                    Ivol=Ivolp, Iint=Iintp,
                                    sig0=True, noint=True, label=label)
    else:
        incp = inc[Nres]
        Itotp = Itot[Nres]
        Isurfp = Isurf[Nres]
        Ivolp = Ivol[Nres]
        Iintp = Iint[Nres]
        label = ('Backscattering Coefficient' + '\n $\\omega$ = ' +
                 str(R.V.omega[Nres][0]) + '$ \quad \\tau$ = ' +
                 str(R.V.tau[Nres][0]))
else:
    incp, Itotp, Isurfp, Ivolp, Iintp = inc, Itot, Isurf, Ivol, Iint
    label = ('Backscattering Coefficient' + '\n $\\omega$ = ' +
             str(R.V.omega) + '$ \quad \\tau$ = ' + str(R.V.tau))

# ---------------- EVALUATION OF HEMISPHERICAL REFLECTANCE ----------------

# plot the hemispherical reflectance associated with the chosen BRDF
# hemr = Plots().hemreflect(R=R, t_0_step=5., simps_N=500)

# ---------------- GENERATION OF POLAR-PLOTS ----------------
#       plot both p and the BRDF in a single plot
# plot1 = Plots().polarplot(R, incp=list(np.linspace(0, 120, 5)),
#              incBRDF=list(np.linspace(0, 90, 5)), pmultip=1.5)

#       plot only the BRDF
# Plots().polarplot(SRF = R.SRF)

#       plot only p
# Plots().polarplot(V = R.V)

#       plot more than one phase-function simultaneously
#       example: henyey-greenstein phase function for various param. chocies
#
# hg = Plots().polarplot(V=[
#    HenyeyGreenstein(tau=0.7,
#                     omega=0.3,
#                     t=tt,
#                     ncoefs=10
#                     ) for tt in [.1, .2, .3, .4, .5, .6]],
#    pmultip=1.,
#    incp=[45],
#    plabel='Henyey Greenstein Phase Function',
#    paprox=False)

#       example: lafortune-lobes
# ll = Plots().polarplot(SRF = [CosineLobe(ncoefs=10, i=5, a=[.8,1.,1.])],
#            BRDFlabel = 'Lafortune-Lobe BRDF', BRDFaprox = False)
#

# ---------------- GENERATION OF BACKSCATTER PLOTS ----------------
#       plot backscattered intensity and fractional contributions

if len(Itot.shape) == 1:
    plot2 = Plots().logmono(incp, Itot=Itotp, Isurf=Isurfp,
                            Ivol=Ivolp, Iint=Iintp, sig0=True, noint=True,
                            label=label, ylim=[-20., 0.])  # , ylim=[-25,0])

#       plot only backscattering coefficient without fractions
# Plots().logmono(inc, Itot=Itot, Isurf=Isurf, Ivol=Ivol, Iint=Iint,
#                sig0=True, fractions=False)

#       plot only Itot and Ivol
# Plots().logmono(inc,Itot = Itot, Ivol = Ivol, fractions = False)

# ---------------- GENERATION OF 3D PLOTS ----------------

assert bistaticplot, 'Manual stop before bistatic evaluation'

tau = 0.5

# define incidence-angle range for generation of backscatter-plots
inc = np.arange(1., 89., 1.)
# list of volume-scattering phase-functions to be combined
phasechoices = [
    # forward-scattering-peak
    HenyeyGreenstein(t=0.5, ncoefs=10, a=[-1., 1., 1.]),
    # backscattering-peak
    HenyeyGreenstein(t=-0.2, ncoefs=10, a=[-1., 1., 1.]),
    # downward-specular peak
    HenyeyGreenstein(t=-0.5, ncoefs=10, a=[1., 1., 1.]),
    # upward-specular peak
    HenyeyGreenstein(t=0.2, ncoefs=10, a=[1., 1., 1.]),
]
# weighting-factors for the individual phase-functions
Vweights = [.3, .3, .2, .2]

# list of surface-BRDF-functions to be combined
BRDFchoices = [
    # backscattering peak
    HGsurface(ncoefs=10, t=-.4, a=[-.8, 1., 1.]),
    # specular peak
    HGsurface(ncoefs=10, t=.5, a=[.8, 1., 1.]),
    # isotropic scattering contribution
    Isotropic(NormBRDF=.1),
]

# weighting-factors for the individual BRDF's
BRDFweights = [.3, .4, .3]

# generate correctly shaped arrays of the phase-functions and their
# corresponding weighting-factors:
Vchoices = [[Vweights[i],
             phasechoices[i]] for i in range(len(phasechoices))]
SRFchoices = [[BRDFweights[i],
               BRDFchoices[i]] for i in range(len(BRDFchoices))]

V = LinCombV(tau=tau, omega=0.4, Vchoices=Vchoices)
SRF = LinCombSRF(SRFchoices=SRFchoices, NormBRDF=0.1)

# definition of incidence-angles
thetainc = 55.
phiinc = 0.

# define plot-range
t_ex, p_ex = (np.linspace(np.deg2rad(1.), np.deg2rad(89.), 25),
              np.linspace(0., np.pi, 25))

theta, phi = np.meshgrid(t_ex, p_ex)
tinc = np.ones_like(theta) * np.deg2rad(thetainc)
pinc = np.ones_like(phi) * np.deg2rad(phiinc)


# definition of function to evaluate bistatic contributions (since this step
# might take a while an estimation of the calculation-time was included)

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
    Rfn = RT1(1., np.deg2rad(thetainc), np.deg2rad(45), phiinc,
              np.pi, V=V, SRF=SRF, geometry='fvfv',
              lambda_backend='cse_symengine_sympy')

    _fnevals = Rfn._fnevals
    # store also fn-coefficients to avoid re-calculation
    # (alternatively one can provide any value but None to avoid calculation)
    fn = Rfn.fn

    toc = timeit.default_timer()
    print('evaluation of 3d coefficients took ' +
          str(round((toc - tic) / 60., 2)) + ' minutes')

    R3d = RT1(1., tinc, theta, pinc, phi, V=V, SRF=SRF,
              fn_input = fn, _fnevals_input=_fnevals, geometry='fvfv')

    tic = timeit.default_timer()
    Itot, Isurf, Ivol, Iint = R3d.calc()

    Itot = 4. * np.pi * np.cos(theta) * Itot
    Isurf = 4. * np.pi * np.cos(theta) * Isurf
    Ivol = 4. * np.pi * np.cos(theta) * Ivol
    Iint = 4. * np.pi * np.cos(theta) * Iint

    toc = timeit.default_timer()
    print('evaluation of 3dplot print-values took ' +
          str(round((toc - tic) / 60., 2)) + ' minutes')

    return Itot, Isurf, Ivol, Iint


# evaluation of bistatic contributions
tot3dplot, surf3dplot, vol3dplot, int3dplot = Rad(theta, phi, thetainc, phiinc)

#       dplot of total, volume, surface and interaction - contribution
plot3d = Plots().linplot3d(theta, phi,
                           Itot=tot3dplot,
                           Isurf=surf3dplot,
                           Ivol=vol3dplot,
                           Iint=int3dplot,
                           zoom=1., surfmultip=1.)

#       plot only volume contribution
# Plots().linplot3d(theta, phi,  Ivol = vol3dplot,
#     zoom = 1.2, surfmultip = 1.)
#       plot only surface contribution
# Plots().linplot3d(theta, phi,  Isurf = surf3dplot,
#     zoom = 1.2, surfmultip = 1.)
#       plot only interaction contribution
# Plots().linplot3d(theta, phi,  Iint = int3dplot,
#     zoom = 1.2, surfmultip = 1.)

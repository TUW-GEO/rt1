# import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import random
import timeit

import matplotlib as mpl
import matplotlib.pyplot as plt

from rt1.rt1 import RT1
from rt1.rtfits import Fits

from rt1.volume import Rayleigh
# from rt1.volume import HenyeyGreenstein
# from rt1.volume import LinCombV

from rt1.surface import HenyeyGreenstein as HGsurface
# from rt1.surface import CosineLobe
# from rt1.surface import Isotropic
# from rt1.surface import LinCombSRF

'''
------------------------------------------------------------------------------
generation of data

each dataset has:
    - varying incidence-angle range
    - varying number of measurements
    - random values for omega, tau, NormBRDF and BRDF-asymmetry-parameter
    - random noise added
    - a random number (Neq) of measurements is chosen to have equal tau-value

fit-specifications:
    - only a single value for omega is fitted
      (ideally resulting in the mean-value of the exact omega-values)
    - since (Neq) measurements have equal tau-value, only a single tau-value
      is fitted to the (Neq) datasets
    - the boundary-conditions are set as (min, max) of the values used to
      generate the dataset
    - the start-values are chosen as random numbers within the boundaries

NOTICE: since the datasets are generated based on randomly generated
        parameters, the time needed to perform the fit can be very different
        for each run of the code.
'''

# ----------------------------------------------------------------------------
# ------------------ PARAMETERS FOR DATA-GENERATION --------------------------

# choose between intensity and sigma_0 datasets
sig0 = True
# choose between linear and dB datasets
dB = False
# number of measurements to be generated
Nmeasurements = 30
# minimal and maximal incidence-angle in degree
mininc = 25
maxinc = 65
# minimum and maximum number of incidence-angles
minincnum = 20
maxincnum = 50
# minimal and maximal values for omega
omin, omax = 0.35, 0.4
# since only a single omega-value is fitted, values should be kept
# close to avoid convergence-issues within the fit-procedure
# minimal and maximal values for tau
taumin, taumax = 0.1, 1.25
# minimal and maximal values for NormBRDF
rmin, rmax = 0.1, 0.5
# minimal and maximal values for t
tmin, tmax = 0.0001, 0.5

# ----------------------------------------------------------------------------
# ------------------------- DATA-GENERATION ----------------------------------

# generate N arrays of incidence-angles that contain maxincnum
# values between mininc and maxinc
inc = np.array([np.deg2rad(np.linspace(mininc, maxinc, maxincnum))]
               * Nmeasurements)

# generate random samples of parameters
omegadata = np.random.uniform(low=omin, high=omax, size=(Nmeasurements,))
taudata = np.random.uniform(low=taumin, high=taumax, size=(Nmeasurements,))
rdata = np.random.uniform(low=rmin, high=rmax, size=(Nmeasurements,))
tdata = np.random.uniform(low=tmin, high=tmax, size=(Nmeasurements,))


# set tau of Neq measurements to be equal and fit only a single parameter
# for tau to all datasets

# choose a random number of equal datasets
if Nmeasurements == 1:
    Neq = 0
    equal_tau_selects = [0]
else:
    Neq = int(Nmeasurements / 5)
    equal_tau_selects = np.random.choice(range(Nmeasurements),
                                         size=Neq + 1, replace=False)

for i in equal_tau_selects:
    # for i in [1,3,5,7,9,11,14,16,17]:
    taudata[i] = taudata[equal_tau_selects[0]]


# define model that is used to generate the data
# the choices for tau, omega and NormBRDF have no effect on the generated data
# since they will be changed to randomly generated values!

V_data = Rayleigh(tau=0.1, omega=0.1)
SRF_data = HGsurface(ncoefs=15, t=sp.var('t_data'), a=[1., 1., 1.])

# setup rt1-object
# (since the fn-coefficients must still be calculated, one must
#  specify the arrays for the parameters afterwards)
R_data = RT1(1., 0., 0., 0., 0., RV=V_data, SRF=SRF_data,
             fn=None, geometry='mono', param_dict={'t_data': .5},
             lambda_backend='cse_symengine_sympy')

# specify parameters and incidence-angles
R_data.t_0 = inc
R_data.p_0 = np.zeros_like(inc)

R_data.RV.omega = omegadata
R_data.RV.tau = taudata
R_data.SRF.NormBRDF = rdata
R_data.param_dict = {'t_data': tdata[:, np.newaxis]}

# calculate the data and add some random noise
data = R_data.calc()[0]
noise = np.random.uniform(low=-np.max(data) / 50.,
                          high=np.max(data) / 50., size=data.shape)
data = data + noise

if sig0 is True:
    # convert the calculated results do sigma0
    signorm = 4. * np.pi * np.cos(inc)
    data = signorm * data

if dB is True:
    # convert the calculated results to dB
    data = 10. * np.log10(data)

# define the mask for selecting non-rectangular arrays of data
# (this is done to show that fitting also works for non-rectangular datasets)
inc_lengths = np.random.randint(minincnum, maxincnum, Nmeasurements)
selects = []
selects = selects + [random.sample(range(maxincnum), inc_lengths[i])
                     for i in range(Nmeasurements)]

# generate dataset of the shape [ [inc_0, data_0], [inc_1, data_1], ...]
# with inc_i and data_i being arrays of varying length
dataset = []
for i, row in enumerate(inc):
    dataset = dataset + [[inc[i][selects[i]], data[i][selects[i]]]]

# %%
# ----------------------------------------------------------------------------
# ------------------------------- FITTING ------------------------------------

tic = timeit.default_timer()
# initialize fit-class (with flag dB = dB and sig0 = sig0 to indicate whether
# the data is given as Intensity or sigma_0 values in dB or linear scale.
testfit = Fits(sig0=sig0, dB=dB)

# define sympy-symbols for definition of V and SRF
t1 = sp.Symbol('t1')

V = Rayleigh(omega=0.1, tau=0.1)
# values for NormBRDF are set to known values (i.e. rdata)
SRF = HGsurface(ncoefs=15, t=t1, NormBRDF=rdata, a=[1., 1., 1.])

# select start numbers within the boundaries as sart-values
ostart = np.mean([omin, omax])
tstart = np.mean([tmin, tmax])
taustart = np.mean([taumin, taumax])

# define which parameter should be fitted
param_dict = {'tau': [taustart] * (Nmeasurements - Neq),
              'omega': ostart,
              't1': [tstart] * (Nmeasurements)}


# optionally define fixed parameters
fixed_dict = {'NormBRDF': rdata}

# define boundary-conditions
bounds_dict = {'t1': ([tmin] * (Nmeasurements),
                      [tmax] * (Nmeasurements)),
               'tau': ([taumin] * (Nmeasurements - Neq),
                       [taumax] * (Nmeasurements - Neq)),
               'omega': ([omin],
                         [omax])}

# setup param_dyn_dict
param_dyn_dict = {}
for key in param_dict:
    param_dyn_dict[key] = np.linspace(1,
                                      len(np.atleast_1d(param_dict[key])),
                                      Nmeasurements)

# fit only a single parameter to the datasets that have equal tau
for i in equal_tau_selects:
    param_dyn_dict['tau'][i] = param_dyn_dict['tau'][equal_tau_selects[0]]

# provide true-values for comparison of fitted results
truevals = {'tau': taudata,
            'omega': omegadata,
            't1': tdata,
            }

tic = timeit.default_timer()
# perform fit
fit = testfit.monofit(V=V, SRF=SRF, dataset=dataset,
                      param_dict=param_dict,
                      bounds_dict=bounds_dict,
                      fixed_dict=fixed_dict,
                      param_dyn_dict=param_dyn_dict,
                      verbose=2,
                      ftol=1.e-8, xtol=1.e-8, gtol=1.e-8,
                      x_scale='jac',
                      lambda_backend='cse_symengine_sympy'
                      # loss='cauchy'
                      )
toc = timeit.default_timer()
print('performing the fit took', round(toc-tic, 2), 'seconds')
# %%

# ----------------------------------------------------------------------------
# ------------------------------- RESULTS ------------------------------------

# print results
asdf = testfit.printresults(fit, truevals=truevals, startvals=False)

# print errors
bsdf = testfit.printerr(fit)

# print scatterplot using additionally a color-coding that corresponds to
# the incidence-angles of the datapoints (just to show the possibilities)
csdf, r2 = testfit.printscatter(fit, pointsize=4,
                                c=fit[3][~fit[4]], cmap='coolwarm',
                                regression=True)

# generate colormap after plot has been created
norm = mpl.colors.Normalize(vmin=np.min(fit[3][~fit[4]]),
                            vmax=np.max(fit[3][~fit[4]]))
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])

cbticks = np.linspace(np.min(fit[3][~fit[4]]), np.max(fit[3][~fit[4]]), 10)
cb = csdf.colorbar(sm, ax=csdf.axes[0], ticks=cbticks)
cb.ax.set_yticklabels(np.round(np.rad2deg(cbticks), 1))
cb.ax.set_title('     $\\theta_0 [deg]$ \n', fontsize=10)


# print result of specific measurements with underlying hexbin-plot
if Nmeasurements == 1:
    selects = [0]
else:
    Nsingle = int(Nmeasurements / 5)
    if Nsingle == 0:
        Nsingle = 1
    selects = np.random.choice(range(Nmeasurements),
                               size=Nsingle, replace=False)

dsdf = testfit.printsingle(fit, measurements=selects,
                           hexbinQ=True, hexbinargs={'gridsize': 20})

# print result of specific measurements with underlying hexbin-plot
esdf = testfit.printseries(fit)

toc = timeit.default_timer()
print('it took ' + str(toc - tic))

tauerr = taudata - fit[6]['tau']
omegaerr = omegadata - fit[6]['omega']
terr = tdata - fit[6]['t1']

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title('percentiles of error')
ax.plot(np.percentile(np.abs(terr), np.arange(0, 100, 1)),
        label='t', marker='.')
ax.plot(np.percentile(np.abs(tauerr), np.arange(0, 100, 1)),
        label='tau', marker='.')
ax.plot(np.percentile(np.abs(omegaerr), np.arange(0, 100, 1)),
        label='omega', marker='.')
ax.legend()

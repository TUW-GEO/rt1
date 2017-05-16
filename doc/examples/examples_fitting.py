#import matplotlib.pyplot as plt
import numpy as np
import random

from rt1.rt1 import RT1

from rt1.rtplots import Plots
from rt1.rtfits import Fits

from rt1.volume import Rayleigh
from rt1.volume import HenyeyGreenstein
from rt1.volume import LinCombV

from rt1.surface import CosineLobe
from rt1.surface import HenyeyGreenstein as HGsurface
from rt1.surface import Isotropic
from rt1.surface import LinCombSRF


'''
------------------------------------------------------------------------------
generation of data

each dataset has:
    - varying incidence-angle range
    - varying number of measurements
    - random values for omega, tau and NormBRDF
    - random noise added


NOTICE: since the datasets are generated based on randomly generated
        parameters, the time needed to perform the fit can be very different
        for each run of the code.
'''

# ----------------------------------------------------------------------------
# ------------------ PARAMETERS FOR DATA-GENERATION --------------------------

sig0 = True            # choose between intensity and sigma_0 datasets
dB = True              # choose between linear and dB datasets

Nmeasurements = 5      # number of measurements to be generated

mininc = 25             # minimal incidence-angle in degree
maxinc = 65             # maximal incidence-angle in degree

minincnum = 10          # minimum number of incidence-angles
maxincnum = 40          # maximum number of incidence-angles

omin, omax = 0.2, 0.5   # minimal and maximal values for omega
tmin, tmax = 0.1, 0.85  # minimal and maximal values for tau
rmin, rmax = 0.1, 0.5   # minimal and maximal values for NormBRDF

noiserate = 50.         # scale of noise added to the data
# noise-values hereby are random numbers in the range
#   (- noise_max, noise_max)
# where noise_max is given by:
#   noise_max = max(data) / noiserate

# define model that is used to generate the data
# the choices for tau, omega and NormBRDF have no effect on the generated data
# since they will be changed to randomly generated values!
V = Rayleigh(tau=0.1, omega=0.1)
SRF = CosineLobe(ncoefs=8, i=5)


# ----------------------------------------------------------------------------
# ------------------------- DATA-GENERATION ----------------------------------

# define function to generate the data
def fun(V, SRF, params, inc, sig0, noiserate, fn=None):
    V.omega = params[0]
    V.tau = params[1]
    SRF.NormBRDF = params[2]

    # calculate fn-coefficients if they are not provided explicitly
    if fn is None:
        R = RT1(1., 0., 0., 0., 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
        fn = R.fn

    R = RT1(1., inc, inc, np.ones_like(inc) * 0., np.ones_like(inc) * 0.,
            RV=V, SRF=SRF, fn=fn, geometry='mono')

    data = R.calc()[0]
    dataeps = np.random.randn(*data.shape) * np.max(data) / noiserate
    data = data + dataeps

    if sig0 is True:
        # convert the calculated results do sigma0 in dB
        signorm = 4. * np.pi * np.cos(inc)
        data = signorm * data

    if dB is True:
        data = 10. * np.log10(data)

    return data


# generate N arrays of incidence-angles that contain maxincnum
# values between mininc and maxinc
inc = np.array([np.deg2rad(np.linspace(mininc, maxinc, maxincnum))]
               * Nmeasurements)

# generate random samples of parameters
omegadata = np.random.uniform(low=omin, high=omax, size=(Nmeasurements,))
taudata = np.random.uniform(low=tmin, high=tmax, size=(Nmeasurements,))
rdata = np.random.uniform(low=rmin, high=rmax, size=(Nmeasurements,))

# define the mask for selecting non-rectangular arrays of data
# (this is done to show that fitting also works for non-rectangular datasets)
inc_lengths = np.random.randint(minincnum, maxincnum, Nmeasurements)
selects = []
selects = selects + [random.sample(range(maxincnum), inc_lengths[i])
                     for i in range(Nmeasurements)]

# calculate data
data = fun(V=V, SRF=SRF, params=[omegadata, taudata, rdata],
           inc=inc, sig0=sig0, noiserate=noiserate)

# generate dataset of the shape [ [inc_0, data_0], [inc_1, data_1], ...]
# with inc_i and data_i being arrays of varying length
dataset = []
for i, row in enumerate(inc):
    dataset = dataset + [[inc[i][selects[i]], data[i][selects[i]]]]


# --------------------- END OF DATA-GENERATION -------------------------------
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ------------------------------- FITTING ------------------------------------
# initialize fit-class (with flag dB = dB and sig0 = sig0 to indicate whether
# the data is given as Intensity or sigma_0 values in dB or linear scale.
testfit = Fits(sig0=sig0, dB=dB)

# define the model (equal to the model defined for data-generation)
V = Rayleigh(tau=0.1, omega=0.1)
SRF = CosineLobe(ncoefs=8, i=5)

#       plot the phase-functions used in the model
#modelplot = Plots().polarplot(SRF=SRF, V = V)
#       plot the hemispherical reflectance that results from the SRF definition
#modelhemreflect = Plots().hemreflect(SRF=SRF)


# perform a fit with the V and SRF functions used for data-generation
fit0 = testfit.monofit(V=V, SRF=SRF, dataset=dataset,
                       verbose=2,
                       ftol=1.e-5, xtol=1.e-5, gtol=1.e-5,
                       x_scale='jac',)

# define true-values for comparison
truevals = np.concatenate([omegadata, taudata, rdata])
# print fit-results
fit_figure0 = testfit.printresults(fit0, truevals=truevals)
# print residuals
fit_err_figure0 = testfit.printerr(fit0)


assert False, 'Manual stop before second fit'

# ---------- fit a different model -------------------------------------------
# define new fit-model
V = HenyeyGreenstein(t=0.01, ncoefs=5, tau=0.1, omega=0.1)

# the parameter overallnorm is introduced in order to have NormBRDF
# within the range (0,1).
# Notice that this just scales NormBRDF-values to fit the range as set in the
# boundaries for the fit, it does not alter the dynamics of NormBRDF-values!
overallnorm = .05
SRF = LinCombSRF([  # specular contribution
    [overallnorm, HGsurface(t=0.5, ncoefs=10)]
])

#       plot the phase-functions used in the new model
#newmodelplot = Plots().polarplot(SRF=SRF, V = V)
#       plot the hem. reflectance that results from the new SRF definition
#newmodelhemreflect = Plots().hemreflect(SRF=SRF)

# set boundaries for fit
bounds = (
    [0.] * len(dataset) + [0.] * len(dataset) + [0.] * len(dataset),
    [1.] * len(dataset) + [1.] * len(dataset) + [1.] * len(dataset)
)

# set start-values for fit
startvals = np.concatenate((
    np.linspace(1, 1, len(dataset)) * 0.3,
    np.linspace(1, 1, len(dataset)) * 0.3,
    np.linspace(1, 1, len(dataset)) * 0.3
))

# perform a fit with the defined V and SRF functions
fit = testfit.monofit(V=V, SRF=SRF, dataset=dataset,
                      verbose=2,
                      ftol=1.e-4, xtol=1.e-4, gtol=1.e-4,
                      bounds=bounds,
                      startvals=startvals,
                      x_scale='jac',)

# define true-values for comparison
truevals = np.concatenate([omegadata, taudata, rdata])
# print fit-results
fit_figure = testfit.printresults(fit, truevals=truevals)
# print residuals
fit_err_figure = testfit.printerr(fit)

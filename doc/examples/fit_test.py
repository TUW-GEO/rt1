#import matplotlib.pyplot as plt
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


import matplotlib.pyplot as plt
from scipy.optimize import least_squares


'''
setup model
'''

NormBRDF = .5
tau = .5
omega = .5
# set incident intensity to 1.
I0 = 1.

example = 3

# ----------- definition of examples -----------
if example == 1:
    # Example 1
    V = Rayleigh(tau=tau, omega=omega)
    SRF = CosineLobe(ncoefs=10, i=5, NormBRDF=NormBRDF)
    label = 'Example 1'
elif example == 2:
    V = HenyeyGreenstein(tau=tau, omega=omega, t=0.7, ncoefs=20)
    SRF = CosineLobe(ncoefs=10, i=5, NormBRDF=NormBRDF)
    label = 'Example 2'
elif example == 3:
    # list of volume-scattering phase-functions to be combined
    phasechoices = [HenyeyGreenstein(t=0.5, ncoefs=8, a=[-1., 1., 1.]),  # forward-scattering-peak
                    HenyeyGreenstein(t=-0.2, ncoefs=8, a=[-1., 1., 1.]),  # backscattering-peak
                    HenyeyGreenstein(t=-0.5, ncoefs=8, a=[1., 1., 1.]),  # downward-specular peak
                    HenyeyGreenstein(t=0.2, ncoefs=8, a=[1., 1., 1.]),  # upward-specular peak
                    ]
    # weighting-factors for the individual phase-functions
    Vweights = [.3, .3, .2, .2]

    # list of surface-BRDF-functions to be combined
    BRDFchoices = [HGsurface(ncoefs=8, t=-.4, a=[-.8, 1., 1.]),         # backscattering peak
                   HGsurface(ncoefs=8, t=.5, a=[.8, 1., 1.]),         # specular peak
                   Isotropic(),        # isotropic scattering contribution
                   ]

    # weighting-factors for the individual BRDF's
    BRDFweights = [.3, .4, .3]

    # generate correctly shaped arrays of the phase-functions and their corresponding weighting-factors:
    Vchoices = [[Vweights[i], phasechoices[i]] for i in range(len(phasechoices))]
    SRFchoices = [[BRDFweights[i], BRDFchoices[i]] for i in range(len(BRDFchoices))]

    V = LinCombV(tau=tau, omega=omega, Vchoices=Vchoices)

    SRF = LinCombSRF(SRFchoices=SRFchoices, NormBRDF=NormBRDF)
else:
    assert False, 'Choose an existing example-number or specify V and SRF explicitly'


'''
-----------------------------
pre-calculate fn-coefficients
-----------------------------
'''
fn = None
# IMPORTANT: fn-coefficients must be evaluated with single values for incident- and exit angles !
R = RT1(I0, 0., 0., 0., 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
fn = R.fn  # store coefficients for faster iteration


'''
----------------------------------------------------------------------------
define fit for multiple measurements

N measurements are generated with random omega, tau and NormBRDF,
a random noise is added and then all N measurements are fitted
simultaneously using an analytic estimate of the jacobian provided by the 
surface and volume-term (the contribution of the interaction-term to the 
jacobian is currently neglected)
----------------------------------------------------------------------------
'''


# number of measurements that are processed simultaneously
Nmeasurements = 5

# objective function for the fit


def fun(params, inc, data):
    params = params
    V.omega = params[0:int(len(params) / 3)]
    V.tau = params[int(len(params) / 3):int(2 * len(params) / 3)]
    SRF.NormBRDF = params[int(2 * len(params) / 3):int(len(params))]
    R = RT1(I0, inc, inc, np.ones_like(inc) * 0., np.ones_like(inc) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')

    errs = R.calc()[0]
    errs = np.concatenate(errs) - data

    return errs

# function to evaluate the jacobian


def dfun(params, x, data):
    params = params
    V.omega = params[0:int(len(params) / 3)]
    V.tau = params[int(len(params) / 3):int(2 * len(params) / 3)]
    SRF.NormBRDF = params[int(2 * len(params) / 3):int(len(params))]

    #x = np.concatenate([x]*(int(len(params)/3)))

    R = RT1(I0, x, x, np.ones_like(x) * 0., np.ones_like(x) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')

    jac = np.concatenate(R.jacobian()).T
    return jac


'''
---------------------------------------------
generation of random data with a bit of noise
---------------------------------------------
'''

mininc = 25  # minimal incidence-angle
maxinc = 65  # maximal incidence-angle
minincsep = 15  # minimal incidence-angle separation
incnum = 20  # number of incidence-angles

omin, omax = 0.2, 0.5  # minimal and maximal values for omega
tmin, tmax = 0.1, 0.85  # minimal and maximal values for tau
rmin, rmax = 0.1, 0.5  # minimal and maximal values for NormBRDF

noiserate = 50.  # scale of noise added to the data (   data = data + max(data)/noiserate   )


# generate random samples of incidence-angle separation
inc = []
for i in range(Nmeasurements):
    x = np.random.randint(mininc, maxinc - minincsep - 1)
    y = np.random.randint(x + minincsep, maxinc)
    #length = [5,100][i]
    length = np.random.randint(20, 50)
    inc = inc + [np.linspace(x, y, length)]


# rectangularize numpy array by adding nan-values
# (this is necessary because numpy can only deal with rectangular arrays)
maxLen = np.max(np.array([len(j) for i, j in enumerate(inc)]))
for i, j in enumerate(inc):
    if len(j) < maxLen:
        inc[i] = np.append(j, np.tile(np.nan, maxLen - len(j)))


inc = np.deg2rad(np.array(inc))


# generate random samples of parameters
omegadata = np.random.uniform(low=omin, high=omax, size=(Nmeasurements,))
taudata = np.random.uniform(low=tmin, high=tmax, size=(Nmeasurements,))
rdata = np.random.uniform(low=rmin, high=rmax, size=(Nmeasurements,))

# generate data based on randomly generated parameters
xdata = np.concatenate([omegadata, taudata, rdata])
data = fun(xdata, inc, 0.)

# add noise
dataeps = np.random.randn(len(data)) * max(data) / noiserate
data = data + dataeps


# set start-values
x02 = np.concatenate((np.ones_like(omegadata) * 0.1,
                      np.ones_like(omegadata) * 0.2,
                      np.ones_like(omegadata) * 0.3))


# prepare data to avoid nan-values
# TODO numpy only supports rectangular arrays, least_squares does not
#      support nan-values, and also it doesn't work with masked arrays...
#      currently the problem for missing values is adressed by repeating
#      the values from the nearest available neighbour....

#      this results in an inhomogeneous treatment of the measurements!
#      -> mehtod only implemented to see if it's actually working...
#      -> the inhomogeneous weighting of the duplicates is now corrected
#         using a weighting-matrix

weights = np.ones_like(data)

i = 0

while i < len(data):
    if np.isnan(data[i]):
        j = 0
        while np.isnan(data[i + j]):
            data[i + j] = data[i + j - 1]
            j = j + 1
            if i + j >= len(data):
                break
        # the weights are calculated as one over the square-root of the number of repetitions
        # in order to cancel out the repeated measurements in the sum of SQUARED residuals
        weights[i - 1: i + j] = 1. / np.sqrt(float(j + 1))
    i = i + 1


for i, j in enumerate(inc):
    for k, l in enumerate(j):
        if np.isnan(l):
            inc[i][k] = inc[i][k - 1]

# define a new function that corrects for non-rectangular arrays
# by weighting the repeated values with respect to the number of repetitions


def funnew(params, inc, data):
    return weights * fun(params, inc, data)


# for i, j in enumerate(data):
#    if np.isnan(j):
#        data[i] = data[i - 1]
#
# for i, j in enumerate(inc):
#    for k, l in enumerate(j):
#        if np.isnan(l):
#            inc[i][k] = inc[i][k - 1]


'''
-------------------
perform actual fits
-------------------
'''

tic = timeit.default_timer()

# fit with incorrect weighting of manually added duplicates
#res_lsq1 = least_squares(fun, x02, args=(inc, data), verbose=2, bounds=([0.] * len(x02), [1.] * len(x02)), jac=dfun, xtol=1.e-4, ftol=1.e-4, gtol=1.e-4)

# fit with correct weighting of duplicates
res_lsq2 = least_squares(funnew, x02, args=(inc, data), verbose=2, bounds=([0.1] * len(x02), [1.] * len(x02)), jac=dfun, xtol=1.e-4, ftol=1.e-4, gtol=1.e-4)

toc = timeit.default_timer()
print('it took ' + str(toc - tic) + ' seconds')

''' 
-------------
print results
-------------
'''

# function to evaluate the model on the estimated parameters


def fun(x, t, y):
    V.omega = x[0]
    V.tau = x[1]
    SRF.NormBRDF = x[2]
    R = RT1(I0, t, t, np.ones_like(t) * 0., np.ones_like(t) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')

    return R.calc()[0] - y


# prepare the data that has been used in the fit for plotting
fitdata = np.array(np.split(data, Nmeasurements))


fig = plt.figure()

ax = fig.add_subplot(211)

for i, j in enumerate(fitdata):
    ax.plot(inc[i], j, '.')

plt.gca().set_prop_cycle(None)


omegafits = res_lsq2.x[0:int(len(res_lsq2.x) / 3)]
taufits = res_lsq2.x[int(len(res_lsq2.x) / 3):int(2 * len(res_lsq2.x) / 3)]
rfits = res_lsq2.x[int(2 * len(res_lsq2.x) / 3):int(len(res_lsq2.x))]

incplot = np.array([np.deg2rad(np.linspace(1., 89., 100))] * Nmeasurements)

fitplot = fun([omegafits, taufits, rfits], incplot, 0.)


for i, val in enumerate(fitplot):
    ax.plot(incplot[i], val, alpha=0.4, label=i)

ax.plot(incplot[0], fun(x02[::Nmeasurements], incplot[0], 0.), 'k--', linewidth=2, label='fitstart')
plt.legend(loc=1)
plt.xlabel('$\\theta_0$ [deg]')
plt.ylabel('$I_{tot}$')


ax2 = fig.add_subplot(212)
ilabel = ['omega', 'tau', 'R']


# plot actual values
plt.gca().set_prop_cycle(None)
for i, val in enumerate(np.split(xdata, 3)):
    ax2.plot(val, '--', alpha=0.75)
plt.gca().set_prop_cycle(None)
for i, val in enumerate(np.split(xdata, 3)):
    ax2.plot(val, 'o')

# plot fitted values
plt.gca().set_prop_cycle(None)
for i, val in enumerate(np.split(res_lsq2.x, 3)):
    ax2.plot(val, alpha=0.75, label=ilabel[i])
plt.gca().set_prop_cycle(None)
for i, val in enumerate(np.split(res_lsq2.x, 3)):
    ax2.plot(val, 'k.', alpha=0.75)


# plot errors
for i, val in enumerate(np.split(res_lsq2.x - xdata, 3)):
    ax2.plot(val, ':', alpha=.5)
plt.gca().set_prop_cycle(None)
for i, val in enumerate(np.split(res_lsq2.x - xdata, 3)):
    ax2.plot(val, '.', alpha=.5)

import matplotlib.lines as mlines
h1 = mlines.Line2D([], [], color='black', label='data', linestyle='--', alpha=0.75, marker='o')
h2 = mlines.Line2D([], [], color='black', label='estimates', linestyle='-', alpha=0.75, marker='.')
h3 = mlines.Line2D([], [], color='black', label='errors', linestyle=':', alpha=0.5, marker='.')

handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles=handles + [h1, h2, h3], loc=1)

# set ticks
plt.xticks(range(Nmeasurements))
plt.xlabel('# Measurement')
plt.ylabel('Parameters / Errors')

plt.tight_layout()
'''
------------------------------------------------------
the rest is comments (i.e. unfinished stuff and tests)
------------------------------------------------------
'''


#
#
#datanewvals = fun([omegadata[:,np.newaxis],taudata[:,np.newaxis],rdata[:,np.newaxis]], inc, 0.)
#
#datanew = []
# for i,j in enumerate(datanewvals):
#    datanew = datanew + [[inc[i],j]]
#datanew = np.array(datanew)
#
#
# def fitfunct(R, data, startvals):
#
#    from scipy.optimize import least_squares
#
#    # rectangularize numpy array by adding nan-values
#    # (this is necessary because numpy can only deal with rectangular arrays)
#    maxLen = np.max(np.array([len(j[0]) for i,j in enumerate(inputdata)]))
#    for row in inputdata:
#        if len(row[0]) < maxLen:
#            row[0].extend([np.nan])
#            row[1].extend([np.nan])
#
#
#    # extract incidence-angle arrays from given dataset to a 1d-array
#    datainc = np.array([])
#    for i,j in enumerate(data):
#        datainc = np.append(datainc,j[0])
#
#    # extract data arrays from given dataset to a 1d-array
#    dataval = np.array([])
#    for i,j in enumerate(data):
#        dataval = np.append(dataval,j[1])
#
#
#    # define objective function for the fit
#    def fun(params, x, data):
#        V.omega = params[0:int(len(params)/3)]
#        V.tau = params[int(len(params)/3):int(2*len(params)/3)]
#        SRF.NormBRDF = params[int(2*len(params)/3):int(len(params))]
#
#        R = RT1(I0, x, x, np.ones_like(x) * 0., np.ones_like(x) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
#
#        errs = R.calc()[0]
#        errs = np.concatenate(errs) - dataval
#        return errs
#
#
#    # define function to evaluate the jacobian
#    def dfun(params, x, data):
#        V.omega = params[0:int(len(params)/3)]
#        V.tau = params[int(len(params)/3):int(2*len(params)/3)]
#        SRF.NormBRDF = params[int(2*len(params)/3):int(len(params))]
#
#        R = RT1(I0, x, x, np.ones_like(x) * 0., np.ones_like(x) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
#
#        jac = np.concatenate(R.jacobian2()).T
#        return jac
#
#
#    # concatenate start-values to a 1d-array
#    x0 = np.concatenate((startvals[0],startvals[1],startvals[2]))
#
#    inc = datainc
#    data = dataval
#
#    #res_lsq = least_squares(fun, x0, args=(inc, data), verbose=2, bounds=([0.]*len(x0), [1.]*len(x0)), jac = dfun, xtol=1.e-6, ftol = 1.e-6, gtol=1.e-6)
#
#    return data, inc, fun(x0, inc, data)
#
#
#
#
#'''
# datastructure:
#
# inc_i ... array of incidence-angles for the i'th measurement
# val_i ... array of corresponding measurements
#
# data = [
#        [[inc_1], [val_1]], # measurement 1
#        [[inc_2], [val_2]], # measurement 2
#        [[inc_3], [val_3]], # measurement 3
#        ...
#        ]
#
#'''


# initialize data-array
#
# inputdata = [[[.2,.3,.4],[.99,.87,.76]],
#             [[.2,.3],[.99,.87]],
#             [[.2,.3,.4],[.4654,.57,.5646]],
#             [[.2,.43],[.659,.56]],
#             [[.2,.2,.4],[.49,.47,.66]],
#             ]*200
#
#
#
# rectangularize numpy array by adding nan-values
#maxLen = np.max(np.array([len(j[0]) for i,j in enumerate(inputdata)]))
# for row in inputdata:
#    if len(row[0]) < maxLen:
#        row[0].extend([np.nan])
#        row[1].extend([np.nan])
#
#
# extract incidence-angle arrays from given dataset to a 1d-array
#datainc = np.array([])
# for i,j in enumerate(inputdata):
#    datainc = np.append(datainc,j[0])
#
# extract data arrays from given dataset to a 1d-array
#dataval = np.array([])
# for i,j in enumerate(inputdata):
#    dataval = np.append(dataval,j[1])
#
#
#R.fn = fn
#
#t_0 = np.array(np.split(datainc,len(inputdata)))
#
#t_0 = np.array([np.linspace(0.1,0.9,5)]*len(inputdata))
#
#V.tau = np.random.rand(len(inputdata))
#V.omega = np.random.rand(len(inputdata))
#SRF.NormBRDF = np.random.rand(len(inputdata))
#
#
#V.tau = V.tau[:,np.newaxis]
#V.omega = V.omega[:,np.newaxis]
#SRF.NormBRDF = SRF.NormBRDF[:,np.newaxis]
#
#V.tau[2][0] = 0.
#
#R = RT1(I0, t_0, t_0, np.ones_like(t_0) * 0., np.ones_like(t_0) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
#
#
#tottest = R.calc()[0]
#
#
#
#fitfunct(R, inputdata, np.array([[.3]*3,[.3]*3,[.3]*3]))
#
#


'''
-------------------------------------
------------------------------------
'''


#
#
#'''
# specification of objective function
#'''
#
# def fun(x, t, y):
#    V.omega = x[0]
#    V.tau = x[1]
#    SRF.NormBRDF = x[2]
#    R = RT1(I0, t, t, np.ones_like(t) * 0., np.ones_like(t) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
#
#    return R.calc()[0] - y
#
# start-values
#x0 = [.3,.4,.2]
#
##x = [omegas, taus, Rs]
#
#
#'''
# generate sample data
#'''
#inc = np.deg2rad(np.linspace(1., 89., 50))
#
# xdata = np.random.rand(3)#[.234,.347,.327]
#data = fun(xdata, inc, 0.)
#
#data = data
#dataeps = np.random.randn(len(data))*max(data)/50.
#data = data + dataeps
#
#from scipy.optimize import least_squares
#
#tic = timeit.default_timer()
#res_lsq = least_squares(fun, x0, args=(inc, data), verbose=2, ftol = 1.e-4, xtol = 1.e-2)
#toc = timeit.default_timer()
#print('it took ' + str(toc-tic) + 'sec without jac')
#
# def dfun(x, t, y):
#    V.omega = x[0]
#    V.tau = x[1]
#    SRF.NormBRDF = x[2]
#    R = RT1(I0, t, t, np.ones_like(t) * 0., np.ones_like(t) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
#
#    return R.jacobian().T
#
#tic = timeit.default_timer()
#res_lsq_withdfun = least_squares(fun, x0, args=(inc, data), verbose=2, jac = dfun, x_scale = 'jac', ftol = 1.e-4, xtol = 1.e-2)
#toc = timeit.default_timer()
#print('it took ' + str(toc-tic) + 'sec if jac is provided')
#
#
#
#import matplotlib.pyplot as plt
#
#plt.plot(inc, data,'ro')
#plt.plot(inc, fun(res_lsq.x,inc,0.), 'g')
#plt.plot(inc, fun(res_lsq_withdfun.x,inc,0.), 'r')
#plt.title(str(xdata) + '\n' +  str(res_lsq.x) + '\n' + str(res_lsq_withdfun.x))
# plt.tight_layout()
#
#
# def dfun2(params, x, data):
#    V.omega = params[0:int(len(params)/3)]
#    V.tau = params[int(len(params)/3):int(2*len(params)/3)]
#    SRF.NormBRDF = params[int(2*len(params)/3):int(len(params))]
#
#    #x = np.concatenate([x]*(int(len(params)/3)))
#
#    R = RT1(I0, x, x, np.ones_like(x) * 0., np.ones_like(x) * 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
#
#    jac = R.jacobian()
#    return jac
#
#
# def newjac(params, x, data):
#    testjac = np.zeros((int(len(params)),int(len(x)*len(params)/3)))
#
#    for i in range(dfun2(params,x,data).shape[1]):
#        testjac[i][int(i*len(x)):int((i+1)*len(x))] = dfun2(params,x,data)[0][i]
#        testjac[i+int(len(params)/3)][int(i*len(x)):int((i+1)*len(x))] = dfun2(params,x,data)[1][i]
#        testjac[i+int(2*len(params)/3)][int(i*len(x)):int((i+1)*len(x))] = dfun2(params,x,data)[2][i]
#    return testjac.T

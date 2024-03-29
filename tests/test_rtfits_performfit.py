"""
Test the fits-module by generating a dataset and fitting the model to itself


This is the very same test as test_rtfits but it uses the
rtfits.performfit-wrapper
"""

import unittest

# import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.stats import linregress

from rt1.rt1 import RT1
from rt1.rtfits import Fits

from rt1.volume import Rayleigh
from rt1.surface import HenyeyGreenstein as HGsurface
import pandas as pd


class TestRTfits(unittest.TestCase):
    def performfit(
        self,
        sig0,
        dB,
        Nmeasurements,
        mininc,
        maxinc,
        minincnum,
        maxincnum,
        omin,
        omax,
        taumin,
        taumax,
        rmin,
        rmax,
        tmin,
        tmax,
    ):
        """
        fucntion adapted from doc/examples/examples_fitting.py

        the values are generated using np.random.seed(0) for all
        calls of np.random...
        """

        # ---------------------------------------------------------------------
        # ------------------------- DATA-GENERATION ---------------------------

        # generate N arrays of incidence-angles that contain maxincnum
        # values between mininc and maxinc
        inc = np.array(
            [np.deg2rad(np.linspace(mininc, maxinc, maxincnum))] * Nmeasurements
        )

        np.random.seed(0)  # reset seed to have a reproducible test
        # generate random samples of parameters
        omegadata = np.random.uniform(low=omin, high=omax, size=(Nmeasurements,))

        np.random.seed(0)  # reset seed to have a reproducible test
        taudata = np.random.uniform(low=taumin, high=taumax, size=(Nmeasurements,))

        np.random.seed(0)  # reset seed to have a reproducible test
        rdata = np.random.uniform(low=rmin, high=rmax, size=(Nmeasurements,))

        np.random.seed(0)  # reset seed to have a reproducible test
        tdata = np.random.uniform(low=tmin, high=tmax, size=(Nmeasurements,))

        # set tau of Neq measurements to be equal and fit only a single param.
        # for tau to all datasets

        # choose a random number of equal datasets
        if Nmeasurements == 1:
            Neq = 0
            equal_tau_selects = [0]
        else:
            Neq = int(Nmeasurements / 5)

            np.random.seed(0)  # reset seed to have a reproducible test
            equal_tau_selects = np.random.choice(
                range(Nmeasurements), size=Neq + 1, replace=False
            )

        for i in equal_tau_selects:
            # for i in [1,3,5,7,9,11,14,16,17]:
            taudata[i] = taudata[equal_tau_selects[0]]

        # define model that is used to generate the data
        # the choices for tau, omega and NormBRDF have no effect on the
        # dataset since they will be changed to randomly generated values!

        V_data = Rayleigh(tau=0.1, omega=0.1)
        SRF_data = HGsurface(ncoefs=10, t=sp.var("t_data"), a=[1.0, 1.0, 1.0])

        # setup rt1-object
        # (since the fn-coefficients must still be calculated, one must
        #  specify the arrays for the parameters afterwards)
        R_data = RT1(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            V=V_data,
            SRF=SRF_data,
            geometry="mono",
            param_dict={"t_data": 0.5},
        )

        # specify parameters and incidence-angles
        R_data.t_0 = inc
        R_data.p_0 = np.zeros_like(inc)

        R_data.V.omega = omegadata[:, np.newaxis]
        R_data.V.tau = taudata[:, np.newaxis]
        R_data.SRF.NormBRDF = rdata[:, np.newaxis]
        R_data.param_dict = {"t_data": tdata[:, np.newaxis]}
        # calculate the data and add some random noise
        data = R_data.calc()[0]

        np.random.seed(0)  # reset seed to have a reproducible test
        noise = np.random.uniform(
            low=-np.max(data) / 50.0, high=np.max(data) / 50.0, size=data.shape
        )
        data = data + noise

        if sig0 is True:
            # convert the calculated results do sigma0
            signorm = 4.0 * np.pi * np.cos(inc)
            data = signorm * data

        if dB is True:
            # convert the calculated results to dB
            data = 10.0 * np.log10(data)

        # define the mask for selecting non-rectangular arrays of data (this is
        # done to show that fitting also works for non-rectangular datasets)

        np.random.seed(0)  # reset seed to have a reproducible test
        inc_lengths = np.random.randint(minincnum, maxincnum, Nmeasurements)
        selects = []
        for i in range(Nmeasurements):
            np.random.seed(0)  # reset seed to have a reproducible test
            selects += [
                np.random.choice(range(maxincnum), inc_lengths[i], replace=False)
            ]

        # generate dataset of the shape [ [inc_0, data_0], [inc_1, data_1], ..]
        # with inc_i and data_i being arrays of varying length

        dfindex = [nsel for nsel, i in enumerate(selects) for j in i]
        dfinc = [j for i, inc_i in enumerate(inc) for j in inc_i[selects[i]]]
        dfsig = [j for i, data_i in enumerate(data) for j in data_i[selects[i]]]

        dataset = pd.DataFrame(
            {"inc": dfinc, "sig": dfsig}, index=pd.to_datetime(dfindex, unit="D")
        )
        # ---------------------------------------------------------------------
        # ------------------------------- FITTING -----------------------------

        def set_V_SRF(t1, N, tau, omega, **kwargs):
            V = Rayleigh(omega=omega, tau=tau)
            SRF = HGsurface(ncoefs=10, t=t1, NormBRDF=N, a=[1.0, 1.0, 1.0])
            return V, SRF

        # specify additional arguments for scipy.least_squares
        lsq_kwargs = {
            "ftol": 1e-8,
            "gtol": 1e-8,
            "xtol": 1e-8,
            "max_nfev": 500,
            "method": "trf",
            "tr_solver": "lsmr",
            "x_scale": "jac",
        }

        # select random numbers within the boundaries as sart-values
        np.random.seed(0)  # reset seed to have a reproducible test
        ostart = (omax - omin) * np.random.random() + omin
        np.random.seed(0)  # reset seed to have a reproducible test
        tstart = (tmax - tmin) * np.random.random() + tmin
        taustart = (taumax - taumin) * np.random.random() + taumin

        # fit only a single parameter to the datasets that have equal tau
        _, fittau_dyn = np.unique(dataset.index, return_inverse=True)
        fittau_dyn[np.isin(fittau_dyn, equal_tau_selects)] = fittau_dyn[
            np.isin(fittau_dyn, equal_tau_selects)
        ][0]
        # add manual parameter dynamics for tau
        dataset["tau_dyn"] = fittau_dyn

        # specify the treatment of the parameters in the retrieval procedure
        defdict = {
            "t1": [True, tstart, "D", ([tmin], [tmax])],
            "N": [False, "auxiliary"],
            "tau": [True, taustart, "manual", ([taumin], [taumax])],
            "omega": [True, ostart, None, ([omin], [omax])],
            "bsf": [False, 0.0],
        }

        # append auxiliary datasets

        N_auxdata = pd.DataFrame({"N": rdata}, pd.unique(dataset.index)).loc[
            dataset.index
        ]
        dataset["N"] = N_auxdata

        # initialize fit-class
        testfit = Fits(
            sig0=sig0,
            dB=dB,
            dataset=dataset,
            defdict=defdict,
            set_V_SRF=set_V_SRF,
            lsq_kwargs=lsq_kwargs,
            int_Q=True,
            verbose=2,
        )

        # print model definition
        testfit.model_definition

        # perform the fit
        testfit.performfit(print_progress=True)

        # check if _calc_slope_curv is working
        # TODO this only tests if a result is obtained, not if the result
        # is actually correct !!
        slops, curvs = testfit._calc_slope_curv()

        # provide true-values for comparison of fitted results
        truevals = {
            "tau": taudata,
            "omega": omegadata,
            "t1": tdata,
        }

        # ----------- calculate R^2 values and errors of parameters -----------

        # sicne fit[0].fun gives the residuals weighted with respect to
        # weights, the model calculation can be gained via
        # estimates = fit[0].fun/weights + measurements

        # apply mask
        measures = testfit.data[~testfit.mask]
        estimates = testfit.fit_output.fun + measures

        # evaluate linear regression to get r-value etc.
        slope, intercept, r_value, p_value, std_err = linregress(estimates, measures)
        # calculate R^2 value
        r2 = r_value**2

        # check if r^2 between original and fitted data is > 0.95
        self.assertTrue(r2 > 0.95, msg=f"r^2 condition not  met , R={r2:4f}")

        # set mean-error values for the derived parameters
        if sig0 is True and dB is False:
            errdict = {"tau": 0.03, "omega": 0.008, "t1": 0.05}

        if sig0 is False and dB is False:
            errdict = {"tau": 0.03, "omega": 0.01, "t1": 0.08}

        if sig0 is True and dB is True:
            errdict = {"tau": 0.03, "omega": 0.01, "t1": 0.09}

        if sig0 is False and dB is True:
            errdict = {"tau": 0.03, "omega": 0.01, "t1": 0.09}

        fittedvals = testfit.res_df
        for key in truevals:
            err = abs(fittedvals[key] - truevals[key]).mean()
            self.assertTrue(
                err < errdict[key],
                msg="derived error" + str(err) + "too high for " + str(key),
            )

        return truevals, r2

    def test_sig0_linear(self):
        self.performfit(
            sig0=True,
            dB=False,
            Nmeasurements=10,
            mininc=25,
            maxinc=65,
            minincnum=20,
            maxincnum=50,
            omin=0.35,
            omax=0.4,
            taumin=0.1,
            taumax=1.25,
            rmin=0.1,
            rmax=0.5,
            tmin=0.0001,
            tmax=0.5,
        )

    def test_sig0_dB(self):
        self.performfit(
            sig0=True,
            dB=True,
            Nmeasurements=10,
            mininc=25,
            maxinc=65,
            minincnum=20,
            maxincnum=50,
            omin=0.35,
            omax=0.4,
            taumin=0.1,
            taumax=1.25,
            rmin=0.1,
            rmax=0.5,
            tmin=0.0001,
            tmax=0.5,
        )

    def test_I_linear(self):
        self.performfit(
            sig0=False,
            dB=False,
            Nmeasurements=10,
            mininc=25,
            maxinc=65,
            minincnum=20,
            maxincnum=50,
            omin=0.35,
            omax=0.4,
            taumin=0.1,
            taumax=1.25,
            rmin=0.1,
            rmax=0.5,
            tmin=0.0001,
            tmax=0.5,
        )

    def test_I_dB(self):
        self.performfit(
            sig0=False,
            dB=True,
            Nmeasurements=10,
            mininc=25,
            maxinc=65,
            minincnum=20,
            maxincnum=50,
            omin=0.35,
            omax=0.4,
            taumin=0.1,
            taumax=1.25,
            rmin=0.1,
            rmax=0.5,
            tmin=0.0001,
            tmax=0.5,
        )


if __name__ == "__main__":
    unittest.main()


# asdf = TestRTfits()
#
# truevals, fit, r2 = asdf.performfit(
#                       sig0=True,
#                       dB=False,
#                       Nmeasurements=10,
#                       mininc=25,
#                       maxinc=65,
#                       minincnum=20,
#                       maxincnum=50,
#                       omin=0.35,
#                       omax=0.4,
#                       taumin=0.1,
#                       taumax=1.25,
#                       rmin=0.1,
#                       rmax=0.5,
#                       tmin=0.0001,
#                       tmax=0.5,
#                       )
#
# truevals2, fit2, r22 = asdf.performfit(
#                       sig0=False,
#                       dB=False,
#                       Nmeasurements=10,
#                       mininc=25,
#                       maxinc=65,
#                       minincnum=20,
#                       maxincnum=50,
#                       omin=0.35,
#                       omax=0.4,
#                       taumin=0.1,
#                       taumax=1.25,
#                       rmin=0.1,
#                       rmax=0.5,
#                       tmin=0.0001,
#                       tmax=0.5,
#                       )
#
#
# truevals3, fit3, r23 = asdf.performfit(
#                       sig0=True,
#                       dB=True,
#                       Nmeasurements=10,
#                       mininc=25,
#                       maxinc=65,
#                       minincnum=20,
#                       maxincnum=50,
#                       omin=0.35,
#                       omax=0.4,
#                       taumin=0.1,
#                       taumax=1.25,
#                       rmin=0.1,
#                       rmax=0.5,
#                       tmin=0.0001,
#                       tmax=0.5,
#                       )
#
#
#
# truevals4, fit4, r24 = asdf.performfit(
#                       sig0=False,
#                       dB=True,
#                       Nmeasurements=10,
#                       mininc=25,
#                       maxinc=65,
#                       minincnum=20,
#                       maxincnum=50,
#                       omin=0.35,
#                       omax=0.4,
#                       taumin=0.1,
#                       taumax=1.25,
#                       rmin=0.1,
#                       rmax=0.5,
#                       tmin=0.0001,
#                       tmax=0.5,
#                       )
#
# print(
# abs(fit[6]['tau'] - truevals['tau']).mean(),
# abs(fit[6]['omega'] - truevals['omega']).mean(),
# abs(fit[6]['t1'] - truevals['t1']).mean())
# 0.0153978186281 0.00742310815396 0.0416097423517
#
# print(
# abs(fit2[6]['tau'] - truevals2['tau']).mean(),
# abs(fit2[6]['omega'] - truevals2['omega']).mean(),
# abs(fit2[6]['t1'] - truevals2['t1']).mean())
# 0.0194644435896 0.00910352864062 0.0753220593444
#
# print(
# abs(fit3[6]['tau'] - truevals3['tau']).mean(),
# abs(fit3[6]['omega'] - truevals3['omega']).mean(),
# abs(fit3[6]['t1'] - truevals3['t1']).mean())
# 0.0202776696237 0.00849788787843 0.080153724151
#
# print(
# abs(fit4[6]['tau'] - truevals4['tau']).mean(),
# abs(fit4[6]['omega'] - truevals4['omega']).mean(),
# abs(fit4[6]['t1'] - truevals4['t1']).mean())
# 0.0202776696237 0.00849788787843 0.080153724151

#
#
# sig0=False
# dB=True
# Nmeasurements=10
# mininc=25
# maxinc=65
# minincnum=20
# maxincnum=50
# omin=0.35
# omax=0.4
# taumin=0.1
# taumax=1.25
# rmin=0.1
# rmax=0.5
# tmin=0.0001
# tmax=0.5

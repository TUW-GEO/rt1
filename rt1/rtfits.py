"""
Class to perform least_squares fitting of given datasets.
(wrapper for scipy.optimize.least_squares)
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import least_squares

from .scatter import Scatter
from .rt1 import RT1


class Fits(Scatter):
    '''
    Class to perform nonlinear least-squares fits to data.

    Parameters:
    ------------
    sig0 : boolean (default = False)
           Indicator whether the data is given as sigma_0-values (sig_0) or as
           intensity-values (I). The applied relation is:
               sig_0 = 4. * np.pi * np.cos(inc) * I
           where inc is the corresponding incident zenith-angle.
    dB : boolean (default = False)
         Indicator whether the data is given in linear units or in dB.
         The applied relation is:    x_dB = 10. * np.log10( x_linear )
    '''

    def __init__(self, sig0=False, dB=False, **kwargs):
        self.sig0 = sig0
        self.dB = dB

    def preparedata(self, dataset):
        '''
        prepare data such that it is applicable to least_squres fitting
        - separate incidence-angles and data-values
        - rectangularize the data-array
        - provide weighting-matrix to correct for rectangularization

        Parameters:
        ------------
        dataset: array-like
                 input-dataset as list of the shape:
                     [[inc_0, data_0], [inc_1, data_2], ...]

        Returns:
        ---------
        inc : array-like
              a rectangular array consisting of the incidence-angles as
              provided in the dataset, rectangularized by repeating the last
              values of each row to fit in length.
        data : array-like
               a rectangular array consisting of the data-values as provided
               in the dataset, rectangularized by repeating the last values
               of each row to fit in length
        weights : array-like
                  an array of equal shape as inc and data, consisting of the
                  weighting-factors that need to be applied in order to correct
                  for the rectangularization. The weighting-factors for each
                  individual data-element are given by
                      weight_i = 1 / np.sqrt(N_i)
                  where N_i is the number of repetitions of the i'th value
                  that have been added in order to rectangularize the dataset.

                  Including the weighting-factors within the least-squares
                  approach will result in a cancellation of the repeated
                  results such that the artificially added values (necessary
                  to have a rectangular array) will have no effect on the fit.
        N : int
            number of measurements that have been provided within the dataset
        '''
        # save number of datasets
        N = len(dataset)

        # get incidence-angles and data-values into separate lists
        inc, data = [], []
        for i, val in enumerate(dataset):
            inc = inc + [val[0]]
            data = data + [val[1]]

        # rectangularize numpy array by adding nan-values
        # (necessary because numpy can only deal with rectangular arrays)
        maxLen = np.max(np.array([len(j) for i, j in enumerate(inc)]))
        for i, j in enumerate(inc):
            if len(j) < maxLen:
                inc[i] = np.append(inc[i],
                                   np.tile(np.nan, maxLen - len(inc[i])))
                data[i] = np.append(data[i],
                                    np.tile(np.nan, maxLen - len(data[i])))

        # generate a mask to be able to re-create the initial datset
        mask = np.isnan(data)

        # concatenate data-matrix to 1d-array
        # (necessary since least_squares can only deal with 1d arrays)
        data = np.concatenate(data)
        inc = np.concatenate(inc)

        # prepare data to avoid nan-values
        #      since numpy only supports rectangular arrays, and least_squares
        #      does neither support masked arrays, nor containing nan-values,
        #      the problem of missing values within the dataset is currently
        #      adressed by repeating the values from the nearest available
        #      neighbour to rectangularize the dataset.

        #      this of course results in an inhomogeneous treatment of the
        #      measurements since the duplicates have been added artificially!
        #      -> in order to correct for the added duplicates, a
        #      weighting-matrix is provided that can be used to correct for
        #      the unwanted duplicates.

        weights = np.ones_like(data)
        i = 0

        while i < len(data):
            if np.isnan(data[i]):
                j = 0
                while np.isnan(data[i + j]):
                    data[i + j] = data[i + j - 1]
                    inc[i + j] = inc[i + j - 1]
                    j = j + 1
                    if i + j >= len(data):
                        break
                # the weights are calculated as one over the square-root of
                # the number of repetitions in order to cancel out the
                # repeated measurements in the sum of SQUARED residuals.
                weights[i - 1: i + j] = 1. / np.sqrt(float(j + 1))
            i = i + 1

        inc = np.array(np.split(inc, N))

        return inc, data, weights, N, mask

    def monofit(self, V, SRF, dataset,
                startvals=None,
                bounds=None,
                fn=None,
                **kwargs):
        '''
        Perform a simultaneous fit of the model defined via V and SRF to
        to the monostatic measurements provided via the dataset.


        Parameters:
        ------------
        V : RT1.volume class object
            The volume scattering phase-function used to define the fit-model
        SRF: RT1.surface class object
             The surface BRDF used to define the fit-model
        dataset : list
                 list of input-data and incidence-angles arranged in the form
                     [[inc_0,data_0], [inc_1,data_1], ...]
                 where inc_i denotes the incident zenith-angles in radians
                 and data_i denotes the corresponding data-values of the i^th
                 measurement. Each measurement can contain different numbers
                 of datapoints!

        Other Parameters:
        ------------------
        startvals : array-like
                    start-values for the fit
                    Default, the start-values for all measurements
                    are set to 0.3
        bounds : array-like
                 boundary-values for the fit
                 Default, the boundaries for all measurements are set
                 to (0., 1.)
        fn : array-like
             pre-calculated fn-coefficients
        **kwargs : -
                 **kwargs passed to scipy's least_squares function

        Returns:
        ---------
        res_lsq2 : dict
                   output of scipy's least_squares function
        data : array-like
               used dataset for the fit
               (rectangularized to allow array processing)
        inc : array-like
              used incidence-angle data for the fit
              (rectangularized to allow array processing)
        V : volume-class element
            used volume-scattering phase function for the fit
        SRF : surface-class element
              used surface-BRDF for the fit
        fn : array-like
             used fn-coefficients for the fit
        startvals : array-like
                    used start-values for the fit
        mask : array-like (boolean)
               a mask that shows the added values to the data and inc arrays
               which can be used to generate masked-arrays containing only
               the original entries by using:
                   inc_orig  = numpy.ma.masked_array(inc, mask)
                   data_orig = numpy.ma.masked_array(data, mask)
        '''

        # prepare data for fit
        inc, data, weights, Nmeasurements, mask = self.preparedata(dataset)

        # pre-calculate fn-coefficients if they are not provided explicitly
        R = RT1(1., 0., 0., 0., 0., RV=V, SRF=SRF, fn=fn, geometry='mono')
        fn = R.fn  # store coefficients for faster iteration

        def fun(params, inc, data):
            '''
            function to evaluate the residuals in the shape as required
            by scipy's least_squares function
            '''
            V.omega = params[0:int(len(params) / 3)]
            V.tau = params[int(len(params) / 3):int(2 * len(params) / 3)]
            SRF.NormBRDF = params[int(2 * len(params) / 3):int(len(params))]
            R = RT1(1.,
                    inc, inc,
                    np.ones_like(inc) * 0., np.ones_like(inc) * np.pi,
                    RV=V, SRF=SRF, fn=fn, geometry='mono')

            errs = R.calc()[0]

            if self.sig0 is True:
                # convert the calculated results to sigma_0
                signorm = 4. * np.pi * np.cos(inc)
                errs = signorm * errs

            if self.dB is True:
                # convert the calculated results to dB
                errs = 10. * np.log10(errs)

            return np.concatenate(errs) - data

        def funnew(params, inc, data):
            '''
            function to incorporate the weighting-matrix
            '''
            return weights * fun(params, inc, data)

        # function to evaluate the jacobian
        def dfun(params, inc, data):
            '''
            function to evaluate the jacobian in the shape as required
            by scipy's least_squares function
            '''
            V.omega = params[0:int(len(params) / 3)]
            V.tau = params[int(len(params) / 3):int(2 * len(params) / 3)]
            SRF.NormBRDF = params[int(2 * len(params) / 3):int(len(params))]
            R = RT1(1.,
                    inc, inc,
                    np.ones_like(inc) * 0., np.ones_like(inc) * np.pi,
                    RV=V, SRF=SRF, fn=fn, geometry='mono')

            jac = R.jacobian(sig0=self.sig0, dB=self.dB)

            return np.concatenate(jac).T

        # define boundaries if none have been provided explicitly
        if bounds is None:
            bounds = ([0.] * 3 * Nmeasurements, [1.] * 3 * Nmeasurements)

        # define start-values if none have been provided explicitly
        if startvals is None:
            startvals = np.concatenate((
                np.linspace(1, 1, Nmeasurements) * 0.3,
                np.linspace(1, 1, Nmeasurements) * 0.3,
                np.linspace(1, 1, Nmeasurements) * 0.3))

        # fit with correct weighting of duplicates
        res_lsq2 = least_squares(funnew, startvals, args=(inc, data),
                                 bounds=bounds, jac=dfun, **kwargs)

        # get the data in the same shape as the incidence-angles
        data = np.array(np.split(data, Nmeasurements))

        return res_lsq2, data, inc, V, SRF, fn, startvals, mask

    def calc_res(self, res_lsq2, data, inc, V, SRF, fn, _dummy, mask):
        '''
        function to evaluate the residuals, i.e. :
            res = np.sqrt( (model - data)**2 )
        '''

        params = res_lsq2.x
        V.omega = params[0:int(len(params) / 3)]
        V.tau = params[int(len(params) / 3):int(2 * len(params) / 3)]
        SRF.NormBRDF = params[int(2 * len(params) / 3):int(len(params))]

        R = RT1(1.,
                inc, inc,
                np.ones_like(inc) * 0., np.ones_like(inc) * np.pi,
                RV=V, SRF=SRF, fn=fn, geometry='mono')

        estimates = R.calc()[0]

        if self.sig0 is True:
            # convert the calculated results to sigma_0
            estimates = 4. * np.pi * np.cos(inc) * estimates

        if self.dB is True:
            # convert the calculated results to dB
            estimates = 10. * np.log10(estimates)

        # !!IMPORTANT!!
        # calculate the residuals based on masked arrays
        masked_estimates = np.ma.masked_array(estimates, mask=mask)
        masked_data = np.ma.masked_array(data, mask=mask)

        res = np.ma.sqrt((masked_estimates - masked_data)**2)
        return res

    def printresults(self, fit, truevals=None):
        '''
        a function to quickly print fit-results

        Parametsrs:
        ------------
        fit : list
            output of monofit()-function

        truevals : array-like (default = None)
                   array of the expected parameter-values (must be of the
                   same shape as the parameter-values gained from the fit).
                   if provided, the difference between the expected- and
                   fitted values is plotted
        '''

        fit_res, data, inc, V, SRF, fn, startvals, mask = fit

        Nmeasurements = len(inc)

        # function to evaluate the model on the estimated parameters
        def fun(x, t):
            V.omega = x[0]
            V.tau = x[1]
            SRF.NormBRDF = x[2]
            R = RT1(1.,
                    t, t,
                    np.ones_like(t) * 0., np.ones_like(t) * 0.,
                    RV=V, SRF=SRF, fn=fn, geometry='mono')

            errs = R.calc()[0]

            if self.sig0 is True:
                signorm = 4. * np.pi * np.cos(t)
                errs = signorm * errs

            if self.dB is True:
                errs = 10. * np.log10(errs)

            return errs

        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(211)
        ax.set_title('Fit-results')

        for i, j in enumerate(data):
            ax.plot(inc[i], j, '.')

        plt.gca().set_prop_cycle(None)

        ofits = fit_res.x[0:int(len(fit_res.x) / 3)]
        tfits = fit_res.x[int(len(fit_res.x) / 3):int(2 * len(fit_res.x) / 3)]
        rfits = fit_res.x[int(2 * len(fit_res.x) / 3):int(len(fit_res.x))]

        incplot = np.array([np.linspace(np.min(inc), np.max(inc), 100)]
                           * Nmeasurements)

        fitplot = fun([ofits, tfits, rfits], incplot)

        for i, val in enumerate(fitplot):
            ax.plot(incplot[i], val, alpha=0.4, label=i + 1)

        # ----------- plot error-bars ------------
        #fitdata = fun([ofits, tfits, rfits], inc)
        # plt.gca().set_prop_cycle(None)
        # for i, val in enumerate(fitdata):
        #    errors = data[i] - val
        #    ax.errorbar(inc[i], val, errors, linestyle='None', fmt = '-')

        ax.plot(incplot[0],
                fun(startvals[::Nmeasurements], incplot[0]),
                'k--', linewidth=2, label='fitstart')

        plt.legend(loc=1)

        mintic = np.round(np.rad2deg(np.min(inc)) + 4.9, -1)
        if mintic < 0.:
            mintic = 0.
        maxtic = np.round(np.rad2deg(np.max(inc)) + 4.9, -1)
        if maxtic > 360.:
            maxtic = 360.

        ticks = np.arange(np.rad2deg(np.min(inc)),
                          np.rad2deg(np.max(inc)) + 1.,
                          (maxtic - mintic) / 10.)
        plt.xticks(np.deg2rad(ticks), np.array(ticks, dtype=int))
        plt.xlabel('$\\theta_0$ [deg]')
        plt.ylabel('$I_{tot}$')

        ax2 = fig.add_subplot(212)
        ax2.set_title('Estimated parameters')

        ax2.set_ylim(0., 1.)

        if truevals is not None:

            # plot actual values
            plt.gca().set_prop_cycle(None)
            for i, val in enumerate(np.split(truevals, 3)):
                ax2.plot(np.arange(1, Nmeasurements + 1), val,
                         '--', alpha=0.75)
            plt.gca().set_prop_cycle(None)
            for i, val in enumerate(np.split(truevals, 3)):
                ax2.plot(np.arange(1, Nmeasurements + 1), val, 'o')

            # plot errors
            param_errs = np.split(fit_res.x - truevals, 3)

            plt.gca().set_prop_cycle(None)
            for i, val in enumerate(param_errs):
                ax2.plot(np.arange(1, Nmeasurements + 1), val, ':', alpha=.5)
            plt.gca().set_prop_cycle(None)
            for i, val in enumerate(param_errs):
                ax2.plot(np.arange(1, Nmeasurements + 1), val, '.', alpha=.5)

            # set boundaries to allow displaying errors
            param_errs_min = np.min(param_errs)
            if param_errs_min > 0.:
                param_errs_min = 0

            ax2.set_ylim(np.min(param_errs), 1.)

            h2 = mlines.Line2D([], [], color='black', label='data',
                               linestyle='--', alpha=0.75, marker='o')
            h3 = mlines.Line2D([], [], color='black', label='errors',
                               linestyle=':', alpha=0.5, marker='.')

        ilabel = ['omega', 'tau', 'R']

        # plot fitted values
        plt.gca().set_prop_cycle(None)
        for i, val in enumerate(np.split(fit_res.x, 3)):
            ax2.plot(np.arange(1, Nmeasurements + 1), val,
                     alpha=0.75, label=ilabel[i])
        plt.gca().set_prop_cycle(None)
        for i, val in enumerate(np.split(fit_res.x, 3)):
            ax2.plot(np.arange(1, Nmeasurements + 1), val, 'k.', alpha=0.75)

        h1 = mlines.Line2D([], [], color='black', label='estimates',
                           linestyle='-', alpha=0.75, marker='.')

        handles, labels = ax2.get_legend_handles_labels()
        if truevals is None:
            plt.legend(handles=handles + [h1], loc=1)
        else:
            plt.legend(handles=handles + [h1, h2, h3], loc=1)

        # set ticks
        ax2.set_xticks(np.arange(1, Nmeasurements + 1))
        plt.xlabel('# Measurement')
        if truevals is None:
            plt.ylabel('Parameters')
        else:
            plt.ylabel('Parameters / Errors')

        fig.tight_layout()

        return fig

    def printerr(self, fit):
        '''
        a function to quickly print residuals for each measurement
        and for each incidence-angle value

        Parametsrs:
        ------------
        fit : list
            output of monofit()-function
        '''

        fit_res, data, inc, V, SRF, fn, _dummy, mask = fit

        Nmeasurements = len(inc)
        res = self.calc_res(fit_res, data, inc, V, SRF, fn, _dummy, mask)

        # apply mask to data and incidence-angles
        inc = np.ma.masked_array(inc, mask=mask)
        data = np.ma.masked_array(data, mask=mask)

        # make new figure
        figres = plt.figure(figsize=(14, 10))
        axres = figres.add_subplot(212)
        axres.set_title('Mean residual per measurement')

        axres2 = figres.add_subplot(211)
        axres2.set_title('Residuals per incidence-angle')

        # plot residuals for each emasurement
        for i, resval in enumerate(res):
            for j in resval:
                # the use of np.array() is incorporated for
                # python 2 compatibility since otherwise an error occurs
                # when masked values are plotted
                axres.plot(np.array(i + 1), np.array(j), '.', alpha=0.5)

        # plot mean residual for each measurement
        axres.plot(np.arange(1, Nmeasurements + 1), np.ma.mean(res, axis=1),
                   'k', linewidth=3, marker='o', fillstyle='none')

        # add some legends
        res_h = mlines.Line2D(
            [], [], color='black', label='Mean res.  per measurement',
            linestyle='-', linewidth=3, marker='o', fillstyle='none')
        res_h_dots = mlines.Line2D(
            [], [], color='black', label='Residuals',
            linestyle='-', linewidth=0, marker='.', alpha=0.5)

        handles, labels = axres.get_legend_handles_labels()
        axres.legend(handles=handles + [res_h_dots] + [res_h], loc=1)
        axres.set_xticks(np.arange(1, Nmeasurements + 1))

        axres.set_xlabel('# Measurement')
        axres.set_ylabel('Residual')

#        # evaluate mean residuals per incidence-angle
#        incsorted = np.full_like(inc, 0.)
#        ressorted = np.full_like(res, 0.)
#        for i, j in enumerate(res):
#            sortpattern = np.argsort(inc[i])
#            incsorted[i] = inc[i][sortpattern]
#            ressorted[i] = res[i][sortpattern]
#
#        meanincs = np.unique(np.concatenate(incsorted))
#        mean = np.full_like(meanincs, 0.)
#
#        for a, meanval in enumerate(mean):
#            count = 0.
#            for i, incrow in enumerate(incsorted):
#                for j, incval in enumerate(incrow):
#                    if incval == meanincs[a]:
#                        count = count + 1.
#                        mean[a] = mean[a] + ressorted[i][j]
#            mean[a] = mean[a] / count

        meanincs = np.unique(np.concatenate(inc))
        mean = np.full_like(meanincs, 0.)

        for a, incval in enumerate(meanincs):
            select = np.where(inc == incval)
            res_selected = res[select[0][:, np.newaxis],
                               select[1][:, np.newaxis]]
            mean[a] = np.mean(res_selected)

        sortpattern = np.argsort(meanincs)
        meanincs = meanincs[sortpattern]
        mean = mean[sortpattern]

        # plot residuals per incidence-angle for each measurement
        for i, resval in enumerate(res):
            sortpattern = np.argsort(inc[i])
            axres2.plot(inc[i][sortpattern], res[i][sortpattern],
                        ':', alpha=0.5, marker='.')

        # plot mean residual per incidence-angle
        axres2.plot(meanincs, mean,
                    'k', linewidth=3, marker='o', fillstyle='none')

        # add some legends
        res_h2 = mlines.Line2D(
            [], [], color='black', label='Mean res.  per inc-angle',
            linestyle='-', linewidth=3, marker='o', fillstyle='none')
        res_h_lines = mlines.Line2D(
            [], [], color='black', label='Residuals',
            linestyle=':', alpha=0.5)

        handles2, labels2 = axres2.get_legend_handles_labels()
        axres2.legend(handles=handles2 + [res_h_lines] + [res_h2], loc=1)

        axres2.set_xlabel('$\\theta_0$ [deg]')
        axres2.set_ylabel('Residual')

        # set incidence-angle ticks
        mintic = np.round(np.rad2deg(np.min(inc)) + 4.9, -1)
        if mintic < 0.:
            mintic = 0.
        maxtic = np.round(np.rad2deg(np.max(inc)) + 4.9, -1)
        if maxtic > 360.:
            maxtic = 360.

        ticks = np.arange(np.rad2deg(np.min(inc)),
                          np.rad2deg(np.max(inc)) + 1.,
                          (maxtic - mintic) / 10.)
        plt.xticks(np.deg2rad(ticks), np.array(ticks, dtype=int))
        figres.tight_layout()

        return figres

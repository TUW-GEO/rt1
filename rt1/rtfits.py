"""
Class to perform least_squares fitting of given datasets.
(wrapper for scipy.optimize.least_squares)
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import least_squares

from scipy.stats import linregress

from .scatter import Scatter
from .rt1 import RT1

import copy  # used to copy objects


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

    def _preparedata(self, dataset):
        '''
        prepare data such that it is applicable to least_squres fitting
        - separate incidence-angles and data-values
        - rectangularize the data-array
          (this is necessary in order to allow array-processing)
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



    def _calc_model(self, R, res_dict):
        '''
        function to calculate the model-results (intensity or sigma_0) based
        on the provided parameters in linear-units or dB

        Parameters:
        ------------
        R : RT1-object
            the rt1-object for which the results shall be calculated
        res_dict : dict
                   a dictionary containing all parameter-values that should
                   be updated before calling R.calc()
        Returns:
        ----------
        model_calc : the output of R.calc() (as intensity or sigma_0)
                     in linear-units or dB corresponding to the specifications
                     defined in the rtfits-class.
        '''

        # set omega, tau and NormBRDF-values to input
        if 'omega' in res_dict:
            R.RV.omega = res_dict['omega']
        if 'tau' in res_dict:
            R.RV.tau = res_dict['tau']
        if 'NormBRDF' in res_dict:
            R.SRF.NormBRDF = res_dict['NormBRDF']

        # make a dict that only contains the variables needed within the
        # fn-coefficient generation
        param_fn = res_dict.copy()
        param_fn.pop('omega', None)
        param_fn.pop('tau', None)
        param_fn.pop('NormBRDF', None)

        # ensure that the keys of the dict are strings
        strparam_fn = dict([[str(key),
                             np.expand_dims(param_fn[key], 1)]
                            for i, key in enumerate(param_fn.keys())])

        R.param_dict = strparam_fn

        # calculate total backscatter-values
        model_calc = R.calc()[0]

        if self.sig0 is True:
            # convert the calculated results to sigma_0
            signorm = 4. * np.pi * np.cos(R.t_0)
            model_calc = signorm * model_calc

        if self.dB is True:
            # convert the calculated results to dB
            model_calc = 10. * np.log10(model_calc)

        return model_calc



    def monofit(self, V, SRF, dataset,
                    param_dict, bounds_dict={}, fixed_dict = {},
                    fn=None, **kwargs):
        '''
        Perform least-squares fitting of omega, tau, NormBRDF and any
        parameter used to define V and SRF to sets of monostatic measurements.

        Parameters:
        ------------
        V : RT1.volume class object
            The volume scattering phase-function used to define the fit-model.
            Attention: if omega and/or tau are set to None, the values
            provided by V.omega and/or V.tau are used as constants!
        SRF : RT1.surface class object
             The surface BRDF used to define the fit-model
             Attention if NormBRDF is set to None, the values defined by
             SRF.NormBRDF will be used as constants!
        dataset : list
                 list of input-data and incidence-angles arranged in the form
                 [[inc_0, data_0], [inc_1, data_1], ...]
                 where inc_i denotes the incident zenith-angles in radians
                 and the data_i denotes the corresponding data-values
                 (If the dataset provided is not rectangular, it will be
                 rectangularized using the _preparedata()-function.)
        param_dict : dict
                    A dictionary containing the names of the parameters that
                    are intended to be fitted together with the desired
                    startvalues for the fit.

                    possible keys are:
                        - 'tau' ... to fit the optical depth
                        - 'omega' ... to fit the single scattering albedo
                        - 'NormBRDF' ... to fit the normalization-factor of
                          the brdf
                        - any string corresponding to a sympy.Symbol object
                          that has been used in the definition of V and SRF.

                    possible values are:
                        - None ... the parameters will be excluded in the fit,
                          and the values provided in V and SRF will
                          be used as constants.
                          (alternatively one can just simply not provide the
                          corresponding key in param_dict)
                        - array_like(float) ... the provided array must
                          have the same length as the dataset used for the fit!
                          If the value is an array, a unique value for the
                          parameter will be fitted to each measurement, using
                          the values of the provided array as startvalues.
                        - scalar(float) ... a single value will be fitted
                          to all measurements, using the provided value as
                          startvalue.

        bounds_dict : dict
                      A dictionary containing the names of the parameters that
                      are intended to be fitted together with the desired
                      boundaries for the fit.

                      optional keys are:
                          - 'tau', 'omega' and 'NormBRDF'. If the keys are
                            not provided in bounds_dict (but appear
                            in param_dict), the boundary-values will be set
                            to: ([0.], [1.]) for all three parameters.

                      requred keys are:
                          - any string corresponding to sympy.Symbol objects
                            that have been used in the definition of V and SRF.

                      possible values are:
                          - if the corresponding value of param_dict is scalar,
                            the boundary-conditions must be specified as
                            bounds_dict[key] = ([lower_bound], [upper_bound])
                            where lower_bound and upper_bound are scalars.
                          - if the corresponding value of param_dict is an
                            array, the boundary-conditions must be specified as
                            bounds_dict[key] = ([lower_bound], [upper_bound])
                            where lower_bound and upper_bound are arrays of
                            the same length as the dataset.
        fixed_dict : dict
                     A dictionary containing the names of the parameters that
                     have been used to define the phase-functions, but whose
                     values are intended to be used as constants throughout
                     the fit.

                     The primary use of this dict is for passing individual
                     values of a parameter for each measurement.
                     (if the parameter is constant for all measurements it
                     can equally well be passed directly in the definition
                     of the phase-functions.)


        Other Parameters:
        ------------------
        fn : array-like
             a slot for pre-calculated fn-coefficients.
             if the same model has to be fitted to multiple datasets, the
             fn-coefficients that are returned in the first fit can be used
             as input for the second fit to avoid repeated calculations.
        kwargs :
                 keyword arguments passed to scipy's least_squares function

        Returns:
        ---------
        res_lsq2 : dict
                   output of scipy's least_squares function
        data : array-like
               used dataset for the fit
        inc : array-like
              used incidence-angle data for the fit
        V : volume-class element
            used volume-scattering phase function for the fit
        SRF : surface-class element
              used surface-BRDF for the fit
        fn : array-like
             used fn-coefficients for the fit
        startvals : array-like
                    used start-values for the fit
        res_dict : dict
                   a dictionary containing the names of the fitted parameters
                   and the corresponding fit-results
        '''

        # generate a list of the names of the parameters that will be fitted.
        # (this is necessary to ensure correct broadcasting of values since
        # dictionarys do)
        order = [*{k: v for k, v in param_dict.items() if v is not None}]

        # preparation of data for fitting
        inc, data, weights, Nmeasurements, mask = self._preparedata(dataset)

        # get values for omega, tau and NormBRDF from parameter-dictionary
        omega = param_dict.get('omega', None)
        tau = param_dict.get('tau', None)
        NormBRDF = param_dict.get('NormBRDF', None)



        # check of general input-requirements
        #   check if all parameters have been provided
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, V._func.free_symbols)) - angset
        srfsymb = set(map(str, SRF._func.free_symbols)) - angset

        paramset = ((set(map(str, param_dict.keys()))
                    ^ set(map(str, fixed_dict.keys())) )
                    - {'tau', 'omega', 'NormBRDF'})

        assert paramset >= (vsymb | srfsymb), ('the parameters ' +
                       str((vsymb | srfsymb) - paramset) +
                       ' must be provided in param_dict')


        if omega is not None and not np.isscalar(omega):
            assert len(omega) == Nmeasurements, ('length of omega-array must' +
                      'be equal to the length of the dataset')
        if omega is None:
            assert len(V.omega) == Nmeasurements, ('length of' +
                      ' omega-array provided in the definition of V must' +
                      ' be equal to the length of the dataset')

        if tau is not None and not np.isscalar(tau):
            assert len(tau) == Nmeasurements, ('length of tau-array' +
                      ' must be equal to the length of the dataset')
        if tau is None:
            assert len(V.tau) == Nmeasurements, ('length of tau-array' +
                      ' provided in the definition of V must be equal to' +
                      ' the length of the dataset')

        if NormBRDF is not None and not np.isscalar(NormBRDF):
            assert len(NormBRDF) == Nmeasurements, ('length of' +
                      ' NormBRDF-array must be equal to the' +
                      ' length of the dataset')
        if NormBRDF is None:
            assert len(SRF.NormBRDF) == Nmeasurements, ('length of' +
                      ' NormBRDF-array provided in the definition of SRF' +
                      ' must be equal to the length of the dataset')


        # generate a dict containing only the parameters needed to evaluate
        # the fn-coefficients
        param_R = dict(**param_dict, **fixed_dict)
        param_R.pop('omega', None)
        param_R.pop('tau', None)
        param_R.pop('NormBRDF', None)

        if fn is None:
            # define rt1-object
            R = RT1(1., 0., 0., 0., 0.,
                RV=V, SRF=SRF, fn=None, geometry='mono',
                param_dict = param_R)

            # set geometry
            R.t_0 = inc
            R.p_0 = np.zeros_like(inc)
            R.t_ex = inc
            R.p_ex = np.full_like(inc, np.pi)
        else:
            # define rt1-object
            R = RT1(1., inc, inc, np.zeros_like(inc), np.full_like(inc, np.pi),
                    RV=V, SRF=SRF, fn=fn, geometry='mono',
                    param_dict = param_R)



        # define a function that evaluates the model in the shape as needed
        # for scipy's least_squares function
        def fun(params):
            # generate a dictionary to assign values based on input
            count_0, count_1 = 0, 0
            newdict = {}
            for key in order:
                count_1 = count_1 + len(np.atleast_1d(param_dict[key]))
                if np.isscalar(param_dict[key]):
                    newdict[key] = np.array(list(params[count_0: count_1])
                                            * Nmeasurements)
                else:
                    newdict[key] = params[count_0: count_1]
                count_0 = count_0 + len(np.atleast_1d(param_dict[key]))

            # incorporate values provided in fixed_dict
            # (i.e. incorporate fixed but dynamic parameter-values)
            newdict = dict(newdict, **fixed_dict)

            errs = np.concatenate(self._calc_model(R, newdict)) - data
            # incorporate weighting-matrix to ensure correct treatment
            # of artificially added values (see _preparedata()-fucntion)
            errs = weights * errs
            return errs



        # function to evaluate the jacobian
        def dfun(params):
            '''
            function to evaluate the jacobian in the shape as required
            by scipy's least_squares function

            Parameters:
            ------------
            params : array_like(float)
                     parameters for the fit, aranged as [omega, tau, NormBRDF]
            Returns:
            --------
            jac : array_like(float)
                  the jacobian corresponding to the fit-parameters in the
                  shape applicable to scipy's least_squres-function
            '''

            # generate a dictionary to assign values based on input
            count_0, count_1 = 0, 0
            newdict = {}
            for key in order:
                count_1 = count_1 + len(np.atleast_1d(param_dict[key]))
                if np.isscalar(param_dict[key]):
                    newdict[key] = np.array(list(params[count_0: count_1])
                                            * Nmeasurements)
                else:
                    newdict[key] = params[count_0: count_1]
                count_0 = count_0 + len(np.atleast_1d(param_dict[key]))

            # incorporate values provided in fixed_dict
            # (i.e. incorporate fixed but dynamic parameter-values)
            newdict = dict(newdict, **fixed_dict)

            # set omega, tau and NormBRDF-values to input
            if 'omega' in newdict:
                V.omega = newdict['omega']
            if 'tau' in newdict:
                V.tau = newdict['tau']
            if 'NormBRDF' in newdict:
                SRF.NormBRDF = newdict['NormBRDF']

            # make a dict that only contains the variables needed within the
            # fn-coefficient generation
            param_fn = newdict.copy()
            param_fn.pop('omega', None)
            param_fn.pop('tau', None)
            param_fn.pop('NormBRDF', None)

            strparam_fn = dict([[str(key),
                                 np.array(param_fn[key])[:, np.newaxis]]
                                for i, key in enumerate(param_fn.keys())])

            R.param_dict = strparam_fn

            # calculate the jacobian
            jac = R.jacobian(sig0=self.sig0, dB=self.dB,
                                 param_list = order)

            # remove unwanted columns from the jacobian
            splitjac = np.split(np.concatenate(jac), len(order))
            newjacdict = {}
            for i, key in enumerate(order):
                if np.isscalar(param_dict[key]):
                    newjacdict[key] = np.array([np.sum(splitjac[i], axis=0)])
                else:
                    newjacdict[key] = splitjac[i]

            jac_lsq = np.concatenate([newjacdict[key] for key in newjacdict])

            # return the transposed jacobian as needed by scipy's least_squares
            # return np.concatenate(jac).T
            return jac_lsq.T


        # define boundaries for omega, tau and NormBRDF if none have been
        # provided explicitly
        omega_bounds = bounds_dict.get('omega', None)
        tau_bounds = bounds_dict.get('tau', None)
        NormBRDF_bounds = bounds_dict.get('NormBRDF', None)

        if omega is not None:
            if omega_bounds is None:
                if np.isscalar(omega):
                    omega_bounds = ([0.], [1.])
                else:
                    omega_bounds = ([0.] * Nmeasurements,
                                    [1.] * Nmeasurements)
        else:
            omega_bounds = ([], [])

        if tau is not None:
            if tau_bounds is None:
                if np.isscalar(tau):
                    tau_bounds = ([0.], [1.])
                else:
                    tau_bounds = ([0.] * Nmeasurements,
                                  [1.] * Nmeasurements)
        else:
            tau_bounds = ([], [])

        if NormBRDF is not None:
            if NormBRDF_bounds is None:
                if np.isscalar(NormBRDF):
                    NormBRDF_bounds = ([0.], [1.])
                else:
                    NormBRDF_bounds = ([0.] * Nmeasurements,
                                       [1.] * Nmeasurements)
        else:
            NormBRDF_bounds = ([], [])

        bounds_dict['omega'] = omega_bounds
        bounds_dict['tau'] = tau_bounds
        bounds_dict['NormBRDF'] = NormBRDF_bounds

        bounds = [[], []]
        for key in order:
            bounds[0] = bounds[0] + list(bounds_dict[key][0])
            bounds[1] = bounds[1] + list(bounds_dict[key][1])

        # setup the start-value array as needed for the fit
        startvals = []
        for key in order:
            if param_dict[key] is not None:
                if np.isscalar(param_dict[key]):
                    startvals = startvals + [param_dict[key]]
                else:
                    startvals = startvals + list(param_dict[key])


        # perform actual fitting
        res_lsq = least_squares(fun, startvals, bounds=bounds, jac=dfun,
                                **kwargs)

        # generate a dictionary to assign values based on fit-results
        c_res_0, c_res_1 = 0, 0
        res_dict = {}
        start_dict = {}
        for key in order:
            c_res_1 = c_res_1 + len(np.atleast_1d(param_dict[key]))
            if np.isscalar(param_dict[key]):
                res_dict[key] = np.array(list(res_lsq.x[c_res_0: c_res_1]
                                              ) * Nmeasurements)
                start_dict[key] = np.array(list(startvals[c_res_0: c_res_1]
                                              ) * Nmeasurements)
            else:
                res_dict[key] = res_lsq.x[c_res_0: c_res_1]
                start_dict[key] = startvals[c_res_0: c_res_1]

            c_res_0 = c_res_0 + len(np.atleast_1d(param_dict[key]))


        # ------------------------------------------------------------------
        # ------------ prepare output-data for later convenience -----------

        # get the data in the same shape as the incidence-angles
        data = np.array(np.split(data, Nmeasurements))

        return res_lsq, R, data, inc, mask, res_dict, start_dict, fixed_dict




    def printresults(self, fit, truevals=None, startvals = False,
                     datelist = None, legends = True):
        '''
        a function to quickly print the fit-results and the gained parameters

        Parametsrs:
        ------------
        fit : list
              output of monofit_all()-function

        truevals : dict (default = None)
                   dictionary of the expected parameter-values (must be of the
                   same shape as the parameter-values gained from the fit).
                   if provided, the difference between the expected- and
                   fitted values is plotted
        startvals : bool (default = False)
                    if True, the model-results using the start-values are
                    plotted as black lines
        datelist : list
                   a list used to label the x-axis
                   the shape must be [ groups, indicators ]
                   where groups is a list of G values that are used to
                   group the measurements, and
                   indicators is a list containing N elements grouped into
                   G groups, where N is the number of measurements.

                   e.g. if you had a measurement in January and February
                   for the years 2014 and 2015, you have:
                       datelist = [[2014, 2015], [[1,2],[1,2]] ]
        legends : bool (default = True)
                  indicator if legends should be plotted

        Returns:
        ---------
        fig : matplotlib.figure object
        '''

        res_lsq, R, data, inc, mask, res_dict, start_dict, fixed_dict = fit

        # reset incidence-angles in case they have been altered beforehand
        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)


        Nmeasurements = len(inc)

        if truevals is not None:
            truevals = copy.copy(truevals)

            # generate a dictionary to assign values based on input
            for key in truevals:
                if np.isscalar(truevals[key]):
                    truevals[key] = np.array([truevals[key]]* Nmeasurements)
                else:
                    truevals[key] = truevals[key]


        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(211)
        ax.set_title('Fit-results')

        for i, j in enumerate(data):
            ax.plot(inc[i], j, '.')

        plt.gca().set_prop_cycle(None)

        incplot = np.array([np.linspace(np.min(inc), np.max(inc), 100)]
                           * Nmeasurements)

        R.t_0 = incplot
        R.p_0 = np.zeros_like(incplot)

        calc_dict = dict(**res_dict, **fixed_dict)

        fitplot = self._calc_model(R, calc_dict)

        for i, val in enumerate(fitplot):
            ax.plot(incplot[i], val, alpha=0.4, label=i + 1)

        # ----------- plot start-values ------------
        if startvals is True:
            startplot = self._calc_model(R, start_dict)
            for i, val in enumerate(startplot):
                ax.plot(incplot[i], val, 'k--', linewidth=1,
                        alpha = 0.5, label='fitstart')

        if legends is True:
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

        ax2.set_ylim(0., np.max([*res_dict.values()]))

        if truevals is not None:

            # plot actual values
            plt.gca().set_prop_cycle(None)
            for key in truevals:
                ax2.plot(np.arange(1, Nmeasurements + 1), truevals[key],
                         '--', alpha=0.75)
            plt.gca().set_prop_cycle(None)
            for key in truevals:
                ax2.plot(np.arange(1, Nmeasurements + 1), truevals[key], 'o')

            param_errs = {}
            param_errs_min = 0.
            for key in truevals:
                param_errs[key] = res_dict[key] - truevals[key]
                # find minumum value of errors
                param_errs_min = np.min([param_errs_min,
                                         np.min(param_errs[key])])

            plt.gca().set_prop_cycle(None)
            for key in truevals:
                ax2.plot(np.arange(1, Nmeasurements + 1), param_errs[key],
                         ':', alpha=.25)
            plt.gca().set_prop_cycle(None)
            for key in truevals:
                ax2.plot(np.arange(1, Nmeasurements + 1), param_errs[key],
                         '.', alpha=.25)

            ax2.set_ylim(param_errs_min, 1.)

            h2 = mlines.Line2D([], [], color='black', label='data',
                               linestyle='--', alpha=0.75, marker='o')
            h3 = mlines.Line2D([], [], color='black', label='errors',
                               linestyle=':', alpha=0.5, marker='.')

        # plot fitted values
        plt.gca().set_prop_cycle(None)
        for key in res_dict:
            ax2.plot(np.arange(1, Nmeasurements + 1), res_dict[key],
                     alpha=0.5, label=key)
        plt.gca().set_prop_cycle(None)
        for key in res_dict:
            ax2.plot(np.arange(1, Nmeasurements + 1), res_dict[key],
                     'k.', alpha=0.5)

        h1 = mlines.Line2D([], [], color='black', label='estimates',
                           linestyle='-', alpha=0.75, marker='.')

        handles, labels = ax2.get_legend_handles_labels()
        if truevals is None:
            plt.legend(handles=handles + [h1], loc=1)
        else:
            plt.legend(handles=handles + [h1, h2, h3], loc=1)

        # set ticks
        if datelist == None:
            ax2.set_xticks(np.arange(1, Nmeasurements + 1))
            plt.xlabel('# Measurement')
        else:
            ax2.set_xticks(np.arange(1, Nmeasurements + 1))
            ax2.set_xticklabels(np.concatenate(datelist[1]))

            locs = [len(i)/2 for i in datelist[1]]
            locmax = [len(i) for i in datelist[1]]
            locs = [locs[i] + np.sum(locmax[:i]) for i in range(len(locs))]

            for i, y in enumerate(datelist[0]):
                plt.annotate(s = str(y), xy = (locs[i], 0), xytext = (0, -20),
                             xycoords='data', textcoords='offset points',
                             va='top')

            plt.xlabel('# dates', labelpad = 20)

        if truevals is None:
            plt.ylabel('Parameters')
        else:
            plt.ylabel('Parameters / Errors')

        fig.tight_layout()

        return fig




    def printerr(self, fit, datelist = None):
        '''
        a function to quickly print residuals for each measurement
        and for each incidence-angle value

        Parametsrs:
        ------------
        fit : list
            output of monofit()-function
        datelist : list
           a list used to label the x-axis
           the shape must be [ groups, indicators ]
           where groups is a list of G values that are used to
           group the measurements, and
           indicators is a list containing N elements grouped into
           G groups, where N is the number of measurements.

           e.g. if you had a measurement in January and February
           for the years 2014 and 2015, you have:
               datelist = [[2014, 2015], [ [1,2],[1,2]] ]
        '''

        res_lsq, R, data, inc, mask, res_dict, start_dict, fixed_dict = fit

        Nmeasurements = len(inc)

        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)

        calc_dict = dict(**res_dict, **fixed_dict)

        estimates = self._calc_model(R, calc_dict)

        # !!IMPORTANT!!
        # calculate the residuals based on masked arrays
        masked_estimates = np.ma.masked_array(estimates, mask=mask)
        masked_data = np.ma.masked_array(data, mask=mask)

        res = np.ma.sqrt((masked_estimates - masked_data)**2)

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

        # plot mean residual
        axres.plot(np.arange(1, Nmeasurements + 1),
                   [np.ma.mean(res)] * Nmeasurements, 'k--')

        # add some legends
        res_h = mlines.Line2D(
            [], [], color='black', label='Mean res.  per measurement',
            linestyle='-', linewidth=3, marker='o', fillstyle='none')
        res_h_dash = mlines.Line2D(
            [], [], color='black', linestyle='--', label='Average mean res.',
            linewidth=1, fillstyle='none')

        res_h_dots = mlines.Line2D(
            [], [], color='black', label='Residuals',
            linestyle='-', linewidth=0, marker='.', alpha=0.5)

        handles, labels = axres.get_legend_handles_labels()
        axres.legend(handles=handles + [res_h_dots] + [res_h] + [res_h_dash],
                     loc=1)

        axres.set_ylabel('Residual')

        # set ticks

        if datelist is None:
            axres.set_xticks(np.arange(1, Nmeasurements + 1))
            axres.set_xlabel('# Measurement')
        else:
            axres.set_xticks(np.arange(1, Nmeasurements + 1))
            axres.set_xticklabels(np.concatenate(datelist[1]))

            locs = [len(i)/2 for i in datelist[1]]
            locmax = [len(i) for i in datelist[1]]
            locs = [locs[i] + np.sum(locmax[:i]) for i in range(len(locs))]

            for i, y in enumerate(datelist[0]):
                axres.annotate(s = str(y), xy = (locs[i], 0),
                               xytext = (0, -32), xycoords='data',
                               textcoords='offset points', va='top')

            axres.set_xlabel('# dates', labelpad = 20)


#        # evaluate mean residuals per incidence-angle
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



    def printscatter(self, fit, mima = None, pointsize = 0.5,
                     regression = True, **kwargs):
        '''
        geerate a scatterplot to investigate the quality of the fit

        Parameters:
        ------------
        fit : list
              output of monofit()-function
        Other Parameters:
        ------------------
        mima : list
               manual definition plot-boundaries via mima = [min, max]
        pointsize : float
                    manual specification of pointsize
        regression : bool (default = True)
                     indicator if the scipy.stats.linregress should be called
                     to get the regression-line and the r^2 value
        kwargs : -
                 kwargs passed to matplotlib.pyplot.scatter()
        '''

        res_lsq, R, data, inc, mask, res_dict, start_dict, fixed_dict = fit

        # reset incidence-angles in case they have been altered beforehand
        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        mask = mask

        calc_dict = dict(**res_dict, **fixed_dict)
        estimates = self._calc_model(R, calc_dict)

        # apply mask
        estimates = estimates[~mask]
        measures = data[~mask]

        if mima is None:
            mi = np.min((measures, estimates))
            ma = np.max((measures, estimates))
        else:
            mi, ma = mima

        ax.scatter(estimates, measures, s=pointsize, alpha = 0.7, **kwargs)

        # plot 45degree-line
        ax.plot([mi, ma], [mi, ma], 'k--')

        if self.sig0 is True:
            quantity = '$\sigma_0$'
        else:
            quantity = 'Intensity'

        if self.dB is True:
            scale = '[dB]'
        else:
            scale = ''

        ax.set_xlabel('measured ' + quantity + scale)
        ax.set_ylabel('modelled ' + quantity + scale)

        if regression is True:
            # evaluate linear regression to get r-value etc.
            slope, intercept, r_value, p_value, std_err = linregress(estimates,
                                                                     measures)

            ax.plot(np.sort(measures), intercept + slope*np.sort(measures), 'r--', alpha = 0.4)

            ax.text(0.8, .1,'$R^2$ = ' + str(np.round(r_value**2, 2)),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax.transAxes)

        return fig



    def printsingle(self, fit, measurements = None, datelist = None,
                    dates = None, hexbinQ = True, hexbinargs={}):
        '''
        a function to investigate the quality of the individual fits

        Parameters:
        ------------
        fit : list
              output of the monofit()-function
        measurements : list
                       a list containing the number of the measurement
                       that should be plotted
        Other Parameters:
        ------------------
        datelist : list
                   a list used to label the x-axis
                   the shape must be [ groups, indicators ]
                   where groups is a list of G values that are used to
                   group the measurements, and
                   indicators is a list containing N elements grouped into
                   G groups, where N is the number of measurements.

                   e.g. if you had a measurement in January and February
                   for the years 2014 and 2015, you have:
                       datelist = [[2014, 2015], [ [1,2],[1,2]] ]
        dates : list
                if datelist has been provided, the dates-list can be used to
                select the measurement that shall be plotted.
                The shape is : [[group_0, indicator_0],
                                [group_1, indicator_1],
                                ...]

                using the example for the datelist above, the second and fourth
                measurement can be selected via:
                    dates = [[2014, 2], [2015, 2]]

        hexbinQ : bool (default = False)
                  indicator if a hexbin-plot should be underlayed
                  that shows the distribution of the datapoints

        hexbinargs : dict
                     a dict containing arguments to customize the hexbin-plot
        '''

        res_lsq, R, data, inc, mask, res_dict, start_dict, fixed_dict = fit

        # reset incidence-angles in case they have been altered beforehand
        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)


        estimates = self._calc_model(R, dict(**res_dict, **fixed_dict))


        if datelist is None and dates is not None:
            assert False, 'you can only provide dates if dateslist is provided'


        if dates is not None:
            measurements = []

            for dd in dates:
                yval = np.argwhere(np.array(datelist[0]) == dd[0])[0][0]
                try:
                    mval = np.argwhere(np.array(datelist[1][yval]) ==
                                       dd[1])[0][0]
                    mval = (mval +
                            int(np.sum([len(i)for i in datelist[1][:yval]])))

                    measurements = measurements + [mval]
                except:
                    assert False, 'There is something wrong with the dates'

        # set colormaps to be used

        cmaps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']
        colors = ['blue', 'orange', 'green', 'red', 'purple']

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for m_i, m in enumerate(measurements):
            y = estimates[m][~mask[m]]

            # plot data
            if datelist is None:
                label = m
            else:
                monlen = 0
                for i, mon in enumerate(datelist[1]):
                    monlen = monlen + len(mon)
                    if m < monlen:
                        labelyear = datelist[0][i]
                        break

                label = (str(labelyear) + ' ' +
                         str(np.concatenate(datelist[1])[m]))

            xdata = np.rad2deg(inc[m][~mask[m]])
            ydata = data[m][~mask[m]]

            if hexbinQ is True:
                args = dict(gridsize = 5, mincnt = 1,
                            linewidths = 0., vmin=0.5, alpha = 0.5)
                args.update(hexbinargs)

                ax.hexbin(xdata, ydata, cmap = cmaps[m_i % len(cmaps)], **args)

            asdf, = ax.plot(xdata, ydata, '.',
                            color = colors[m_i % len(colors)], alpha = 1.,
                            label = label, markersize = 10)

            # plot results
            iii = inc[m][~mask[m]]
            ax.plot(np.rad2deg(iii[np.argsort(iii)]), y[np.argsort(iii)],
                    '-', color = 'w', linewidth = 3)

            ax.plot(np.rad2deg(iii[np.argsort(iii)]), y[np.argsort(iii)],
                    '-', color = asdf.get_color(), linewidth = 2)

            ax.set_xlabel('$\\theta_0$ [deg]')
            ax.set_ylabel('$\\sigma_0$ [dB]')


        ax.legend()

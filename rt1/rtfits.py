"""
Class to perform least_squares fitting of given datasets.
(wrapper for scipy.optimize.least_squares)
"""

import numpy as np
import sympy as sp

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

        # store original V and SRF
        orig_V = copy.deepcopy(R.RV)
        orig_SRF = copy.deepcopy(R.SRF)

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        # and generate a function to evaluate the symbolic representation
        try:
            tausymb = R.RV.tau[0].free_symbols
            taufunc = sp.lambdify(tausymb, R.RV.tau[0],
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.RV.omega[0].free_symbols
            omegafunc = sp.lambdify(omegasymb, R.RV.omega[0],
                                    modules=['numpy'])
        except Exception:
            omegasymb = set()
            omegafunc = None
        try:
            Nsymb = R.SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF[0],
                                modules=['numpy'])
        except Exception:
            Nsymb = set()
            Nfunc = None

        # a list of all symbols used to define tau, omega and NormBRDF
        toNlist = set(map(str, list(tausymb) + list(omegasymb) + list(Nsymb)))

        # update the numeric representations of omega, tau and NormBRDF
        # based on the values for the used symbols provided in res_dict
        if omegafunc is None:
            R.RV.omega = res_dict['omega']
        else:
            R.RV.omega = omegafunc(*[res_dict[str(i)] for i in omegasymb])
        if taufunc is None:
            R.RV.tau = res_dict['tau']
        else:
            R.RV.tau = taufunc(*[res_dict[str(i)] for i in tausymb])
        if Nfunc is None:
            R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = Nfunc(*[res_dict[str(i)] for i in Nsymb])

#        # set omega, tau and NormBRDF-values to input
#        if 'omega' in res_dict:
#            R.RV.omega = res_dict['omega']
#        if 'tau' in res_dict:
#            R.RV.tau = res_dict['tau']
#        if 'NormBRDF' in res_dict:
#            R.SRF.NormBRDF = res_dict['NormBRDF']

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input (i.e. "omega", "tau", "NormBRDF"
        # and the symbols used to define them must be removed)
        param_fn = res_dict.copy()
        param_fn.pop('omega', None)
        param_fn.pop('tau', None)
        param_fn.pop('NormBRDF', None)
        for i in toNlist:
            param_fn.pop(str(i), None)

        # ensure that the keys of the dict are strings and not sympy-symbols
        strparam_fn = dict([[str(key),
                             np.expand_dims(param_fn[key], 1)]
                            for i, key in enumerate(param_fn.keys())])

        # set the param-dict to the newly generated dict
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

        # restore V and SRF to original values
        R.RV = orig_V
        R.SRF = orig_SRF

        return model_calc

    # function to evaluate the jacobian
    def _calc_jac(self, R, res_dict, param_dyn_dict, order):
        '''
        function to evaluate the jacobian in the shape as required
        by scipy's least_squares function

        Parameters:
        ------------
        R : RT1-object
            the rt1-object for which the results shall be calculated
        res_dict : dict
                   a dictionary containing all parameter-values that should
                   be updated before calling R.jac()
        Returns:
        --------
        jac : array_like(float)
              the jacobian corresponding to the fit-parameters in the
              shape applicable to scipy's least_squres-function
        '''
        # store original V and SRF
        orig_V = copy.deepcopy(R.RV)
        orig_SRF = copy.deepcopy(R.SRF)

#        # set omega, tau and NormBRDF-values to input
#        if 'omega' in res_dict:
#            R.RV.omega = res_dict['omega']
#        if 'tau' in res_dict:
#            R.RV.tau = res_dict['tau']
#        if 'NormBRDF' in res_dict:
#            R.SRF.NormBRDF = res_dict['NormBRDF']

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        try:
            tausymb = R.RV.tau[0].free_symbols
            taufunc = sp.lambdify(tausymb, R.RV.tau[0],
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.RV.omega[0].free_symbols
            omegafunc = sp.lambdify(omegasymb, R.RV.omega[0],
                                    modules=['numpy'])
        except Exception:
            omegasymb = set()
            omegafunc = None
        try:
            Nsymb = R.SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF[0],
                                modules=['numpy'])
        except Exception:
            Nsymb = set()
            Nfunc = None

        toNlist = set(map(str, list(tausymb) + list(omegasymb) + list(Nsymb)))

        # update the numeric representations of omega, tau and NormBRDF
        # based on the values for the used symbols provided in res_dict
        if omegafunc is None:
            R.RV.omega = res_dict['omega']
        else:
            R.RV.omega = omegafunc(*[res_dict[str(i)] for i in omegasymb])
        if taufunc is None:
            R.RV.tau = res_dict['tau']
        else:
            R.RV.tau = taufunc(*[res_dict[str(i)] for i in tausymb])
        if Nfunc is None:
            R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = Nfunc(*[res_dict[str(i)] for i in Nsymb])

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input (i.e. "omega", "tau", "NormBRDF"
        # and the symbols used to define them must be removed)
        param_fn = res_dict.copy()
        param_fn.pop('omega', None)
        param_fn.pop('tau', None)
        param_fn.pop('NormBRDF', None)
        for i in toNlist:
            param_fn.pop(str(i), None)

        # ensure that the keys of the dict are strings
        strparam_fn = dict([[str(key),
                             np.expand_dims(param_fn[key], 1)]
                            for i, key in enumerate(param_fn.keys())])

        # set the param-dict to the newly generated dict
        R.param_dict = strparam_fn

        neworder = [o for o in order]

        # if tau, omega or NormBRDF have been provided in terms of symbols,
        # remove the symbols that are intended to be fitted (that are also
        # in param_dyn_dict) and replace them by 'omega', 'tau' and 'NormBRDF'
        # so that calling R.jacobian will calculate the "outer" derivative
        if len(tausymb) != 0:
            for i in map(str, tausymb) & param_dyn_dict.keys():
                neworder[neworder.index(i)] = 'tau'
        if len(omegasymb) != 0:
            for i in map(str, omegasymb) & param_dyn_dict.keys():
                neworder[neworder.index(i)] = 'omega'
        if len(Nsymb) != 0:
            for i in map(str, Nsymb) & param_dyn_dict.keys():
                neworder[neworder.index(i)] = 'NormBRDF'

        # calculate the jacobian based on neworder
        # (evaluating only "outer" derivatives with respect to omega,
        # tau and NormBRDF)
        jac = R.jacobian(sig0=self.sig0, dB=self.dB,
                         param_list=neworder)

        # remove unwanted columns from the jacobian
        splitjac = np.split(np.concatenate(jac), len(order))
        newjacdict = {}
        for i, key in enumerate(order):
            newjacdict[key] = np.zeros_like(
                    splitjac[i][:len(np.unique(param_dyn_dict[key]))])
            col = 0
            for n in np.unique(param_dyn_dict[key]):
                rule = (param_dyn_dict[key] == n)
                newjacdict[key][col] = np.sum(splitjac[i][rule], axis=0)
                col = col + 1

        # evaluate jacobians of the functional representations of tau, omega
        # and NormBRDF and add them to newjacdict
        for i in map(str, tausymb) & param_dyn_dict.keys():
            # generate a function that evaluates the 'inner' derivative, i.e.:
            # df/dx = df/dtau * dtau/dx = df/dtau * d_inner
            d_inner = sp.lambdify(tausymb, sp.diff(orig_V.tau[0], i),
                                  modules=['numpy'])
            # evaluate the inner derivative
            dtau_dx = d_inner(*[res_dict[str(i)] for i in tausymb])
            # calculate the derivative with respect to the parameters
            # TODO this is a bit sloppy...
            if not np.isscalar(dtau_dx):
                dtau_dx = np.concatenate(
                    [dtau_dx] * len(np.atleast_2d(R.t_0)[0]))

            # calculate "outer" * "inner" derivative
            newjacdict[str(i)] = newjacdict[str(i)] * dtau_dx

        for i in map(str, omegasymb) & param_dyn_dict.keys():
            # generate a function that evaluates the 'inner' derivative, i.e.:
            # df/dx = df/dtau * dtau/dx = df/dtau * d_inner
            d_inner = sp.lambdify(omegasymb, sp.diff(orig_V.omega[0], i),
                                  modules=['numpy'])
            # evaluate the inner derivative
            domega_dx = d_inner(*[res_dict[str(i)] for i in omegasymb])
            # calculate the derivative with respect to the parameters
            # TODO this is a bit sloppy...
            if not np.isscalar(domega_dx):
                domega_dx = np.concatenate(
                        [domega_dx]*len(np.atleast_2d(R.t_0)[0]))

            # calculate "outer" * "inner" derivative
            newjacdict[str(i)] = newjacdict[str(i)] * domega_dx

        for i in map(str, Nsymb) & param_dyn_dict.keys():
            # generate a function that evaluates the 'inner' derivative, i.e.:
            # df/dx = df/dtau * dtau/dx = df/dtau * d_inner
            d_inner = sp.lambdify(Nsymb, sp.diff(orig_SRF.NormBRDF[0], i),
                                  modules=['numpy'])
            # evaluate the inner derivative
            dN_dx = d_inner(*[res_dict[str(i)] for i in Nsymb])
            # calculate the derivative with respect to the parameters
            # TODO this is a bit sloppy...
            if not np.isscalar(dN_dx):
                dN_dx = np.concatenate(
                        [dN_dx]*len(np.atleast_2d(R.t_0)[0]))

            # calculate "outer" * "inner" derivative
            newjacdict[str(i)] = newjacdict[str(i)] * dN_dx

        # return the transposed jacobian as needed by scipy's least_squares
        jac_lsq = np.concatenate([newjacdict[key] for key in order]).T

        # restore V and SRF to original values
        R.RV = orig_V
        R.SRF = orig_SRF

        return jac_lsq

    def monofit(self, V, SRF, dataset, param_dict,
                bounds_dict={}, fixed_dict={}, param_dyn_dict={},
                fn=None, _fnevals=None, int_Q=True,
                lambda_backend='cse', **kwargs):
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
        param_dyn_dict : dict
                     A dictionary containing the names of the parameters that
                     are intended to be fitted together with a list of unique
                     integers for each key, specifying the number of individual
                     parameters that shall be fitted.

                     For example (Number of measurements = 4):

                         - param_dyn_dict = {'tau' : [1, 2, 3, 4]}
                           results in 4 distinct values for tau
                         - param_dyn_dict = {'tau' : [1, 1, 1, 1]}
                           results in a single value for tau.
                         - param_dyn_dict = {'tau' : [1, 2, 2, 1]}
                           results in two values for tau, where the first
                           value is used in the fit of the fist and last
                           measurement, and the second value is used in the
                           fit of the second and third measurement.

                    If param_dyn_dict is not set explicitly ( = {}),
                    the number of start-values in param_dict will be used to
                    generate an appropriate dictionary. i.e.:

                        - param_dict = {'tau' : .3}
                          results in param_dyn_dict = {'tau' : [1, 1, 1, 1]}
                        - param_dict = {'tau' : [.2, .2, .2, .2]}
                          results in param_dyn_dict = {'tau' : [1, 2, 3, 4]}

                          (no distinction between equal and varying
                          start-values is implemented...)


        Other Parameters:
        ------------------
        fn : array-like
             a slot for pre-calculated fn-coefficients.
             if the same model has to be fitted to multiple datasets, the
             fn-coefficients that are returned in the first fit can be used
             as input for the second fit to avoid repeated calculations.
        _fnevals : callable
             a slot for pre-compiled function to evaluate the fn-coefficients
             Note that once the _fnevals function is provided, the
             fn-coefficients are no longer needed and have no effect on the
             calculated results!
        int_Q : bool (default = True)
                indicator if interaction-terms should be included
                (note: they are always ommitted when calculating the jacobian!)
        lambda_backend : string (default = 'cse')
                         select method for generating _fnevals functions
                         if they are not provided explicitly
        kwargs :
                 keyword arguments passed to scipy's least_squares function

        Returns:
        ---------
        res_lsq : dict
                  output of scipy's least_squares function
        R : RT1-object
            the RT1-object used to perform the fit
        data : array-like
               used dataset for the fit
        inc : array-like
              used incidence-angle data for the fit
        mask : array-like(bool)
               the masked that needs to be applied to the rectangularized
               dataset to get the valid entries (see preparedata-function)
        weights : array-like
                  the weighting-matrix that has been applied to correct for the
                  rectangularization of the dataset (see preparedata-function)
        res_dict : dict
                   a dictionary containing the fit-results for the parameters
        start_dict : dict
                     a dictionary containing the used start-values
        fixed_dict : dict
                   a dictionary containing the parameter-values that have been
                   used as constants during the fit
        '''

        # generate a list of the names of the parameters that will be fitted.
        # (this is necessary to ensure correct broadcasting of values since
        # dictionarys do)
        order = [i for i, v in param_dict.items() if v is not None]
        # preparation of data for fitting
        inc, data, weights, Nmeasurements, mask = self._preparedata(dataset)

        # get values for omega, tau and NormBRDF from parameter-dictionary
#        omega = param_dict.get('omega', None)
#        tau = param_dict.get('tau', None)
#        NormBRDF = param_dict.get('NormBRDF', None)

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        try:
            tausymb = V.tau[0].free_symbols
            taufunc = sp.lambdify(tausymb, V.tau[0], modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = V.omega[0].free_symbols
            omegafunc = sp.lambdify(omegasymb, V.omega[0], modules=['numpy'])
        except Exception:
            omegasymb = set()
            omegafunc = None
        try:
            Nsymb = SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, SRF.NormBRDF[0], modules=['numpy'])
        except Exception:
            Nsymb = set()
            Nfunc = None

        toNlist = set(map(str, list(tausymb) + list(omegasymb) + list(Nsymb)))

        # check of general input-requirements
        #   check if all parameters have been provided
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, V._func.free_symbols)) - angset
        srfsymb = set(map(str, SRF._func.free_symbols)) - angset

        paramset = ((set(map(str, param_dict.keys()))
                     ^ set(map(str, fixed_dict.keys())))
                    - {'tau', 'omega', 'NormBRDF'})

        assert paramset >= (vsymb | srfsymb), (
            'the parameters ' +
            str((vsymb | srfsymb) - paramset) +
            ' must be provided in param_dict')

# TODO fix asserts
#        if omega is not None and not np.isscalar(omega):
#            assert len(omega) == Nmeasurements, ('len. of omega-array must' +
#                      'be equal to the length of the dataset')
#        if omega is None:
#            assert len(V.omega) == Nmeasurements, ('length of' +
#                      ' omega-array provided in the definition of V must' +
#                      ' be equal to the length of the dataset')
#
#        if tau is not None and not np.isscalar(tau):
#            assert len(tau) == Nmeasurements, ('length of tau-array' +
#                      ' must be equal to the length of the dataset')
#
#        if tau is None:
#            assert len(V.tau) == Nmeasurements, ('length of tau-array' +
#                      ' provided in the definition of V must be equal to' +
#                      ' the length of the dataset')
#
#        if NormBRDF is not None and not np.isscalar(NormBRDF):
#            assert len(NormBRDF) == Nmeasurements, ('length of' +
#                      ' NormBRDF-array must be equal to the' +
#                      ' length of the dataset')
#        if NormBRDF is None:
#            assert len(SRF.NormBRDF) == Nmeasurements, ('length of' +
#                      ' NormBRDF-array provided in the definition of SRF' +
#                      ' must be equal to the length of the dataset')
#

        # generate a dict containing only the parameters needed to evaluate
        # the fn-coefficients
        # for python > 3.4
        # param_R = dict(**param_dict, **fixed_dict)
        param_R = dict((k, v) for k, v in list(param_dict.items())
                       + list(fixed_dict.items()))

        param_R.pop('omega', None)
        param_R.pop('tau', None)
        param_R.pop('NormBRDF', None)
        # remove also other symbols that are used in the definitions of
        # tau, omega and NormBRDF
        for i in toNlist:
            param_R.pop(i)

        if fn is None:
            # define rt1-object
            R = RT1(1., 0., 0., 0., 0.,
                    RV=V, SRF=SRF, fn=None, geometry='mono',
                    param_dict=param_R, int_Q=int_Q,
                    lambda_backend=lambda_backend)

            # set geometry
            R.t_0 = inc
            R.p_0 = np.zeros_like(inc)
            R.t_ex = inc
            R.p_ex = np.full_like(inc, np.pi)

            fn = R.fn
            _fnevals = R._fnevals
        else:
            if _fnevals is None:
                # define rt1-object
                R = RT1(1., inc, inc, np.zeros_like(inc),
                        np.full_like(inc, np.pi), RV=V, SRF=SRF, fn=fn,
                        geometry='mono', param_dict=param_R, int_Q=int_Q,
                        lambda_backend=lambda_backend)

                _fnevals = R._fnevals
            else:
                # define rt1-object
                R = RT1(1., inc, inc, np.zeros_like(inc),
                        np.full_like(inc, np.pi), RV=V, SRF=SRF, fn=fn,
                        _fnevals=_fnevals, geometry='mono',
                        param_dict=param_R, int_Q=int_Q,
                        lambda_backend=lambda_backend)

        # if param_dyn_dict is not set explicitly, use the number of
        # start-values provided in param_dict to assign the dynamics of
        # the parameters (i.e. either constant or varying for each measurement)
        if param_dyn_dict == {}:
            for key in param_dict:
                param_dyn_dict[key] = np.linspace(
                    1,
                    len(np.atleast_1d(param_dict[key])),
                    Nmeasurements)

        # define a function that evaluates the model in the shape as needed
        # for scipy's least_squares function
        def fun(params):
            # generate a dictionary to assign values based on input
            count = 0
            newdict = {}
            for i, key in enumerate(order):
                # find unique parameter estimates and how often they occur
                uniques, index, inverse, Nuniq = np.unique(param_dyn_dict[key],
                                                           return_counts=True,
                                                           return_index=True,
                                                           return_inverse=True)

                # shift index to next parameter (this is necessary since params
                # is provided as a concatenated array)
                inverse = inverse + count
                # select the fitted values for the corresponding parameter
                newdict[key] = np.array(params)[inverse]
                # increase counter
                count = count + len(np.unique(param_dyn_dict[key]))

            # incorporate values provided in fixed_dict
            # (i.e. incorporate fixed but possibly dynamic parameter-values)

            # for python > 3.4
            # newdict = dict(newdict, **fixed_dict)
            newdict = dict(list(newdict.items()) +
                           list(fixed_dict.items()))

            # calculate the residuals
            errs = np.concatenate(self._calc_model(R, newdict)) - data
            # incorporate weighting-matrix to ensure correct treatment
            # of artificially added values (see _preparedata()-fucntion)
            errs = weights * errs

            return errs

        # function to evaluate the jacobian
        def dfun(params):
            # generate a dictionary to assign values based on input
            count = 0
            newdict = {}
            for i, key in enumerate(order):
                # find unique parameter estimates and how often they occur
                uniques, index, inverse, Nuniq = np.unique(param_dyn_dict[key],
                                                           return_counts=True,
                                                           return_index=True,
                                                           return_inverse=True)

                # shift index to next parameter (this is necessary since params
                # is provided as a concatenated array)
                inverse = inverse + count
                # select the fitted values for the corresponding parameter
                newdict[key] = np.array(params)[inverse]
                # increase counter
                count = count + len(np.unique(param_dyn_dict[key]))

            # incorporate values provided in fixed_dict
            # (i.e. incorporate fixed but possibly dynamic parameter-values)
            # for python > 3.4
            # newdict = dict(newdict, **fixed_dict)
            newdict = dict(list(newdict.items()) +
                           list(fixed_dict.items()))

            # calculate the jacobian
            # (no need to include weighting matrix in here since the jacobian
            # of the artificially added colums must be the same!)
            jac = self._calc_jac(R, newdict, param_dyn_dict, order)

            return jac

        # TODO define boundaries for omega, tau and NormBRDF if none have been
        # provided explicitly
#        omega_bounds = bounds_dict.get('omega', None)
#        tau_bounds = bounds_dict.get('tau', None)
#        NormBRDF_bounds = bounds_dict.get('NormBRDF', None)
#
#        if omega is not None:
#            if omega_bounds is None:
#                if np.isscalar(omega):
#                    omega_bounds = ([0.], [1.])
#                else:
#                    omega_bounds = ([0.] * Nmeasurements,
#                                    [1.] * Nmeasurements)
#        else:
#            omega_bounds = ([], [])
#
#        if tau is not None:
#            if tau_bounds is None:
#                if np.isscalar(tau):
#                    tau_bounds = ([0.], [1.])
#                else:
#                    tau_bounds = ([0.] * Nmeasurements,
#                                  [1.] * Nmeasurements)
#        else:
#            tau_bounds = ([], [])
#
#        if NormBRDF is not None:
#            if NormBRDF_bounds is None:
#                if np.isscalar(NormBRDF):
#                    NormBRDF_bounds = ([0.], [1.])
#                else:
#                    NormBRDF_bounds = ([0.] * Nmeasurements,
#                                       [1.] * Nmeasurements)
#        else:
#            NormBRDF_bounds = ([], [])
#
#        bounds_dict['omega'] = omega_bounds
#        bounds_dict['tau'] = tau_bounds
#        bounds_dict['NormBRDF'] = NormBRDF_bounds

        # generate list of boundary conditions as needed for the fit
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
        count = 0
        res_dict = {}
        start_dict = {}
        for i, key in enumerate(order):
            # find unique parameter estimates and how often they occur
            uniques, index, inverse, Nuniq = np.unique(param_dyn_dict[key],
                                                       return_counts=True,
                                                       return_index=True,
                                                       return_inverse=True)

            # shift index to next parameter (this is necessary since the result
            # is provided as a concatenated array)
            inverse = inverse + count
            # select the fitted values for the corresponding parameter
            res_dict[key] = np.array(res_lsq.x)[inverse]
            start_dict[key] = np.array(startvals)[inverse]
            # increase counter
            count = count + len(np.unique(param_dyn_dict[key]))

        # ------------------------------------------------------------------
        # ------------ prepare output-data for later convenience -----------

        # get the data in the same shape as the incidence-angles
        data = np.array(np.split(data, Nmeasurements))

        return [res_lsq, R, data, inc, mask, weights,
                res_dict, start_dict, fixed_dict]

    def printresults(self, fit, truevals=None, startvals=False,
                     datelist=None, legends=True):
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

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        # reset incidence-angles in case they have been altered beforehand
        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)

        # evaluate number of measurements
        Nmeasurements = len(inc)

        if truevals is not None:
            truevals = copy.copy(truevals)

            # generate a dictionary to assign values based on input
            for key in truevals:
                if np.isscalar(truevals[key]):
                    truevals[key] = np.array([truevals[key]] * Nmeasurements)
                else:
                    truevals[key] = truevals[key]

        # generate figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(211)
        ax.set_title('Fit-results')

        # plot datapoints
        for i, j in enumerate(np.ma.masked_array(data, mask)):
            ax.plot(inc[i], j, '.')

        # reset color-cycle
        plt.gca().set_prop_cycle(None)

        # define incidence-angle range for plotting
        incplot = np.array([np.linspace(np.min(inc), np.max(inc), 100)]
                           * Nmeasurements)
        # set new incidence-angles
        R.t_0 = incplot
        R.p_0 = np.zeros_like(incplot)

        # get parameter-values
        # for python > 3.4
        # calc_dict = dict(**res_dict, **fixed_dict)
        calc_dict = dict((k, v) for k, v in list(res_dict.items())
                         + list(fixed_dict.items()))

        # calculate results
        fitplot = self._calc_model(R, calc_dict)

        # generate a mask that hides all measurements where no data has
        # been provided (i.e. whose parameter-results are still the startvals)
        newmask = np.ones_like(incplot) * np.all(mask, axis=1)[:, np.newaxis]
        fitplot = np.ma.masked_array(fitplot, newmask)

        for i, val in enumerate(fitplot):
            ax.plot(incplot[i], val, alpha=0.4, label=i + 1)

        # ----------- plot start-values ------------
        if startvals is True:
            startplot = self._calc_model(R, start_dict)
            for i, val in enumerate(startplot):
                ax.plot(incplot[i], val, 'k--', linewidth=1,
                        alpha=0.5, label='fitstart')

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
            ax2.plot(np.arange(1, Nmeasurements + 1),
                     np.ma.masked_array(res_dict[key], np.all(mask, axis=1)),
                     alpha=1., label=key)
#            ax2.plot(np.arange(1, Nmeasurements + 1),
#                     np.ma.masked_array(res_dict[key], np.all(mask, axis=1)),
#                     'k.', alpha=0.5)
        plt.gca().set_prop_cycle(None)

        h1 = mlines.Line2D([], [], color='black', label='estimates',
                           linestyle='-', alpha=0.75, marker='.')

        handles, labels = ax2.get_legend_handles_labels()
        if truevals is None:
            plt.legend(handles=handles + [h1], loc=1)
        else:
            plt.legend(handles=handles + [h1, h2, h3], loc=1)

        # set ticks
        if datelist is None:
            ax2.set_xticks(np.arange(1, Nmeasurements + 1))
            plt.xlabel('# Measurement')
        else:
            ax2.set_xticks(np.arange(1, Nmeasurements + 1))
            ax2.set_xticklabels(np.concatenate(datelist[1]))

            locs = [len(i)/2 for i in datelist[1]]
            locmax = [len(i) for i in datelist[1]]
            locs = [locs[i] + np.sum(locmax[:i]) for i in range(len(locs))]

            for i, y in enumerate(datelist[0]):
                plt.annotate(s=str(y), xy=(locs[i], 0), xytext=(0, -20),
                             xycoords='data', textcoords='offset points',
                             va='top')

            plt.xlabel('# dates', labelpad=20)

        if truevals is None:
            plt.ylabel('Parameters')
        else:
            plt.ylabel('Parameters / Errors')

        fig.tight_layout()

        return fig

    def printerr(self, fit, datelist=None, newcalc=False):
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
        newcalc : bool (default = False)
                  indicator whether the residuals shall be re-calculated
                  or not.

                  True:
                      the residuals are calculated using R, inc, mask,
                      res_dict and fixed_dict from the fit-argument
                  False:
                      the residuals are taken from the output of
                      res_lsq from the fit-argument
        '''

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        Nmeasurements = len(inc)

        if newcalc is False:
            # get residuals from fit into desired shape for plotting
            # Attention -> incorporate weights and mask !
            res = np.ma.masked_array(np.reshape(
                    np.abs(res_lsq.fun/weights), data.shape), mask)
        else:
            # Alternative way of calculating the residuals
            # (based on R, inc and res_dict)

            R.t_0 = inc
            R.p_0 = np.zeros_like(inc)

            # for python > 3.4
            # calc_dict = dict(**res_dict, **fixed_dict)
            calc_dict = dict((k, v) for k, v in list(res_dict.items())
                             + list(fixed_dict.items()))

            estimates = self._calc_model(R, calc_dict)
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

        # the use of masked arrays might cause python 2 compatibility issues!
        axres.plot(np.arange(len(res)) + 1, res, '.', alpha=0.5)

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
                axres.annotate(s=str(y), xy=(locs[i], 0),
                               xytext=(0, -32), xycoords='data',
                               textcoords='offset points', va='top')

            axres.set_xlabel('# dates', labelpad=20)


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


    def printscatter(self, fit, mima=None, pointsize=0.5,
                     regression=True, **kwargs):
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

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        # reset incidence-angles in case they have been altered beforehand
        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        mask = mask

        # for python > 3.4
        # calc_dict = dict(**res_dict, **fixed_dict)
        calc_dict = dict((k, v) for k, v in list(res_dict.items())
                         + list(fixed_dict.items()))

        estimates = self._calc_model(R, calc_dict)

        # apply mask
        estimates = estimates[~mask]
        measures = data[~mask]

        if mima is None:
            mi = np.min((measures, estimates))
            ma = np.max((measures, estimates))
        else:
            mi, ma = mima

        ax.scatter(estimates, measures, s=pointsize, alpha=0.7, **kwargs)

        # plot 45degree-line
        ax.plot([mi, ma], [mi, ma], 'k--')

        if self.sig0 is True:
            quantity = r'$\sigma_0$'
        else:
            quantity = 'Intensity'

        if self.dB is True:
            scale = '[dB]'
        else:
            scale = ''

        ax.set_xlabel('modelled ' + quantity + scale)
        ax.set_ylabel('measured ' + quantity + scale)

        if regression is True:
            # evaluate linear regression to get r-value etc.
            slope, intercept, r_value, p_value, std_err = linregress(estimates,
                                                                     measures)

            ax.plot(np.sort(measures),
                    intercept + slope * np.sort(measures), 'r--', alpha=0.4)

            ax.text(0.8, .1, '$R^2$ = ' + str(np.round(r_value**2, 2)),
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

        return fig, r_value**2

    def printsingle(self, fit, measurements=None, datelist=None,
                    dates=None, hexbinQ=True, hexbinargs={}):
        '''
        a function to investigate the quality of the individual fits

        Parameters:
        ------------
        fit : list
              output of the monofit()-function
        measurements : list
                       a list containing the number of the measurement
                       that should be plotted (starting from 1)

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

        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.colors import Normalize

        # function to generate colormap that fades between colors
        def CustomCmap(from_rgb, to_rgb):

            # from color r,g,b
            r1, g1, b1 = from_rgb

            # to color r,g,b
            r2, g2, b2 = to_rgb

            cdict = {'red': ((0, r1, r1),
                             (1, r2, r2)),
                     'green': ((0, g1, g1),
                               (1, g2, g2)),
                     'blue': ((0, b1, b1),
                              (1, b2, b2))}

            cmap = LinearSegmentedColormap('custom_cmap', cdict)
            return cmap

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        # reset incidence-angles in case they have been altered beforehand
        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)

        # for python > 3.4
        # calc_dict = dict(**res_dict, **fixed_dict)
        calc_dict = dict((k, v) for k, v in list(res_dict.items())
                         + list(fixed_dict.items()))

        estimates = self._calc_model(R, calc_dict)

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
                except Exception:
                    # TODO this is a stupid way to handle this
                    assert False, 'There is something wrong with the dates'
        else:
            # since python starts counting at 0 ...
            measurements = np.array(measurements) - 1

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for m_i, m in enumerate(measurements):
            y = estimates[m][~mask[m]]
            # plot data
            if datelist is None:
                label = m + 1
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

            # get color that will be applied to the next line drawn
            dummy, = ax.plot(xdata[0], ydata[0], '.', alpha=0.)
            color = dummy.get_color()

            if hexbinQ is True:
                args = dict(gridsize=15, mincnt=1,
                            linewidths=0., vmin=0.5, alpha=0.7)
                args.update(hexbinargs)

                # evaluate the hexbinplot once to get the maximum number of
                # datapoints within a single hexagonal (used for normalization)
                dummyargs = args.copy()
                dummyargs.update({'alpha': 0.})
                hb = ax.hexbin(xdata, ydata, **dummyargs)

                # generate colormap that fades from white to the color
                # of the plotted data  (asdf.get_color())
                cmap = CustomCmap([1.00, 1.00, 1.00],
                                  plt.cm.colors.hex2color(color))
                # setup correct normalizing instance
                norm = Normalize(vmin=0, vmax=hb.get_array().max())

                ax.hexbin(xdata, ydata, cmap=cmap, norm=norm, **args)

            # plot datapoints
            asdf, = ax.plot(xdata, ydata, '.',
                            color=color, alpha=1.,
                            label=label, markersize=10)

            # plot results
            iii = inc[m][~mask[m]]
            ax.plot(np.rad2deg(iii[np.argsort(iii)]), y[np.argsort(iii)],
                    '-', color='w', linewidth=3)

            ax.plot(np.rad2deg(iii[np.argsort(iii)]), y[np.argsort(iii)],
                    '-', color=asdf.get_color(), linewidth=2)

            ax.set_xlabel('$\\theta_0$ [deg]')
            ax.set_ylabel('$\\sigma_0$ [dB]')

        if dates is None:
            ax.legend(title='# Measurement')
            # plt.setp(legend.get_title(),fontsize=8) # change fontsize
        else:
            ax.legend(title='# Date')


    def printseries(self, fit, index=None, legends=True, minmax=None):
        '''
        a function to quickly print the fit-results and the gained parameters

        Parametsrs:
        ------------
        fit : list
              output of monofit_all()-function
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
        import pandas as pd

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        if minmax is None:
            minmax = [0, len(data)]

        # for python > 3.4
        # calc_dict = dict(**res_dict, **fixed_dict)
        calc_dict = dict((k, v) for k, v in list(res_dict.items())
                         + list(fixed_dict.items()))

        estimates = self._calc_model(R, calc_dict)

        maskedestimates = np.ma.masked_array(estimates,
                                             mask)[minmax[0]:minmax[1]]

        # get number of measurement as index
        Nvals = np.ones_like(maskedestimates,
                             dtype=int) * (np.arange(*minmax))[:, np.newaxis]
        Nvals = np.ma.concatenate(Nvals)
        sorts = np.ma.argsort(Nvals)

        maskedestimates = np.concatenate(maskedestimates)
        maskeddata = np.ma.concatenate(
                np.ma.masked_array(data, mask)[minmax[0]:minmax[1]])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if index is None:
            Nmeasurements = len(inc)
            ax.set_xticks(np.arange(1, Nmeasurements + 1))
            ax.set_xlabel('# Measurement')

            ax.plot(Nvals[sorts], maskeddata[sorts], '-',
                    label='data', marker='.')
            ax.plot(Nvals[sorts], maskedestimates[sorts], '-',
                    label='model', marker='.', alpha=0.5)

        else:
            import matplotlib.dates as dates
            from datetime import timedelta

            fullindex = pd.DatetimeIndex(
                np.concatenate([[i] * len(data[0]) for i in index.values]))

            ax.plot(fullindex,
                    np.ma.concatenate(np.ma.masked_array(data, mask)),
                    '-', label='data', marker='.')

            ax.plot(fullindex,
                    np.ma.concatenate(np.ma.masked_array(estimates, mask)),
                    '-', label='model', marker='.', alpha=0.5)

            ax.set_xlim([index[0] - timedelta(days=20),
                         index[-1] + timedelta(days=20)])

            ax.xaxis.set_minor_locator(dates.MonthLocator())
            ax.xaxis.set_minor_formatter(dates.DateFormatter('%m'))

            ax.xaxis.set_major_locator(dates.YearLocator())
            ax.xaxis.set_major_formatter(dates.DateFormatter('\n%Y'))

        ax.set_xlabel('Measurements')
        ax.set_ylabel('$\\sigma_0$ [dB]')

        if legends is True:
            plt.legend()

        return fig

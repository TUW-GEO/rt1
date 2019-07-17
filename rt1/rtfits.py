"""
Class to perform least_squares fitting of given datasets.
(wrapper for scipy.optimize.least_squares)
"""

import numpy as np
import sympy as sp
try:
    import pandas as pd
except ImportError:
    print('pandas could not be found! ... performfit() and generate_dyn_dict()\
           will not work!')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from scipy.optimize import least_squares
from scipy.stats import linregress
from scipy.sparse import vstack
from scipy.sparse import csr_matrix, isspmatrix

from .scatter import Scatter
from .rt1 import RT1

from functools import partial, update_wrapper
from .rtplots import printsig0timeseries

import copy  # used to copy objects
import datetime


def meandatetime(datetimes):
    '''
    calculate the average date from a given list of datetime-objects
    (can be applied to a pandas-Series via Series.apply(meandatetime))

    Parameters:
    ------------
    datetimes : list
                a list of datetime-objects
    Returns:
    ---------
    meandate : Timestamp
               the center-date
    '''

    if np.count_nonzero(datetimes) == 1:
        return datetimes

    x = pd.to_datetime(datetimes)
    deltas = x[0] - x[1:]
    meandelta = sum(deltas, datetime.timedelta(0))/len(x)
    meandate = x[0] - meandelta
    return meandate


class Fits(Scatter):
    '''
    Class to perform nonlinear least-squares fits to data.

    Parameters:
    ------------
    sig0 : boolean (default = False)
           Indicator whether dataset is given as sigma_0-values (sig_0) or as
           intensity-values (I). The applied relation is:
               sig_0 = 4. * np.pi * np.cos(inc) * I
           where inc is the corresponding incident zenith-angle.
    dB : boolean (default = False)
         Indicator whether dataset is given in linear units or in dB.
         The applied relation is:    x_dB = 10. * np.log10( x_linear )
    dataset : pandas.DataFrame (default = None)
              a pandas.DataFrame with columns 'inc' and 'sig' defined
              where 'inc' referrs to the incidence-angle in radians, and
              'sig' referrs to the measurement value (corresponding to
              the assigned sig0 and dB values)
    defdict : dict (default = None)
              a dictionary of the following structure:
              (the dict will be copied internally using copy.deepcopy(dict))

              >>> defdict = {'key1' : [fitQ, val, freq, ([min], [max])],
              >>>            'key2' : [fitQ, val, freq, ([min], [max])],
              >>>            ...}

              where all keys required to call set_V_SRF must be defined
              and the values are defined via:
                  fitQ : bool
                         indicator if the quantity should be fitted (True)
                         or used as a constant during the fit (False)
                  val : float or array
                        if fitQ is True, val will be used as start-value
                        if fitQ is False, val will be used as constant.
                        Notice: if val is an array, symbolic evaluation
                        of the corresponding parameter is necessary in order
                        to generate a function that can handle array-inputs!
                  freq : the frequency of the fit-parameter as a string
                         (see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)
                         (only needed if fitQ is True)
                  min, max : float
                             the boundary-conditions for the parameter
                             (only needed if fitQ is True)
    set_V_SRF : callable (default = None)
                function with the following structure:

                >>> def set_V_SRF(volume-keys, surface-keys):
                >>>     from rt1.volume import 'Volume-function'
                >>>     from rt1.surface import 'Surface function'
                >>>
                >>>     V = Volume-function(volume-keys)
                >>>     SRF = Surface-function(surface-keys)
                >>>
                >>>     return V, SRF
    '''

    def __init__(self, sig0=False, dB=False, dataset=None,
                 defdict=None, set_V_SRF=None, **kwargs):
        self.sig0 = sig0
        self.dB = dB
        self.dataset = dataset
        self.set_V_SRF = set_V_SRF
        self.defdict = copy.deepcopy(defdict)
        self.result = None

        self.printsig0timeseries = partial(printsig0timeseries, fit = self)
        update_wrapper(self.printsig0timeseries, printsig0timeseries)


    def generate_dyn_dict(self, param_keys, datetime_index,
                          freq=None, freqkeys=[],
                          ):
        '''
        Generate a dictionary to assign the temporal dynamics of the variables.
        Any key in 'param_keys' that is not assigned in freqkeys will be
        treated as a constant.

        Parameter:
        -------------
        param_keys : an iterable of keys corresponding to the names
                     of the parameters that are intended to be fitted
        datetime_index : datetime-indexe list of the measurements
        freq : list
               a list of frequencies used for assigning the temporal
               variability of the parameters ('D', 'M', '5D', etc.)
        freqkeys : list
               a list of parameter-names that will be assigned to the freq-list
               (freqkeys = [['daily_p1', 'daily_p2'], ['monthly_p1']]
                freq = ['D', 'M'])

        Returns:
        ---------
        param_dyn_dict : dict
                         a dictionary that can be used to assign the temporal
                         dynamics to the parameters in the monofit function
        '''

        param_dyn_dict = {}

        # initialize all parameters as scalar parameters
        for key in param_keys:
            param_dyn_dict[key] = np.ones(len(datetime_index))

        # TODO works only for unambiguous datetime-indexes !!!
        # (repeated indexes will be grouped together)
        if freq is not None:
            for i, f in enumerate(freq):
                for key in freqkeys[i]:
                    df = pd.DataFrame(np.arange(1,
                                                len(datetime_index) + 1),
                                                index=datetime_index)
                    dyn_list = []
                    for k, arr in enumerate(df.resample(f).apply(len).values):
                        dyn_list += list(np.full_like(range(arr[0]), k + 1))

                    param_dyn_dict[key] = dyn_list

        return param_dyn_dict


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


    def _calc_model(self, R, res_dict, return_components=False):
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
        return_components : bool (default=False)
                            indicator if the individual components or only
                            the total backscattered radiation are returned
                            (useful for quick evaluation of a model)
        Returns:
        ----------
        model_calc : the output of R.calc() (as intensity or sigma_0)
                     in linear-units or dB corresponding to the specifications
                     defined in the rtfits-class.
        '''
        # store original V and SRF
        orig_V = copy.deepcopy(R.V)
        orig_SRF = copy.deepcopy(R.SRF)

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        # and generate a function to evaluate the symbolic representation
        try:
            tausymb = R.V.tau[0].free_symbols
            taufunc = sp.lambdify(tausymb, R.V.tau[0],
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.V.omega[0].free_symbols
            omegafunc = sp.lambdify(omegasymb, R.V.omega[0],
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
            if 'omega' in res_dict:
                R.V.omega = res_dict['omega']
        else:
            R.V.omega = omegafunc(*[res_dict[str(i)] for i in omegasymb])

        if taufunc is None:
            if 'tau' in res_dict:
                R.V.tau = res_dict['tau']
        else:
            R.V.tau = taufunc(*[res_dict[str(i)] for i in tausymb])

        if Nfunc is None:
            if 'NormBRDF' in res_dict:
                R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = Nfunc(*[res_dict[str(i)] for i in Nsymb])

        if 'bsf' in res_dict:
            R.bsf = res_dict['bsf']

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input. (i.e. "omega", "tau", "NormBRDF"
        # and the symbols used to define them must be removed)

        # symbols used to define the functions
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, R.V._func.free_symbols)) - angset
        srfsymb = set(map(str, R.SRF._func.free_symbols)) - angset

        param_fn = res_dict.copy()
        param_fn.pop('omega', None)
        param_fn.pop('tau', None)
        param_fn.pop('NormBRDF', None)
        param_fn.pop('bsf', None)
        # vsymb and srfsymb must be subtracted in case the same symbol is used
        # for omega, tau or NormBRDF definition and in the function definiton
        for i in set(toNlist - vsymb - srfsymb):
            param_fn.pop(str(i), None)

        # ensure that the keys of the dict are strings and not sympy-symbols
        strparam_fn = dict([[str(key),
                             np.expand_dims(param_fn[key], 1)]
                            for i, key in enumerate(param_fn.keys())])

        # set the param-dict to the newly generated dict
        R.param_dict = strparam_fn

        # calculate total backscatter-values
        if return_components is True:
            model_calc = R.calc()
        else:
            model_calc = R.calc()[0]

        if self.sig0 is True:
            # convert the calculated results to sigma_0
            signorm = 4. * np.pi * np.cos(R.t_0)
            model_calc = signorm * model_calc

        if self.dB is True:
            # convert the calculated results to dB
            model_calc = 10. * np.log10(model_calc)

        # restore V and SRF to original values
        R.V = orig_V
        R.SRF = orig_SRF

        return model_calc


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
        orig_V = copy.deepcopy(R.V)
        orig_SRF = copy.deepcopy(R.SRF)

#        # set omega, tau and NormBRDF-values to input
#        if 'omega' in res_dict:
#            R.V.omega = res_dict['omega']
#        if 'tau' in res_dict:
#            R.V.tau = res_dict['tau']
#        if 'NormBRDF' in res_dict:
#            R.SRF.NormBRDF = res_dict['NormBRDF']

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        try:
            tausymb = R.V.tau[0].free_symbols
            taufunc = sp.lambdify(tausymb, R.V.tau[0],
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.V.omega[0].free_symbols
            omegafunc = sp.lambdify(omegasymb, R.V.omega[0],
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
            if 'omega' in res_dict:
                R.V.omega = res_dict['omega']
        else:
            R.V.omega = omegafunc(*[res_dict[str(i)] for i in omegasymb])

        if taufunc is None:
            if 'tau' in res_dict:
                R.V.tau = res_dict['tau']
        else:
            R.V.tau = taufunc(*[res_dict[str(i)] for i in tausymb])

        if Nfunc is None:
            if 'NormBRDF' in res_dict:
                R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = Nfunc(*[res_dict[str(i)] for i in Nsymb])

        if 'bsf' in res_dict:
            R.bsf = res_dict['bsf']

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input (i.e. "omega", "tau", "NormBRDF",
        # "bsf" and the symbols used to define them must be removed)

        # symbols used in the definitions of the functions
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, R.V._func.free_symbols)) - angset
        srfsymb = set(map(str, R.SRF._func.free_symbols)) - angset

        param_fn = res_dict.copy()
        param_fn.pop('omega', None)
        param_fn.pop('tau', None)
        param_fn.pop('NormBRDF', None)
        param_fn.pop('bsf', None)

        for i in set(toNlist - vsymb - srfsymb):
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

        # generate a scipy.sparse matrix that represents the jacobian for all
        # the individual parameters according to jac_dyn_dict
        # (this is needed to avoid memory overflows for very large jacobians)
        newjacdict = {}
        for i, key in enumerate(order):
            uniques, inds = np.unique(param_dyn_dict[key],
                                      return_index = True)
            # provide unique values based on original occurence
            # (np.unique sorts the result!!)
            uniques = uniques[np.argsort(inds)]

            if len(uniques) == 1:
                newjacdict[key] = np.array([np.concatenate(jac[i], axis=0)])
# TODO is it faster to use numpy directly for small arrays ?
#            elif len(uniques) < 50:
#                data = np.concatenate(jac[i])
#                row_ind = []  # row-indices where jac is nonzero
#                col_ind = []  # col-indices where jac is nonzero
#                for n_uni, uni in enumerate(uniques):
#                    rule = (param_dyn_dict[key] == uni)
#                    where_n = np.where(
#                            np.concatenate(
#                                    np.broadcast_arrays(
#                                            rule[:,np.newaxis], jac[0])[0]))[0]
#
#                    col_ind += list(where_n)
#                    row_ind += list(np.full_like(where_n, n_uni))
#                # generate a sparse matrix
#                m = np.full((max(row_ind) + 1,
#                                      max(col_ind) + 1), 0.)
#                # fill matrix with values
#                m[row_ind, col_ind] = data
#                newjacdict[key] = m
            else:
                # if too many unique values occur, use scipy sparse matrix
                # to avoid memory-overflow due to the large number of zeroes...
                # (this will reduce speed since scipy.sparse does not fully
                # supprot BLAS and so no proper parallelization is performed)
                data = np.concatenate(jac[i])

                row_ind = []  # row-indices where jac is nonzero
                col_ind = []  # col-indices where jac is nonzero
                for n_uni, uni in enumerate(uniques):
                    rule = (param_dyn_dict[key] == uni)
                    where_n = np.where(
                            np.concatenate(np.broadcast_to(rule[:,np.newaxis],
                                                           jac[i].shape)))[0]

                    col_ind += list(where_n)
                    row_ind += list(np.full_like(where_n, n_uni))

                # generate a sparse matrix
                m = csr_matrix((data, (row_ind, col_ind)),
                               shape=(max(row_ind) + 1,
                                      max(col_ind) + 1))
                newjacdict[key] = m

        # evaluate jacobians of the functional representations of tau
        # and add them to newjacdict
        for i in set(map(str, tausymb)) & set(param_dyn_dict.keys()):
            # generate a function that evaluates the 'inner' derivative, i.e.:
            # df/dx = df/dtau * dtau/dx = df/dtau * d_inner
            d_inner = sp.lambdify(tausymb, sp.diff(orig_V.tau[0],
                                                   sp.Symbol(i)),
                                  modules=['numpy'])
            # evaluate the inner derivative
            dtau_dx = d_inner(*[res_dict[str(i)] for i in tausymb])
            # calculate the derivative with respect to the parameters
            if np.isscalar(dtau_dx):
                newjacdict[str(i)] = newjacdict[str(i)] * dtau_dx
            elif isspmatrix(newjacdict[str(i)]):
                # In case the parameter is varying temporally, it must be
                # repeated by the number of incidence-angles in order
                # to have correct array-processing (it is assumed that no
                # parameter is incidence-angle dependent itself)
                dtau_dx = np.repeat(dtau_dx, len(np.atleast_2d(R.t_0)[0]))
                # calculate "outer" * "inner" derivative for sparse matrices
                newjacdict[str(i)] = newjacdict[str(i)].multiply(dtau_dx)
            else:
                dtau_dx = np.repeat(dtau_dx, len(np.atleast_2d(R.t_0)[0]))
                # calculate "outer" * "inner" derivative for numpy arrays
                newjacdict[str(i)] = newjacdict[str(i)] * dtau_dx

        # same for omega
        for i in set(map(str, omegasymb)) & set(param_dyn_dict.keys()):
            d_inner = sp.lambdify(omegasymb, sp.diff(orig_V.omega[0],
                                                     sp.Symbol(i)),
                                  modules=['numpy'])
            domega_dx = d_inner(*[res_dict[str(i)] for i in omegasymb])
            if np.isscalar(domega_dx):
                newjacdict[str(i)] = newjacdict[str(i)] * domega_dx
            elif isspmatrix(newjacdict[str(i)]):
                domega_dx = np.repeat(domega_dx, len(np.atleast_2d(R.t_0)[0]))
                newjacdict[str(i)] = newjacdict[str(i)].multiply(domega_dx)
            else:
                domega_dx = np.repeat(domega_dx, len(np.atleast_2d(R.t_0)[0]))
                newjacdict[str(i)] = newjacdict[str(i)] * domega_dx

        # same for NormBRDF
        for i in set(map(str, Nsymb)) & set(param_dyn_dict.keys()):
            d_inner = sp.lambdify(Nsymb, sp.diff(orig_SRF.NormBRDF[0],
                                                 sp.Symbol(i)),
                                  modules=['numpy'])
            dN_dx = d_inner(*[res_dict[str(i)] for i in Nsymb])
            if np.isscalar(dN_dx):
                newjacdict[str(i)] = newjacdict[str(i)] * dN_dx
            elif isspmatrix(newjacdict[str(i)]):
                dN_dx = np.repeat(dN_dx, len(np.atleast_2d(R.t_0)[0]))
                newjacdict[str(i)] = newjacdict[str(i)].multiply(dN_dx)
            else:
                dN_dx = np.repeat(dN_dx, len(np.atleast_2d(R.t_0)[0]))
                newjacdict[str(i)] = newjacdict[str(i)] * dN_dx

        if hasattr(self, 'intermediate_results'):
            self.intermediate_results['jacobian'] += [newjacdict]


        # return the transposed jacobian as needed by scipy's least_squares
        if np.any([isspmatrix(newjacdict[key]) for key in order]):
            # in case sparse matrices have been used, use scipy to vstack them
            jac_lsq = vstack([newjacdict[key] for key in order]).transpose()
        else:
            jac_lsq = np.vstack([newjacdict[key] for key in order]).transpose()

        # restore V and SRF to original values
        R.V = orig_V
        R.SRF = orig_SRF

        return jac_lsq


    def _calc_slope_curv(self, R, res_dict, return_components=False):
        '''
        function to calculate the monostatic slope and curvature
        of the model

        Parameters:
        ------------
        R : RT1-object
            the rt1-object for which the results shall be calculated
        res_dict : dict
                   a dictionary containing all parameter-values that should
                   be updated before calling R.calc()
        return_components : bool (default=False)
                            indicator if the individual components or only
                            the total backscattered radiation are returned
                            (useful for quick evaluation of a model)
        Returns:
        ----------
        model_calc : the output of R.calc() (as intensity or sigma_0)
                     in linear-units or dB corresponding to the specifications
                     defined in the rtfits-class.
        '''

        # store original V and SRF
        orig_V = copy.deepcopy(R.V)
        orig_SRF = copy.deepcopy(R.SRF)

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        # and generate a function to evaluate the symbolic representation
        try:
            tausymb = R.V.tau[0].free_symbols
            taufunc = sp.lambdify(tausymb, R.V.tau[0],
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.V.omega[0].free_symbols
            omegafunc = sp.lambdify(omegasymb, R.V.omega[0],
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
            if 'omega' in res_dict:
                R.V.omega = res_dict['omega']
        else:
            R.V.omega = omegafunc(*[res_dict[str(i)] for i in omegasymb])

        if taufunc is None:
            if 'tau' in res_dict:
                R.V.tau = res_dict['tau']
        else:
            R.V.tau = taufunc(*[res_dict[str(i)] for i in tausymb])

        if Nfunc is None:
            if 'NormBRDF' in res_dict:
                R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = Nfunc(*[res_dict[str(i)] for i in Nsymb])

        if 'bsf' in res_dict:
            R.bsf = res_dict['bsf']

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input. (i.e. "omega", "tau", "NormBRDF"
        # and the symbols used to define them must be removed)

        # symbols used to define the functions
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, R.V._func.free_symbols)) - angset
        srfsymb = set(map(str, R.SRF._func.free_symbols)) - angset

        param_fn = res_dict.copy()
        param_fn.pop('omega', None)
        param_fn.pop('tau', None)
        param_fn.pop('NormBRDF', None)
        param_fn.pop('bsf', None)
        # vsymb and srfsymb must be subtracted in case the same symbol is used
        # for omega, tau or NormBRDF definition and in the function definiton
        for i in set(toNlist - vsymb - srfsymb):
            param_fn.pop(str(i), None)

        # ensure that the keys of the dict are strings and not sympy-symbols
        strparam_fn = dict([[str(key),
                             np.expand_dims(param_fn[key], 1)]
                            for i, key in enumerate(param_fn.keys())])

        # set the param-dict to the newly generated dict
        R.param_dict = strparam_fn

        # calculate slope-values
        if return_components is True:
            model_slope = [R.tot_slope(sig0=self.sig0, dB=self.dB),
                          R.surface_slope(sig0=self.sig0, dB=self.dB),
                          R.volume_slope(sig0=self.sig0, dB=self.dB)]
        else:
            model_slope = R.tot_slope(sig0=self.sig0, dB=self.dB)

        # calculate curvature-values
        if return_components is True:
            model_curv = [R.tot_curv(sig0=self.sig0, dB=self.dB),
                          R.surface_curv(sig0=self.sig0, dB=self.dB),
                          R.volume_curv(sig0=self.sig0, dB=self.dB)]
        else:
            model_curv = R.tot_curv(sig0=self.sig0, dB=self.dB)


        # restore V and SRF to original values
        R.V = orig_V
        R.SRF = orig_SRF

        return {'slope' : model_slope,
                'curv' : model_curv}


    def monofit(self, V, SRF, dataset, param_dict, bsf=0.,
                bounds_dict={}, fixed_dict={}, param_dyn_dict={},
                fn_input=None, _fnevals_input=None, int_Q=True,
                lambda_backend='cse', verbosity=0,
                intermediate_results=False,
                **kwargs):
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
        fn_input : array-like
             a slot for pre-calculated fn-coefficients.
             if the same model has to be fitted to multiple datasets, the
             fn-coefficients that are returned in the first fit can be used
             as input for the second fit to avoid repeated calculations.
        _fnevals_input : callable
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
        verbosity : int
                  set verbosity level of rt1-module
        intermediate_results : bool (default = False)
                               indicator if intermediate results should be
                               stored (for analysis purpose only). If True, a
                               dictionary will be generated that contains
                               detailed results for each iteration-step of the
                               fit-procedure. It is structured as follows:

                                   'jacobian' : a list of dicts with the
                                                jacobians for each fitted
                                                parameter

                                   'errdict' : {'abserr' : list of RMSE,
                                                'relerr' : list of RMSE/data}

                                   'parameters' : list of parameter-result
                                                  dictionaries for each step

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
        # set up the dictionary for storing intermediate results
        if intermediate_results is True:
            from .rtplots import plot_interres
            self.plot_interres = partial(plot_interres, fit = self)
            update_wrapper(self.plot_interres, plot_interres)

            if not hasattr(self, 'intermediate_results'):
                self.intermediate_results = {'parameters':[],
                                             'residuals':[],
                                             'jacobian':[]}



        # generate a list of the names of the parameters that will be fitted.
        # (this is necessary to ensure correct broadcasting of values since
        # dictionarys do)
        order = [i for i, v in param_dict.items() if v is not None]
        # preparation of data for fitting
        inc, data, weights, Nmeasurements, mask = self._preparedata(dataset)

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        try:
            tausymb = V.tau[0].free_symbols
        except Exception:
            tausymb = set()
        try:
            omegasymb = V.omega[0].free_symbols
        except Exception:
            omegasymb = set()
        try:
            Nsymb = SRF.NormBRDF[0].free_symbols
        except Exception:
            Nsymb = set()

        toNlist = set(map(str, list(tausymb) + list(omegasymb) + list(Nsymb)))

        # check of general input-requirements
        # check if all parameters have been provided
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
        param_R.pop('bsf', None)

        # remove also other symbols that are used in the definitions of
        # tau, omega and NormBRDF
        for i in set(toNlist - vsymb - srfsymb):
            param_R.pop(i)

        # define rt1-object
        R = RT1(1., inc, inc, np.zeros_like(inc), np.full_like(inc, np.pi),
                V=V, SRF=SRF, fn_input=fn_input, _fnevals_input=_fnevals_input,
                geometry='mono', bsf = bsf, param_dict=param_R, int_Q=int_Q,
                lambda_backend=lambda_backend, verbosity=verbosity)
        # store _fnevals functions in case they have not been provided
        # as input-arguments explicitly to avoid recalculation for each step
        R._fnevals_input = R._fnevals
        # set fn_input to any value except None to avoid re-calculation
        R.fn_input = 1

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
            for key in order:
                # find unique parameter estimates and how often they occur
                uniques, ind = np.unique(param_dyn_dict[key],
                                         return_index=True)
                uniques = uniques[np.argsort(ind)]
                # shift index to next parameter (this is necessary since the
                # result is provided as a concatenated array)
                newdict[key] = np.full_like(param_dyn_dict[key], 999,
                                            dtype=float)
                for i, uniq in enumerate(uniques):
                    value_i = np.array(params)[count:count + len(uniques)][i]
                    where_i = np.where((param_dyn_dict[key]) == uniq)
                    newdict[key][where_i] = value_i

                # increase counter
                count = count + len(uniques)

            # incorporate values provided in fixed_dict
            # (i.e. incorporate fixed but possibly dynamic parameter-values)

            # for python > 3.4
            # newdict = dict(newdict, **fixed_dict)
            newdict = dict(list(newdict.items()) +
                           list(fixed_dict.items()))

            # calculate the residuals
            errs = np.concatenate(self._calc_model(R, newdict)) - data
            # incorporate weighting-matrix to ensure correct treatment
            # of artificially added values (see _preparedata()-function)
            errs = weights * errs


            if intermediate_results is True:
                self.intermediate_results['parameters'] += [newdict]
                errdict = {'abserr' : errs,
                           'relerr' : errs/data}
                self.intermediate_results['residuals'] += [errdict]

            return errs

        # function to evaluate the jacobian
        def dfun(params):
            # generate a dictionary to assign values based on input
            count = 0
            newdict = {}
            for key in order:
                # find unique parameter estimates and how often they occur
                uniques, ind = np.unique(param_dyn_dict[key],
                                         return_index=True)
                uniques = uniques[np.argsort(ind)]
                # shift index to next parameter (this is necessary since the
                # result is provided as a concatenated array)
                newdict[key] = np.full_like(param_dyn_dict[key], 999,
                                            dtype=float)
                for i, uniq in enumerate(uniques):
                    value_i = np.array(params)[count:count + len(uniques)][i]
                    where_i = np.where((param_dyn_dict[key]) == uniq)
                    newdict[key][where_i] = value_i

                # increase counter
                count = count + len(uniques)


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
        res_lsq = least_squares(fun, startvals, bounds=bounds,
                                jac=dfun, **kwargs)

        # generate a dictionary to assign values based on fit-results
        count = 0
        res_dict = {}
        start_dict = {}
        for key in order:
            # find unique parameter estimates and how often they occur
            uniques, ind = np.unique(param_dyn_dict[key], return_index=True)
            uniques = uniques[np.argsort(ind)]
            # shift index to next parameter (this is necessary since the result
            # is provided as a concatenated array)
            res_dict[key] = np.full_like(param_dyn_dict[key], 999,
                                            dtype=float)
            for i, uniq in enumerate(uniques):
                value_i = np.array(res_lsq.x)[count:count + len(uniques)][i]
                where_i = np.where((param_dyn_dict[key]) == uniq)
                res_dict[key][where_i] = value_i

            start_dict[key] = np.full_like(param_dyn_dict[key], 999,
                                            dtype=float)
            for i, uniq in enumerate(uniques):
                value_i = np.array(startvals)[count:count + len(uniques)][i]
                where_i = np.where((param_dyn_dict[key]) == uniq)
                start_dict[key][where_i] = value_i

            # increase counter
            count = count + len(uniques)


        # ------------------------------------------------------------------
        # ------------ prepare output-data for later convenience -----------

        # get the data in the same shape as the incidence-angles
        data = np.array(np.split(data, Nmeasurements))
        return [res_lsq, R, data, inc, mask, weights,
                res_dict, start_dict, fixed_dict]


    def performfit(self, dataset=None, defdict=None, set_V_SRF=None,
                   fn_input = None, _fnevals_input = None,
                   int_Q = False, setindex = 'mean',
                   **kwargs):
        '''
        Parameters
        -----------
        dataset : pandas.DataFrame
                  a pandas.DataFrame with a datetime-index and columns 'inc'
                  and 'sig' that correspond to the incidence-angles in radians
                  and the sigma_0 values in linear or dB units (depending on
                  the predefined dB parameter).
        defdict : dict
                  a dictionary used to assign RT1 specifications with the all
                  required keys to specify SRF and V assigned. The keys must
                  coincide with the function-arguments of set_V_SRF()!
        set_V_SRF : callable
                    a function that returns the rt1.volume- and a rt1.surface
                    objects intended to be used in the fit.

                    For example:

                    >>> def set_V_SRF(omega, tau, t, N):
                    >>>    from rt1.volume import Rayleigh
                    >>>    from rt1.surface import HenyeyGreenstein
                    >>>
                    >>>    V=Rayleigh(omega=omega, tau=tau)
                    >>>    SRF=HenyeyGreenstein(t=t, NormBRDF=N)
                    >>>
                    >>>    return V, SRF
        fn_input : list
                   a list of pre-evaluated fn-coefficients
        _fnevals_input : callable
                         a pre-compiled function for evaluation of the
                         fn-coefficients (for speedup in case V and SRF
                         properties are used in multiple fits)
        int_Q : bool (default = False)
                indicator if interaction-terms are evaluated or not
        setindex : str (default = 'mean')
                   selection of the the datetime-index for the fit-results
                   possible values are:
                       'mean' : the center date of the used timespan
                       'first' : the first date of the timespan
                       'last' : the last date of the timespan
                       'original' : return the full list of datetime-objects

        TODO... :
        kwargs_least_squares : dict
                 keyword arguments passed to scipy.optimize.least_squares()
        kwargs_monofit : dict
                 keyword arguments passed to rtfits.monofit()

        Returns:
        -----------

        return_inv : dict
                     a dictionary with the following keys assigned:
                         'fit' : the rtfits.monofit result,
                         'dataset' : the dataset used in the fit,
                         'fn_input' : resulting fn coefficients,
                         '_fnevals_input' : resulting _fnevals functions}
        '''
        assert setindex in ['mean','first',
                            'last', 'original'], 'setindex must be either' \
                            + " 'mean', 'first', 'last' or 'original'"

        # TODO change to only using rtfits objects!
        # (i.e. it is not necessary to manually specify defdict etc.)
        if dataset is None: dataset = self.dataset
        if defdict is None: defdict = self.defdict
        if set_V_SRF is None: set_V_SRF = self.set_V_SRF


        # generate RT1 specifications based on defdict
        # initialize empty dicts
        fixed_dict = {}
        setdict = {}
        startvaldict = {}
        timescaledict = {}
        boundsvaldict = {}
        manual_dyn_df = None
        # set parameters
        for key in defdict.keys():
            # if parameter is intended to be fitted, assign a sympy-symbol
            if defdict[key][0] is True:
                # TODO see why this is actually necessary
                if key != 'omega':  # omega must not be a sympy-symbol name
                    setdict[key] = sp.var(key)
                else:
                    # a dummy value that will be replaced in rtfits.monofit
                    setdict[key] = 100

                # set start-values
                startvaldict[key] = defdict[key][1]

                # set temporal variability
                if defdict[key][2] == 'manual':
                    if manual_dyn_df is None: manual_dyn_df = pd.DataFrame()
                    manual_dyn_df = pd.concat([manual_dyn_df,
                                               defdict[key][4]], axis=1)
                elif defdict[key][2] is not None:

                    timescaledict[key] = defdict[key][2]
                    try:
                        manual_dyn_df = pd.concat([manual_dyn_df,
                                                   defdict[key][4]], axis=1)
                    except Exception:
                        pass

                # set boundaries
                boundsvaldict[key] = defdict[key][3]

            elif defdict[key][0] is False:
                # treat parameters that are intended to be constants
                # if value is provided as a scalar, insert it in the definition
                if np.isscalar(defdict[key][1]):
                    setdict[key] = defdict[key][1]
                else:
                    # if value is provided as array, add it to fixed_dict
                    # TODO same as above ...why is this necessary?
                    # TODO what about 'tau' and 'NormBRDF'
                    if key != 'omega':  # omega must not be a sympy-symbol name
                        setdict[key] = sp.var(key)
                    else:
                        # a dummy value that will be replaced by rtfits.monofit
                        setdict[key] = 100
                    fixed_dict[key] = defdict[key][1]

        # set V and SRF based on setter-function
        V, SRF = set_V_SRF(**setdict)

        # set frequencies of fitted parameters
        freq = []
        freqkeys = []
        for key in timescaledict:
            freq += [timescaledict[key]]
            freqkeys += [[key]]


        def generatedataset(dataset, dyn_keys,
                            freq=None, freqkeys=[],
                            manual_dyn_df=None):
            '''
            a function to group the dataset to arrays based
            on the provided frequency-keys
            '''
            dataset = copy.deepcopy(dataset)
            if manual_dyn_df is not None:
                manual_dyn_df = copy.deepcopy(manual_dyn_df)
                # in case multiple measurements have been made on the same day
                #manual_dyn_df = manual_dyn_df.loc[dataset.index]

            param_dyn_dict = {}
            # initialize all parameters as scalar parameters
            for key in dyn_keys:
                param_dyn_dict[key] = np.ones(len(dataset.index), dtype=int)

            if freq is not None:
                for i, f in enumerate(freq):
                    for key in freqkeys[i]:

                        df = pd.concat([
                                    pd.DataFrame({key:[nval]}, val.index)
                                    for nval, [_, val] in enumerate(
                                            dataset.groupby(
                                                    pd.Grouper(freq=f)))])

                        param_dyn_dict[key] = np.array(df[key].values,
                                                       dtype=int).flatten()

            if manual_dyn_df is not None:
                for key, val in manual_dyn_df.items():
                    # param_dyn_dict values as strings (zfilled to N digits)
                    dd1 = np.char.zfill(np.array(param_dyn_dict[key],
                                                dtype='str'),
                                  len(max(np.array(param_dyn_dict[key],
                                                   dtype='str'), key=len)))
                    # manual_dyn_df values as strings (zfilled to N digits)
                    dd2 = np.char.zfill(np.array(val, dtype='str'),
                                  len(max(np.array(val, dtype='str'),
                                          key=len)))
                    # generate a combined (unique) integer
                    param_dyn_dict[key] = np.array(np.char.add(dd1, dd2),
                                                   dtype=int)

            param_dyn_df = pd.DataFrame(param_dyn_dict, index=dataset.index)
            # get name of parameter with the maximum amount of unique values

            # get final group-indexes
            groupindex = None
            for key, val in param_dyn_df.items():
                dd = np.char.zfill(np.array(val, dtype='str'),
                                   len(max(np.array(val, dtype='str'),
                                           key=len)))
                if groupindex is None:
                    groupindex = dd
                else:
                    groupindex = np.char.add(groupindex, dd)

            dataset['orig_index'] = dataset.index
            dataset = dataset.set_index(groupindex)
            groupdf = dataset.groupby(level=0)

            # generate new data-frame based on groups
            new_df_cols = []
            for key in dataset.keys():
                new_df_cols += [groupdf[key].apply(list).apply(np.array)]
            new_df = pd.concat(new_df_cols, axis=1)


            param_dyn_df['orig_index'] = param_dyn_df.index
            param_dyn_df = param_dyn_df.set_index(groupindex)
            param_dyn_groupdf = param_dyn_df.groupby(level=0)

            index = param_dyn_groupdf['orig_index'].apply(list).apply(np.array)
            vals = [param_dyn_groupdf[key].apply(list).apply(np.take, indices=0)
                    for key in param_dyn_df]
            param_dyn_df = pd.concat([index, *vals], axis=1)

            for key, val in param_dyn_df.items():
                param_dyn_dict[key] = list(val.values)

            #self.param_dyn_dict = param_dyn_dict
            return new_df, param_dyn_dict

        dataset_used, param_dyn_dict = generatedataset(
                dataset=dataset, dyn_keys=startvaldict.keys(),
                freq=freq, freqkeys=freqkeys, manual_dyn_df=manual_dyn_df)


        # re-shape param_dict and bounds_dict to fit needs
        param_dict = {}
        for key in startvaldict:
            uniqueparams = len(pd.unique(param_dyn_dict[key]))
            # adjust shape of startvalues
            if uniqueparams >= 1 and np.isscalar(startvaldict[key]):
                param_dict[key] = [startvaldict[key]]*uniqueparams
            else:
                param_dict[key] = startvaldict[key]

        bounds_dict = {}
        for key in boundsvaldict:
            uniqueparams = len(pd.unique(param_dyn_dict[key]))
            # adjust shape of boundary conditions
            if uniqueparams >= 1 and len(boundsvaldict[key][0]) == 1:
                bounds_dict[key] = (boundsvaldict[key][0]*uniqueparams,
                                    boundsvaldict[key][1]*uniqueparams)
            else:
                bounds_dict[key] = (boundsvaldict[key])

        # perform fit
        fitresult = self.monofit(V=V, SRF=SRF,
                                 dataset=dataset_used[['inc', 'sig']].values,
                                 param_dict=param_dict,
                                 bsf = setdict['bsf'],
                                 bounds_dict=bounds_dict,
                                 fixed_dict=fixed_dict,
                                 param_dyn_dict=param_dyn_dict,
                                 fn_input=fn_input,
                                 _fnevals_input=_fnevals_input,
                                 int_Q=int_Q,
                                 #lambda_backend = 'cse_symengine_sympy',
                                 lambda_backend = 'sympy',
                                 verbosity=2,
                                 **kwargs)

        # generate a datetime-index from the given groups
        if setindex is 'first':
            self.index = pd.to_datetime(
                    dataset_used.orig_index.apply(np.take,indices=0).values)
        elif setindex is 'last':
            self.index = pd.to_datetime(
                    dataset_used.orig_index.apply(np.take,indices=-1).values)
        elif setindex is 'mean':
            self.index = pd.to_datetime(
                    dataset_used.orig_index.apply(meandatetime).values)
        elif setindex is 'original':
            self.index = dataset_used.index

        self.result = fitresult
        self.dataset_used = dataset_used


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

        ax2.set_ylim(0., np.max(list(res_dict.values())))

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

    def printerr(self, fit, datelist=None, newcalc=False, relative=False):
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
        relative : bool (default = False)
                   indicator if relative (True) or absolute (False) residuals
                   shall be plotted
        '''

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        Nmeasurements = len(inc)

        if newcalc is False:
            # get residuals from fit into desired shape for plotting
            # Attention -> incorporate weights and mask !
            res = np.ma.masked_array(np.reshape(res_lsq.fun, data.shape), mask)

            if relative is True:
                res = np.ma.abs(res / (res + np.ma.masked_array(data, mask)))
            else:
                res = np.ma.abs(res)
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

            if relative is True:
                res = res / masked_estimates

        # apply mask to data and incidence-angles (and convert to degree)
        inc = np.ma.masked_array(np.rad2deg(inc), mask=mask)
        data = np.ma.masked_array(data, mask=mask)

        # make new figure
        figres = plt.figure(figsize=(14, 10))
        axres = figres.add_subplot(212)
        if relative is True:
            axres.set_title('Mean relative residual per measurement')
        else:
            axres.set_title('Mean absolute residual per measurement')

        axres2 = figres.add_subplot(211)
        if relative is True:
            axres2.set_title('Relative residuals per incidence-angle')
        else:
            axres2.set_title('Residuals per incidence-angle')

        # the use of masked arrays might cause python 2 compatibility issues!
        axres.plot(np.arange(len(res)) + 1, res, '.', alpha=0.5)

        # plot mean residual for each measurement
        axres.plot(np.arange(1, Nmeasurements + 1), np.ma.mean(res, axis=1),
                   'k', linewidth=3, marker='o', fillstyle='none')

        # plot total mean of mean residuals per measurement
        axres.plot(np.arange(1, Nmeasurements + 1),
                   [np.ma.mean(np.ma.mean(res, axis=1))] * Nmeasurements,
                   'k--')

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
            axres.set_xticklabels(np.arange(1, Nmeasurements + 1))
            axres.set_xlabel('# Measurement')

            majortickinterval = np.ceil(Nmeasurements/50)
            axres.set_xlim(-majortickinterval//2,
                           Nmeasurements + majortickinterval//2)

            majorLocator = MultipleLocator(majortickinterval)
            majorFormatter = FormatStrFormatter('%d')
            minorLocator = MultipleLocator(2)

            axres.xaxis.set_major_locator(majorLocator)
            axres.xaxis.set_major_formatter(majorFormatter)
            axres.xaxis.set_minor_locator(minorLocator)

            if majortickinterval > 1:
                axres.xaxis.set_tick_params(rotation=45)

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

        # find minimum and maximum incidence angle
        maxinc = np.max(inc)
        mininc = np.min(inc)

        axres2.set_xlim(np.floor(mininc) - 1,
                        np.ceil(maxinc) + 1)

        # set major and minor ticks
        axres2.xaxis.set_major_locator(MultipleLocator(1))
        axres2.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axres2.xaxis.set_minor_locator(MultipleLocator(.1))

        figres.tight_layout()

        return figres

    def printscatter(self, fit, mima=None, pointsize=0.5,
                     regression=True, newcalc=False,  **kwargs):
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

        if newcalc is True:

            # for python > 3.4
            # calc_dict = dict(**res_dict, **fixed_dict)
            calc_dict = dict((k, v) for k, v in list(res_dict.items())
                             + list(fixed_dict.items()))

            estimates = self._calc_model(R, calc_dict)

            # apply mask
            estimates = estimates[~mask]
            measures = data[~mask]

        else:
            # get the residuals and apply mask
            residuals = np.reshape(res_lsq.fun, data.shape)
            residuals = np.ma.masked_array(residuals, mask)
            # prepare measurements
            measures = data[~mask]
            # calculate estimates
            estimates = residuals[~mask] + measures

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
                    dates=None, hexbinQ=True, hexbinargs={},
                    convertTodB=False):
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
        convertTodB : bool (default=False)
                      if set to true, the datasets will be converted to dB
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

            if convertTodB is True:
                y = 10.*np.log10(estimates[m][~mask[m]])
            else:
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

            if convertTodB is True:
                ydata = 10.*np.log10(data[m][~mask[m]])
            else:
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

        return fig

    def printseries(self, fit, index=None, legends=True, minmax=None,
                    newcalc=False, convertTodB=False):
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

        (res_lsq, R, data, inc, mask, weights,
         res_dict, start_dict, fixed_dict) = fit

        if minmax is None:
            minmax = [0, len(data)]

        if newcalc is True:

            # for python > 3.4
            # calc_dict = dict(**res_dict, **fixed_dict)
            calc_dict = dict((k, v) for k, v in list(res_dict.items())
                             + list(fixed_dict.items()))

            estimates = self._calc_model(R, calc_dict)
        else:
            # get the residuals and apply mask
            residuals = np.reshape(res_lsq.fun, data.shape)
            residuals = np.ma.masked_array(residuals, mask)
            # prepare measurements
            measures = np.ma.masked_array(data, mask)
            # calculate estimates
            estimates = residuals + measures

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

        if convertTodB is True:
            maskedestimates = 10.*np.ma.log10(maskedestimates)
            maskeddata = 10.*np.ma.log10(maskeddata)

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

    def printviolin(self, inputs, param, together=True,
                    func=np.mean, print_mean=False,
                    bw_method=None, funcargs={}, **kwargs):
        '''
        plot a violin-plot of defined fit-result parameters

        Parameters:
        ------------
        inputs : dict
                 a dictionary containing the plotlabels(keys) and a list of
                 results of the monofit-function (values) that are intended
                 to be compared, e.g.:
                     inputs = {'name1' : [monofit1, monofit2, ...],
                               'name2' : [monofit3, monofit4, ...]'}
        param : str
                the name of the parameter to be plotted (the corresponding key
                of the results dictionary returned by the monofit function)
        together : bool
                   indicator whether or not a combined violin shall be plotted
        func : function-call signature (default = np.mean)
               the call-signature of a function that will be applied to the
               list of fitted-values for the chosen parameter
        print_mean : bool
                     indicator whether or not dotted lines for the mean-values
                     shall be plotted
        bw_method : str, int or None
                    selector for the chosen method for estimating the bandwidth
                    in the generation of the violinplot
                    (parameter of matplotlib.pyplot.violin)
        **kwargs : kwargs
                   keyword-arguments forwraded to matplotlib.pyplot.figure()

        Returns:
        ---------
        fig : matplotlib.figure
        '''

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)
        resval = {}

        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for reskey in inputs:
            inp = inputs[reskey]
            # evaluate sortpattern to sort results based on coverage
            # p, c = sfc.printfracs(inp)
            # plt.close(p)
            # sortpa[reskey] = np.argsort(np.sum(c, axis=0))

            val = []

            if param == 'residual':
                # estimate and print rel. residuals of the test_i measurement
                rel_residual = []
                for test_i in np.array(inp):
                    # get results of i'th measurement of fit-result inp
                    (res_lsq, _, data, _, mask, weights,
                     _, _, _) = test_i[0]

                    # get the residuals and apply mask
                    residuals = np.reshape(res_lsq.fun, data.shape)
                    residuals = np.ma.masked_array(residuals, mask)
                    # prepare measurements
                    measures = np.ma.masked_array(data, mask)
                    # calculate estimates
                    estimates = residuals + measures
                    # calculate relative absolute residuals
                    rel_residual = np.ma.abs(residuals/estimates)
                    # calculate mean relative residual over incidence-angles
                    mean_rel_residual = np.ma.mean(rel_residual, axis=1)
                    # convert result to ordinary list
                    mean_rel_residual = list(mean_rel_residual)

                    # apply function
                    addval = func(mean_rel_residual, **funcargs)
                    if np.isscalar(addval):
                        addval = [addval]
                    else:
                        addval = list(addval)
                    val += addval
            else:
                # extract res_dict from given result
                params = [i[6] for i in np.array(inp)[:, 0]]

                for i in params:
                    addval = func(i[param], **funcargs)
                    if np.isscalar(addval):
                        addval = [addval]
                    else:
                        addval = list(addval)
                    val += addval

            resval[reskey] = np.array(val)

        # print each result as an individual violin plot
        vio_ind = ax.violinplot(resval.values(),
                                showmeans=True,
                                showextrema=True,
                                bw_method=bw_method)
        # now change colors
        i = 0
        for pc in vio_ind['bodies']:
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_edgecolor(colors[i % len(colors)])
            pc.set_alpha(.8)
            i = i + 1

        for i in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            vio_ind[i].set_color('k')
            vio_ind[i].set_alpha(0.5)

        # set labels
        labels = [i for i in resval.keys()]
        nticks = len(vio_ind['bodies']) + 1

        if together is True:
            # print the violinplots together
            # (with only the outer line being plotted)
            i = 0
            for key in resval:
                vio = ax.violinplot(resval[key],
                                    positions=[len(vio_ind['bodies']) + 1],
                                    showmeans=False,
                                    showextrema=False,
                                    bw_method=bw_method)
                # now change colors
                for pc in vio['bodies']:
                    pc.set_facecolor('None')
                    pc.set_edgecolor(colors[i % len(colors)])
                    pc.set_alpha(1.)
                    pc.set_linewidth(1.)
                i = i + 1

            # print a shading for the violinplots
            i = 0
            for key in resval:
                vio = ax.violinplot(resval[key],
                                    positions=[len(vio_ind['bodies']) + 1],
                                    showmeans=False,
                                    showextrema=False,
                                    bw_method=bw_method)
                # now change colors
                for pc in vio['bodies']:
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_edgecolor('None')
                    pc.set_alpha(.2)
                i = i + 1

            labels += ['all']
            nticks += 1

        ax.set_xticks(np.arange(1, nticks))
        ax.set_xticklabels(labels)

        if print_mean is True:
            i = 0
            for key in resval:
                ax.plot([0.5, nticks - 0.5],
                        [np.mean(resval[key])] * 2,
                        linestyle='--',
                        color=colors[i % len(colors)],
                        alpha=0.5,
                        linewidth=0.5)
                i += 1

        if param == 'residual':
            ax.set_title('Parameter: relative residual')
            ax.set_ylabel('relative residual')
        else:
            ax.set_title('Parameter: ' + str(param))
            ax.set_ylabel(str(param))

        return fig
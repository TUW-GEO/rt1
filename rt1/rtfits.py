"""
Class to perform least_squares fitting of RT-1 models to given datasets.
"""

import numpy as np
import sympy as sp
from sympy.abc import _clash
import pandas as pd

from scipy.optimize import least_squares
from scipy.sparse import vstack
from scipy.sparse import csr_matrix, isspmatrix

from .scatter import Scatter
from .rt1 import RT1, _init_lambda_backend
from .general_functions import meandatetime, rectangularize
from .rtplots import plot as rt1_plots

from . import surface as rt1_s
from . import volume as rt1_v

import copy
import multiprocessing as mp
from itertools import chain, repeat
from functools import partial

try:
    import cloudpickle
except ModuleNotFoundError:
    print('cloudpickle could not be imported, .dump() will not work!')

from configparser import ConfigParser

class Fits(Scatter):
    '''
    Class to perform nonlinear least-squares fits to data.


    Attributes:
    ------------
    sig0: boolean (default = False)
           Indicator whether dataset is given as sigma_0-values (sig_0) or as
           intensity-values (I). The applied relation is:
               sig_0 = 4. * np.pi * np.cos(inc) * I
           where inc is the corresponding incident zenith-angle.
    dB: boolean (default = False)
         Indicator whether dataset is given in linear units or in dB.
         The applied relation is:    x_dB = 10. * np.log10( x_linear )
    dataset: pandas.DataFrame (default = None)
             a pandas.DataFrame with columns 'inc' and 'sig' defined
             where 'inc' referrs to the incidence-angle in radians, and
             'sig' referrs to the measurement value (corresponding to
             the assigned sig0 and dB values)

             If a column 'data_weights' is provided, the residuals in the
             fit-procedure will be weighted accordingly.
             (e.g. residuals = weights * calculated_residuals )

    defdict: dict (default = None)
             a dictionary of the following structure:
             (the dict will be copied internally using copy.deepcopy(dict))

             >>> defdict = {'key1' : [fitQ, val, freq, ([min], [max]), dyndf],
             >>>            'key2' : [fitQ, val, freq, ([min], [max]), dyndf],
             >>>            ...}

             where all keys required to call set_V_SRF must be defined
             and the values are defined via:
                 fitQ: bool
                       indicator if the quantity should be fitted (True)
                       or used as a constant during the fit (False)
                 val: float or pandas.DataFrame
                       - if fitQ is True, val will be used as start-value
                       - if fitQ is False, val will be used as constant.
                       Notice: if val is a DataFrame, the index must coinicide
                       with the index of the dataset, and the column-name
                       must be the corresponding variabile-name
                 freq: str or None (only needed if fitQ is True)
                        - if None, a constant value will be fitted
                        - if 'manual', the DataFrame provided as dyndf will
                          be used to assign the temporal variability within
                          the fit
                        - if freq corresponds to a pandas offset-alias, it
                          will be used together with the dataset-index to
                          assign the temporal variability within the fit
                          (see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)
                 min, max: float (only needed if fitQ is True)
                            the boundary-values used within the fit
                 dyndf: pandas.DataFrame (optional)
                         - if freq has been set to 'manual', the provided
                           DataFrame will be used to assign the temporal
                           variability within the fit (a single value will be
                           fitted to all measurements where dyndf has the same
                           value).
                         - if freq is set to 'index', a unique value will be
                           fitted to each unique index of the provided
                           dataset
                         - if freq is a pandas offset-alias and a dyndf is
                           provided, the variability of dyndf will be
                           superimposed onto the variability resulting form
                           the chosen offset-alias
                         Notice: The index must coinicide
                         with the index of the dataset, and the column-name
                         must be the corresponding variabile-name

    set_V_SRF: callable (default = None)
               function with the following structure:

               >>> def set_V_SRF(volume-keys, surface-keys):
               >>>     from rt1.volume import 'Volume-function'
               >>>     from rt1.surface import 'Surface function'
               >>>
               >>>     V = Volume-function(volume-keys)
               >>>     SRF = Surface-function(surface-keys)
               >>>
               >>>     return V, SRF

    fitset: dict (default = dict())
            a dictionary with keyword-arguments passed to monofit() and further
            to scipy.optimize.least_squares

    setindex: str (default = 'mean')
              indicator how the datetime-indices of the fit-results
              should be processed. possible values are:
                  - 'mean': the center date of the used timespan
                  - 'first': the first date of the timespan
                  - 'last': the last date of the timespan
                  - 'original': return the full list of datetime-objects

    Stored Fit-Attributes:
    -------------------

    index: array-like
           the resulting index of the obtained fit-results, to be used via:
                   results = pd.DataFrame(res_dict, index)
    res_dict: dict
              a dictionary of the obtained fit-results

    fit_output: scipy.optimize.OptimizeResult
                 the output of scipy.optimize.least_squares
    R: rt1.RT1 object
       the RT1 object used
    data: array-like
          the used measurement malues
    inc: array-like
         the used incidence-angles
    mask: array-like
          a mask that indicates the values added to "data" and "inc" in order
          to obtain a rectangular array
    weights: array-like
             a weighting-matrix with values 1/sqrt(value-repetitions) where the
             value-repetitions correspond to the number of added values needed
             to obtain a rectangular array
    start_dict: dict
                a dictionary of the used start-values
    fixed_dict: dict
                a dictionary of the used auxiliary datasets
    dataset_used: pandas.DataFrame
                  a DataFrame of the used data grouped with respect to the
                  temporal variations of the parameters that have been fitted

    Methods
    ---------

    performfit(fn_input=None, _fnevals_input=None, int_Q=False,
               setindex='mean', **kwargs)
        perform a fit of the defined model to the dataset

    '''

    def __init__(self, sig0=False, dB=False, dataset=None,
                 defdict=None, set_V_SRF=None, fitset=dict(),
                 setindex = 'mean', **kwargs):
        self.sig0 = sig0
        self.dB = dB
        self.dataset = dataset
        self.set_V_SRF = set_V_SRF
        self.defdict = copy.deepcopy(defdict)
        self.fitset = fitset
        self.setindex = setindex
        # add plotfunctions
        self.plot = rt1_plots(self)



    def __update__(self):
        '''
        needed for downward compatibility
        '''
        if (not hasattr(self, 'R') and
            hasattr(self, 'result') and
            self.result is not None and len(self.result) == 9):
            print('... updating attributes')
            self.fit_output = self.result[0]
            self.R = self.result[1]
            self.data = self.result[2]
            self.inc = self.result[3]
            self.mask = self.result[4]
            self.weights = self.result[5]
            self.res_dict = self.result[6]
            self.start_dict = self.result[7]
            self.fixed_dict = self.result[8]

        if not hasattr(self, 'setindex'):
            self.setindex = 'mean'

        if not hasattr(self, 'plot'):
            print('... re-initializing plot-functions')
            self.plot = rt1_plots(self)


    def __setstate__(self, d):
        # this is done to support downward-compatibility with pickled results
        self.__dict__ = d
        self.__update__()


    def __getstate__(self):
        if '_rt1_dump_mini' in self.__dict__:
            # remove unnecessary data to save storage
            removekeys = ['R', 'fit_output', 'fixed_dict', 'start_dict',
                          'dataset_used']

            delattr(self, '_rt1_dump_mini')

            # remove pre-evaluated fn-evals functions
            if (getattr(self, 'fitset', None) is not None
                and '_fnevals_input' in self.fitset):
                self.fitset['_fnevals_input'] = None

            return {key: val for key, val in self.__dict__.items()
                    if key not in removekeys}

        else:
            return self.__dict__


    def _generatedataset(self, dataset, dyn_keys,
                         freq=None, freqkeys=[],
                         manual_dyn_df=None,
                         fixed_dict=dict()):
        '''
        a function to group the dataset to arrays based
        on the provided frequency-keys

        Parameters:
        -----------
        dataset: pandas.DataFrame
                 A pandas-DataFrame with columns inc and sig that
                 correspond to the incidence-angle- and backscatter
                 values
        dyn_keys: list of strings
                  a list of the names of the parameters that are intended
                  ot be fitted
        freq: list
              a list of frequencies that will be assigned to the
              parameters. For more details check the pandas "DateOffset"
              https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

        freqkeys: list
                  a list of parameter-names to which the corresponding
                  frequency will be assigned.

                  e.g. if freq is ['D', 'M'] then freqkeys might look
                  like [['dayparam1', 'dayparam2'], ['monthparam1']]

        manual_dyn_df: pandas.DataFrame
                       a pandas-DataFrame with column-names corresponding
                       to the keys whose temporal grouping will be
                       assigned based on the values.

        fixed_dict: pandas.DataFrame
                    a pandas-DataFrame with timeseries of parameters
                    that are intended to be used as auxiliary datasets
                    -> the index must match with the index of dataset!
        '''

        dataset = pd.concat([dataset] +
                            [val for key, val in fixed_dict.items() \
                             if isinstance(val, (pd.Series, pd.DataFrame))],
                            axis=1)

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


        if groupindex is None:
            groupindex = dataset.index

        dataset['orig_index'] = dataset.index
        dataset = dataset.set_index(groupindex)
        groupdf = dataset.groupby(level=0)

        # generate new data-frame based on groups
        new_df = groupdf.agg(pd.Series.tolist)

        param_dyn_df['orig_index'] = param_dyn_df.index
        param_dyn_df = param_dyn_df.set_index(groupindex)
        param_dyn_groupdf = param_dyn_df.groupby(level=0)

        index = param_dyn_groupdf['orig_index'].apply(np.array)
        vals = [param_dyn_groupdf[key].apply(list).apply(np.take, indices=0)
                for key in param_dyn_df]
        param_dyn_df = pd.concat([index, *vals], axis=1)

        for key, val in param_dyn_df.items():
            param_dyn_dict[key] = list(val.values)

        return new_df, param_dyn_dict


    def _setindex(self, setindex):
        '''
        set the index-property of the Fits-object
        (dataset_used must be present, e.g. performfit() must have been run
         prior to calling this method!)

        Parameters:
        ---------------
        setindex: str
                  identifyer how to assign the index of the fitted values
                  in case timespans longer than 1 measurement are used.

                  possible values are:

                  - 'first': the first index-value will be used
                  - 'last': the last index-value will be used
                  - 'mean': the center index-value will be used
                    (Note: this results in a index that might not have been
                    present in the input dataset!)
                  - 'original': the list of indices will be returned without
                    alteration (Note: without further processing, this index
                    can not be directly used to generate a pandas DataFrame)
        '''
        # generate a datetime-index from the given groups
        try:
            if setindex == 'first':
                self.index = pd.to_datetime(
                        self.dataset_used.orig_index.apply(np.take,
                                                      indices=0).values)
            elif setindex == 'last':
                self.index = pd.to_datetime(
                        self.dataset_used.orig_index.apply(np.take,
                                                      indices=-1).values)
            elif setindex == 'mean':
                self.index = pd.to_datetime(
                        self.dataset_used.orig_index.apply(meandatetime).values)
            elif setindex == 'original':
                self.index = self.dataset_used.index
        except:
            print('index could not be combined... use original index instead')
            self.index = self.dataset_used.index


    def _preparedata(self, dataset):
        '''
        prepare data such that it is applicable to least_squres fitting
        - separate incidence-angles and data-values
        - rectangularize the data-array
          (this is necessary in order to allow array-processing)
        - provide weighting-matrix to correct for rectangularization

        Parameters:
        ------------
        dataset: pandas.DataFrame or list
                 input-dataset as pandas-DataFrame with columns
                 ['inc', 'sig'] and any number of additional columns that
                 represent auxiliary datasets

                 input-dataset as list of the shape:
                     [[inc_0, data_0], [inc_1, data_2], ...]


        Returns:
        ---------
        inc: array-like
             a rectangular array consisting of the incidence-angles as
             provided in the dataset, rectangularized by repeating the last
             values of each row to fit in length.
        data: array-like
              a rectangular array consisting of the data-values as provided
              in the dataset, rectangularized by repeating the last values
              of each row to fit in length
        weights: array-like
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

                 If a column 'data_weights' has been provided in the dataset,
                 the obtained weights will additionally be multiplied by the
                 values provided as 'data_weights'.
        N: int
           number of measurements that have been provided within the dataset

        mask: array-like
              a mask that indicates the artificially added values

        new_fixed_dict: dict
                        a dictionary with the values of the auxiliary-datasets
                        grouped such that they can be used within the
                        fitting procedure
        '''


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

        # this is here for backward-support
        if isinstance(dataset, list):
            N = len(dataset)
            inc, weights, mask = rectangularize([i[0] for i in dataset],
                                                weights_and_mask = True)
            data = rectangularize([i[1] for i in dataset])
            new_fixed_dict = dict()

        elif isinstance(dataset, pd.DataFrame):
            # save number of datasets
            N = len(dataset)
            inc, weights, mask = rectangularize(dataset.inc.values,
                                                weights_and_mask = True)
            data = rectangularize(dataset.sig.values)

            new_fixed_dict = dict()
            for key in dataset:
                if key not in ['inc', 'sig', 'orig_index']:
                    new_fixed_dict[key] = rectangularize(dataset[key])

        if 'data_weights' in new_fixed_dict:
            print('applying data_weights')
            weights = weights * new_fixed_dict.pop('data_weights')

        return [inc, np.concatenate(data), np.concatenate(weights),
                N, mask, new_fixed_dict]


    def _calc_model(self, R=None, res_dict=None, fixed_dict=None,
                    return_components=False,
                    **kwargs):
        '''
        function to calculate the model-results (intensity or sigma_0) based
        on the provided parameters in linear-units or dB

        Parameters:
        ------------
        R: RT1-object
           the rt1-object for which the results shall be calculated
        res_dict: dict
                  a dictionary containing all parameter-values that should
                  be updated before calling R.calc()
        return_components: bool (default=False)
                           indicator if the individual components or only
                           the total backscattered radiation are returned
                           (useful for quick evaluation of a model)
        Returns:
        ----------
        model_calc: the output of R.calc() (as intensity or sigma_0)
                    in linear-units or dB corresponding to the specifications
                    defined in the rtfits-class.
        '''

        if R is None:
            try:
                R = self.R
            except AttributeError:
                assert False, 'R is not available and must be provided'
        if res_dict is None:
            try:
                res_dict = self.res_dict
            except AttributeError:
                assert False, 'res_dict is not available and must be provided'
        if fixed_dict is None:
            try:
                fixed_dict = self.fixed_dict
            except AttributeError:
                assert False, 'fixed_dict is not available and must be provided'

        # ensure correct array-processing
        res_dict = {key:np.repeat(*val)[:,np.newaxis] for
                    key, val in res_dict.items()}
        res_dict.update(fixed_dict)

        # store original V and SRF
        orig_V = copy.deepcopy(R.V)
        orig_SRF = copy.deepcopy(R.SRF)
        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        # and generate a function to evaluate the symbolic representation
        try:
            tausymb = R.V.tau.free_symbols
            taufunc = sp.lambdify(tausymb, R.V.tau,
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None

        try:
            omegasymb = R.V.omega.free_symbols
            omegafunc = sp.lambdify(omegasymb, R.V.omega,
                                    modules=['numpy'])
        except Exception:
            omegasymb = set()
            omegafunc = None

        try:
            Nsymb = R.SRF.NormBRDF.free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF,
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

        # exclude all keys that are not needed to calculate the fn-coefficients
        # vsymb and srfsymb must be subtracted in case the same symbol is used
        # for omega, tau or NormBRDF definition and in the function definiton
        excludekeys = ['omega', 'tau', 'NormBRDF', 'bsf',
                       *[str(i) for i in set(toNlist - vsymb - srfsymb)]]

        strparam_fn = {str(key) : val for key, val in res_dict.items()
                       if key not in excludekeys}

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


    def _calc_jac(self, R, res_dict, fixed_dict, param_dyn_dict, order):
        '''
        function to evaluate the jacobian in the shape as required
        by scipy's least_squares function

        Parameters:
        ------------
        R: RT1-object
           the rt1-object for which the results shall be calculated
        res_dict: dict
                  a dictionary containing all parameter-values that should
                  be updated before calling R.jac()
        Returns:
        --------
        jac: array_like(float)
             the jacobian corresponding to the fit-parameters in the
             shape applicable to scipy's least_squres-function
        '''
        # ensure correct array-processing
        res_dict = {key:np.repeat(*val)[:,np.newaxis] for
                    key, val in res_dict.items()}
        res_dict.update(fixed_dict)

        # store original V and SRF
        orig_V = copy.deepcopy(R.V)
        orig_SRF = copy.deepcopy(R.SRF)

        # set omega, tau and NormBRDF-values to input
        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        try:
            tausymb = R.V.tau.free_symbols
            taufunc = sp.lambdify(tausymb, R.V.tau,
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.V.omega.free_symbols
            omegafunc = sp.lambdify(omegasymb, R.V.omega,
                                    modules=['numpy'])
        except Exception:
            omegasymb = set()
            omegafunc = None
        try:
            Nsymb = R.SRF.NormBRDF.free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF,
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

        # exclude all keys that are not needed to calculate the fn-coefficients
        # vsymb and srfsymb must be subtracted in case the same symbol is used
        # for omega, tau or NormBRDF definition and in the function definiton
        excludekeys = ['omega', 'tau', 'NormBRDF', 'bsf',
                       *[str(i) for i in set(toNlist - vsymb - srfsymb)]]

        strparam_fn = {str(key) : val for key, val in res_dict.items()
                       if key not in excludekeys}

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
        jac = R.jacobian(sig0=self.sig0, dB=self.dB, param_list=neworder)

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
            d_inner = sp.lambdify(tausymb, sp.diff(orig_V.tau,
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

                # calculate "outer" * "inner" derivative for sparse matrices
                if np.atleast_1d(dtau_dx).shape == R.t_0.shape:
                     newjacdict[str(i)] = newjacdict[str(i)].multiply(
                         dtau_dx.reshape(np.prod(dtau_dx.shape)))
                else:
                    newjacdict[str(i)] = newjacdict[str(i)].multiply(dtau_dx)
            else:
                # calculate "outer" * "inner" derivative for numpy arrays
                if np.atleast_1d(dtau_dx).shape == R.t_0.shape:
                     newjacdict[str(i)] = newjacdict[str(i)] * dtau_dx.reshape(
                         np.prod(dtau_dx.shape))
                else:
                    newjacdict[str(i)] = newjacdict[str(i)] * dtau_dx

        # same for omega
        for i in set(map(str, omegasymb)) & set(param_dyn_dict.keys()):
            d_inner = sp.lambdify(omegasymb, sp.diff(orig_V.omega,
                                                     sp.Symbol(i)),
                                  modules=['numpy'])
            domega_dx = d_inner(*[res_dict[str(i)] for i in omegasymb])

            if np.isscalar(domega_dx):
                newjacdict[str(i)] = newjacdict[str(i)] * domega_dx
            elif isspmatrix(newjacdict[str(i)]):
                # In case the parameter is varying temporally, it must be
                # repeated by the number of incidence-angles in order
                # to have correct array-processing (it is assumed that no
                # parameter is incidence-angle dependent itself)

                # calculate "outer" * "inner" derivative for sparse matrices
                if np.atleast_1d(domega_dx).shape == R.t_0.shape:
                     newjacdict[str(i)] = newjacdict[str(i)].multiply(
                         domega_dx.reshape(np.prod(domega_dx.shape)))
                else:
                    newjacdict[str(i)] = newjacdict[str(i)].multiply(domega_dx)
            else:
                # calculate "outer" * "inner" derivative for numpy arrays
                if np.atleast_1d(domega_dx).shape == R.t_0.shape:
                     newjacdict[str(i)] = newjacdict[str(i)] * domega_dx.reshape(
                         np.prod(domega_dx.shape))
                else:
                    newjacdict[str(i)] = newjacdict[str(i)] * domega_dx

        # same for NormBRDF
        for i in set(map(str, Nsymb)) & set(param_dyn_dict.keys()):
            d_inner = sp.lambdify(Nsymb, sp.diff(orig_SRF.NormBRDF,
                                                 sp.Symbol(i)),
                                  modules=['numpy'])
            dN_dx = d_inner(*[res_dict[str(i)] for i in Nsymb])

            if np.isscalar(dN_dx):
                newjacdict[str(i)] = newjacdict[str(i)] * dN_dx
            elif isspmatrix(newjacdict[str(i)]):
                # In case the parameter is varying temporally, it must be
                # repeated by the number of incidence-angles in order
                # to have correct array-processing (it is assumed that no
                # parameter is incidence-angle dependent itself)

                # calculate "outer" * "inner" derivative for sparse matrices
                if np.atleast_1d(dN_dx).shape == R.t_0.shape:
                     newjacdict[str(i)] = newjacdict[str(i)].multiply(
                         dN_dx.reshape(np.prod(dN_dx.shape)))
                else:
                    newjacdict[str(i)] = newjacdict[str(i)].multiply(dN_dx)
            else:
                # calculate "outer" * "inner" derivative for numpy arrays
                if np.atleast_1d(dN_dx).shape == R.t_0.shape:
                     newjacdict[str(i)] = newjacdict[str(i)] * dN_dx.reshape(
                         np.prod(dN_dx.shape))
                else:
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

    def _get_res_df(self):
        '''
        return a pandas DataFrame with the obtained parameters
        '''
        newdict = dict()
        for key, val in self.res_dict.items():
            if isinstance(val[1], (int, float)):
                newdict[key] = chain.from_iterable(repeat(val[0], val[1]))
            else:
                newdict[key] = chain.from_iterable(
                    [repeat(i, int(j)) for i,j in np.array(val).T])
        return pd.DataFrame(newdict, self.index)
    def __get_data(self, prop):
        if not hasattr(self, 'dataset_used'):
            print('call _reinit() or performfit() to set the data first!')
            return
        else:
            if prop in ['inc', 'sig']:
                return rectangularize(self.dataset_used[prop].values)
            elif prop in ['weights', 'mask']:
                _, weights, mask = rectangularize(self.dataset_used.inc.values,
                                                  weights_and_mask=True)
                if prop == 'weights':
                    return weights
                if prop == 'mask':
                    return mask

    data = property(partial(__get_data, prop='sig'))
    inc = property(partial(__get_data, prop='inc'))
    weights = property(partial(__get_data, prop='weights'))
    mask = property(partial(__get_data, prop='mask'))

    res_df = property(_get_res_df)


    def _calc_slope_curv(self, R=None, res_dict=None, fixed_dict=None,
                         return_components=False):
        '''
        function to calculate the monostatic slope and curvature
        of the model

        Parameters:
        ------------
        R: RT1-object
           the rt1-object for which the results shall be calculated
        res_dict: dict
                  a dictionary containing all parameter-values that should
                  be updated before calling R.calc()
        return_components: bool (default=False)
                           indicator if the individual components or only
                           the total backscattered radiation are returned
                           (useful for quick evaluation of a model)
        Returns:
        ----------
        model_calc: the output of R.calc() (as intensity or sigma_0)
                    in linear-units or dB corresponding to the specifications
                    defined in the rtfits-class.
        '''

        if R is None:
            try:
                R = self.R
            except AttributeError:
                assert False, 'R is not available and must be provided'
        if res_dict is None:
            try:
                res_dict = self.res_dict
            except AttributeError:
                assert False, 'res_dict is not available and must be provided'
        if fixed_dict is None:
            try:
                fixed_dict = self.fixed_dict
            except AttributeError:
                assert False, 'fixed_dict is not available and must be provided'


        # ensure correct array-processing
        res_dict = {key:np.atleast_1d(val)[:,np.newaxis] for
                    key, val in res_dict.items()}
        res_dict.update(fixed_dict)

        # store original V and SRF
        orig_V = copy.deepcopy(R.V)
        orig_SRF = copy.deepcopy(R.SRF)

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        # and generate a function to evaluate the symbolic representation
        try:
            tausymb = R.V.tau.free_symbols
            taufunc = sp.lambdify(tausymb, R.V.tau,
                                  modules=['numpy'])
        except Exception:
            tausymb = set()
            taufunc = None
        try:
            omegasymb = R.V.omega.free_symbols
            omegafunc = sp.lambdify(omegasymb, R.V.omega,
                                    modules=['numpy'])
        except Exception:
            omegasymb = set()
            omegafunc = None
        try:
            Nsymb = R.SRF.NormBRDF.free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF,
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

        # exclude all keys that are not needed to calculate the fn-coefficients
        # vsymb and srfsymb must be subtracted in case the same symbol is used
        # for omega, tau or NormBRDF definition and in the function definiton
        excludekeys = ['omega', 'tau', 'NormBRDF', 'bsf',
                       *[str(i) for i in set(toNlist - vsymb - srfsymb)]]

        strparam_fn = {str(key) : val for key, val in res_dict.items()
                       if key not in excludekeys}

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

    def _init_V_SRF(self, V_props, SRF_props, setdict=dict()):
        '''
        initialize a volume and a surface scattering function based on
        a list of dicts

        Parameters
        ----------
        V_params, SRF_params : dict
            a dict that defines all variables needed to initialize the
            selected volume (surface) scattering function.

            if the value is a string, it will be converted to a sympy
            expression to determine the variables of the resulting expression

        V_name, SRF_name : str
            the name of the volume (surface)-scattering function.
        setdict : TYPE, optional
            DESCRIPTION. The default is dict().

        Returns
        -------
        V : a function of rt1.volume
            the used volume-scattering function.
        SRF : a function of rt1.surface
            the used surface-scattering function.

        '''
        V_dict = dict()
        for key, val in V_props.items():
            if key == 'V_name': continue

            # check if val is directly provided in setdict, if yes use it
            if val in setdict:
                useval = setdict[val]
            # check if val is a number, if yes use it directly
            elif isinstance(val, (int, float, np.ndarray)):
                useval = val
            # in any other case, try to sympify the provided value
            # to determine the free variables of the resulting equation and
            # then replace them by the corresponding values in setdict
            else:
                # convert to sympy expression (check doc for use of _clash)
                useval = sp.sympify(val, _clash)
                # in case parts of the expression are provided in setdict,
                # replace them with the provided values
                replacements = dict()
                for val_i in useval.free_symbols:
                    if str(val_i) in setdict:
                        replacements[val_i] = setdict[str(val_i)]
                useval = useval.xreplace(replacements)


            V_dict[key] = useval

        # same for SRF
        SRF_dict = dict()
        for key, val in SRF_props.items():
            if key == 'SRF_name': continue
            if str(val) in setdict:
                useval = setdict[str(val)]
            elif isinstance(val, (int, float, np.ndarray)):
                useval = val
            else:
                useval = sp.sympify(val, _clash)
                replacements = dict()
                for val_i in useval.free_symbols:
                    if str(val_i) in setdict:
                        replacements[val_i] = setdict[str(val_i)]
                useval = useval.xreplace(replacements)
            SRF_dict[key] = useval

        # initialize the volume-scattering function
        V = getattr(rt1_v, V_props['V_name'])(**V_dict)
        # initialize the surface-scattering function
        SRF = getattr(rt1_s, SRF_props['SRF_name'])(**SRF_dict)

        return V, SRF

    def monofit(self, V, SRF, dataset, param_dict, bsf=0.,
                bounds_dict={}, fixed_dict={}, param_dyn_dict={},
                fn_input=None, _fnevals_input=None, int_Q=True,
                lambda_backend=_init_lambda_backend, verbosity=2,
                intermediate_results=False, re_init=False,
                **kwargs):
        '''
        Perform least-squares fitting of omega, tau, NormBRDF and any
        parameter used to define V and SRF to sets of monostatic measurements.

        Parameters:
        ------------
        V: RT1.volume class object
            The volume scattering phase-function used to define the fit-model.
            Attention: if omega and/or tau are set to None, the values
            provided by V.omega and/or V.tau are used as constants!
        SRF: RT1.surface class object
             The surface BRDF used to define the fit-model
             Attention if NormBRDF is set to None, the values defined by
             SRF.NormBRDF will be used as constants!
        dataset: array-like or pandas.DataFrame
                 - if array-like:
                   list of input-data and incidence-angles arranged in the form
                   [[inc_0, data_0], [inc_1, data_1], ...]
                   where inc_i denotes the incident zenith-angles in radians
                   and the data_i denotes the corresponding data-values
                   (If the dataset provided is not rectangular, it will be
                    rectangularized using the _preparedata()-function.)
                 - if pandas.DataFrame
                   pandas-DataFrame with columns ['inc', 'sig', 'orig_index']
                   and any number of additional columns that represent
                   auxiliary datasets. The values must be provided as lists!
        param_dict: dict
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

        bounds_dict: dict
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
        param_dyn_dict: dict
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
        fn_input: array-like
                  a slot for pre-calculated fn-coefficients. if the same model
                  has to be fitted to multiple datasets, the fn-coefficients
                  that are returned in the first fit can be used as input for
                  the second fit to avoid repeated calculations.
        _fnevals_input: callable
                        a slot for pre-compiled function to evaluate the
                        fn-coefficients. Note that once the _fnevals function
                        is provided, the fn-coefficients are no longer needed
                        and have no effect on the calculated results!
        int_Q: bool (default = True)
               indicator if interaction-terms should be included
               (Note: they are always ommitted when calculating the jacobian!)
        lambda_backend: string
                        select method for generating the _fnevals functions
                        if they are not provided explicitly.
                        The default is 'cse_symengine_sympy' if symengine is
                        installed and 'cse' otherwise.
        verbosity: int
                   set verbosity level of rt1-module
        intermediate_results: bool (default = False)
                              indicator if intermediate results should be
                              stored (for analysis purpose only). If True, a
                              dictionary will be generated that contains
                              detailed results for each iteration-step of the
                              fit-procedure. It is structured as follows:

                                   'jacobian': a list of dicts with the
                                                jacobians for each fitted
                                                parameter

                                   'errdict': {'abserr' : list of RMSE,
                                                'relerr' : list of RMSE/data}

                                   'parameters': list of parameter-result
                                                  dictionaries for each step
        re_init: bool (default = False)
                 if re_init is True, all actions EXCEPT the actual fit will
                 be executed. This re-initializes the Fits object based on the
                 input in the state as if the fit has already been performed.

        kwargs:
                keyword arguments passed to scipy's least_squares function

        Returns:
        ---------
        res_lsq: dict
                 output of scipy's least_squares function
        R: RT1-object
           the RT1-object used to perform the fit
        data: array-like
              used dataset for the fit
        inc: array-like
             used incidence-angle data for the fit
        mask: array-like(bool)
              the masked that needs to be applied to the rectangularized
              dataset to get the valid entries (see preparedata-function)
        weights: array-like
                 the weighting-matrix that has been applied to correct for the
                 rectangularization of the dataset (see preparedata-function)
        res_dict: dict
                  a dictionary containing the fit-results for the parameters
        start_dict: dict
                    a dictionary containing the used start-values
        fixed_dict: dict
                    a dictionary containing the parameter-values that have been
                    used as constants during the fit
        '''
        # set up the dictionary for storing intermediate results
        if intermediate_results is True:
            if not hasattr(self, 'intermediate_results'):
                self.intermediate_results = {'parameters':[],
                                             'residuals':[],
                                             'jacobian':[]}



        # generate a list of the names of the parameters that will be fitted.
        # (this is necessary to ensure correct broadcasting of values since
        # dictionarys do)
        order = [i for i, v in param_dict.items() if v is not None]

        # will be used to assign the values to the individual parameters
        # (the returned parameters are given as a concatenated array
        # of the shape [*p0, *p1, *p2, ...]) where p1,p2,p3 are a list of
        # values for each parameter
        splits = np.cumsum([len(np.unique(param_dyn_dict[key]))
                            for key in order])

        # will be used to re-assign the values to the index of the dataset
        # this procedure is necessary to ensure correct broadcasting in case
        # the param_dyn_dict values are not sorted, e.g. [1,1,2,3,1,1,4,...]
        from itertools import groupby

        repeatdict = dict()
        for key in order:
            # get the (sorted) unique values and locations in param_dyn_dict
            dynnr, uniquewhere = np.unique(param_dyn_dict[key],
                                           return_inverse=True)
            # get value positions and number of repetitions to reconstruct an
            # array indexed like the dataset
            # (param_dyn_dict[key] = np.repeat(dynnr[repeats[0]], repeats[1]))
            repeats = [[], []]
            for i,j in groupby(uniquewhere):
                repeats[0].append(i)
                repeats[1].append(sum(1 for x in j))
            if len(set(repeats[1])) == 1:
                repeats[1] = repeats[1][0]

            repeatdict[key] = repeats

        # set index
        self.dataset_used = dataset
        # generate a datetime-index from the given groups
        self._setindex(self.setindex)
        # get a dict of the mean datetime-values for each parameter

        # preparation of data for fitting
        [inc, data, weights, Nmeasurements,
         mask, new_fixed_dict] = self._preparedata(dataset)

        # update 'fixed_dict' with timeseries provided via 'dataset'
        # ensure that only parameters that are explicitely mentioned in
        # fixed_dict are passed to the fit-procedure as fixed datasets
        # (additional columns of 'dataset' do not affect the fit)

        for key, val in fixed_dict.items():
            if isinstance(val, str) and val == 'auxiliary':
                    assert key in new_fixed_dict, \
                        f"auxiliary data for '{key}' is missing!"

            if key in new_fixed_dict:
                fixed_dict[key] = new_fixed_dict[key]

        # check if tau, omega or NormBRDF is given in terms of sympy-symbols
        try:
            tausymb = V.tau.free_symbols
        except Exception:
            tausymb = set()
        try:
            omegasymb = V.omega.free_symbols
        except Exception:
            omegasymb = set()
        try:
            Nsymb = SRF.NormBRDF.free_symbols
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
            split_vals = np.split(params, splits)[:-1]
            newdict = dict()
            for key, val in zip(order, split_vals):
                # value positions and consecutive repetitions
                reloc, rep = repeatdict[key]
                newdict[key] = (val[reloc], rep)

            # calculate the residuals
            errs = self._calc_model(R, newdict, fixed_dict).flatten() - data
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
            split_vals = np.split(params, splits)[:-1]
            newdict = dict()
            for key, val in zip(order, split_vals):
                # value positions and consecutive repetitions
                reloc, rep = repeatdict[key]
                newdict[key] = (val[reloc], rep)

            # calculate the jacobian
            # (no need to include weighting matrix in here since the jacobian
            # of the artificially added colums must be the same!)
            jac = self._calc_jac(R, newdict, fixed_dict, param_dyn_dict, order)

            return jac


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

        # perform the actual fit
        if re_init is True:
            if getattr(self, 'fit_output', None) is not None:
                res_lsq = self.fit_output
            else:
                res_lsq = None
                self.fit_output = None
        else:
            # perform actual fitting
            res_lsq = least_squares(fun, startvals, bounds=bounds,
                                    jac=dfun, **kwargs)

        # generate a dictionary to assign values based on fit-results
        # split the obtained result with respect to the individual parameters
        if res_lsq is not None:
            # split concatenated parameter-results given in res_lsq.x
            # (don't use the last array since it is empty)

            split_vals = np.split(res_lsq.x, splits)[:-1]
            split_startvals = np.split(res_lsq.x, splits)[:-1]

            res_dict = dict()
            start_dict = dict()
            for key, val, startval in zip(order, split_vals, split_startvals):
                # value positions and consecutive repetitions
                reloc, rep = repeatdict[key]
                res_dict[key] = [val[reloc], rep]
                start_dict[key] = [startval[reloc], rep]

            self.fit_output = res_lsq
            self.res_dict = res_dict
            self.start_dict = start_dict


        # ------------------------------------------------------------------
        # ------------ prepare output-data for later convenience -----------


        self.R = R
        self.fixed_dict = fixed_dict
        self.res_dict = getattr(self, 'res_dict', dict())
        self.start_dict = getattr(self, 'start_dict', dict())


        # for downward compatibility
        return [self.fit_output, self.R, self.data, self.inc, self.mask,
                self.weights, self.res_dict, self.start_dict,
                self.fixed_dict]


    def _set_performfit_dicts(self, defdict=None):
        '''
        Generate RT-1 specifications based on the provided "defdict".

        ... used to simplify the model-specification for 'rtfits.performfit()'

        Parameters
        ----------
        defdict : dict
            see documentation of rt1.rtfits.Fits class

        Returns
        -------
        list
            a list of dicts corresponding to:

            [fixed_dict, setdict, startvaldict,
            timescaledict, boundsvaldict, manual_dyn_df] .

        '''
        if defdict is None:
            defdict = self.defdict

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
                # omega and tau must not be a sympy-symbol name
                if key not in ['omega', 'tau']:
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
                elif defdict[key][2] == 'index':
                    indexdyn = pd.DataFrame({key:1}, self.dataset.index
                                            ).groupby(axis=0, level=0
                                                      ).ngroup().to_frame()
                    indexdyn.columns = [key]
                    if manual_dyn_df is None:
                        manual_dyn_df = pd.DataFrame()
                    manual_dyn_df = pd.concat([manual_dyn_df, indexdyn],
                                              axis=1)
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
                if isinstance(defdict[key][1], (int, float)) and not \
                    isinstance(defdict[key][1], bool):
                    setdict[key] = defdict[key][1]
                else:
                    # if value is provided as array, add it to fixed_dict
                    if key not in ['omega', 'tau']:
                        # omega and tau must not be a sympy-symbol name
                        # TODO same as above ...why is this necessary?
                        # TODO what about 'NormBRDF'?
                        setdict[key] = sp.var(key)
                    else:
                        # a dummy value that will be replaced by rtfits.monofit
                        setdict[key] = 100

                    if isinstance(defdict[key][1], str) and \
                        defdict[key][1] == 'auxiliary':
                        fixed_dict[key] = 'auxiliary'
                    else:
                        fixed_dict[key] = defdict[key][1]

        return [fixed_dict, setdict, startvaldict,
                timescaledict, boundsvaldict, manual_dyn_df]


    def performfit(self, dataset=None, defdict=None, set_V_SRF=None,
                   fitset=None, re_init=False):
        '''
        Setup a RT-1 specifications and perform a fit based on the provided
        inputs (dataset, defdict, set_V_SRF and fitsset).


        Parameters
        -----------
        dataset: pandas.DataFrame
                 override the attribute of the parent Fits-object.
                 see docstring of rtfits.Fits for details
        defdict: dict
                 override the attribute of the parent Fits-object.
                 see docstring of rtfits.Fits for details
        set_V_SRF: callable
                 override the attribute of the parent Fits-object.
                 see docstring of rtfits.Fits for details
        fitset: dict
                 override the attribute of the parent Fits-object.
                 see docstring of rtfits.Fits for details
        re_init: bool (default = False)
                 if re_init is True, all actions EXCEPT the actual fit will
                 be executed. This re-initializes the Fits object based on the
                 input in the state as if the fit has already been performed.

        '''

        assert self.setindex in ['mean','first', 'last', 'original'], \
            "setindex must be either 'mean', 'first', 'last' or 'original'"

        # TODO change to only using rtfits objects!
        # (i.e. it is not necessary to manually specify defdict etc.)
        if dataset is None: dataset = self.dataset
        if defdict is None: defdict = self.defdict
        if set_V_SRF is None: set_V_SRF = self.set_V_SRF
        if fitset is None: fitset = self.fitset

        # get dictionary for initialization of fit
        [fixed_dict, setdict, startvaldict, timescaledict,
         boundsvaldict, manual_dyn_df] =self._set_performfit_dicts(defdict)


        # set V and SRF based on setter-function
        if callable(set_V_SRF):
            V, SRF = set_V_SRF(**setdict)
        elif isinstance(set_V_SRF, dict):
            V, SRF = self._init_V_SRF(**set_V_SRF,
                                      setdict=setdict)

        # set frequencies of fitted parameters
        freq = []
        freqkeys = []
        for key in timescaledict:
            freq += [timescaledict[key]]
            freqkeys += [[key]]

        dataset_used, param_dyn_dict = self._generatedataset(
                dataset=dataset, dyn_keys=startvaldict.keys(),
                freq=freq, freqkeys=freqkeys, manual_dyn_df=manual_dyn_df,
                fixed_dict=fixed_dict)


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
        self.monofit(V=V, SRF=SRF,
                     dataset=dataset_used,
                     param_dict=param_dict,
                     bsf = setdict.get('bsf', 0.),
                     bounds_dict=bounds_dict,
                     fixed_dict=fixed_dict,
                     param_dyn_dict=param_dyn_dict,
                     re_init=re_init,
                     **fitset)



    def _evalfunc(self, reader=None, reader_arg=None, fitset=None,
                  preprocess=None, postprocess=None, exceptfunc=None,
                  preeval_fn=False):
        """
        Initialize a Fits-instance and perform a fit.
        (used for parallel processing)

        Parameters
        ----------
        reader : callable, optional
            A function whose first return-value is a pandas.DataFrame that can
            be used as a 'dataset'. (see documentation of 'rt1.rtfits.Fits'
            for details on how to properly specify a RT-1 dataset).
            Any additional (optional) return-values will be appended to the
            Fits-object in the 'aux_data' attribute. The default is None.
        reader_arg : dict
            A dict of arguments passed to the reader.
        fitset : dict, optional
            override the fitset-dict of the parent Fits-object.
            (see documentation of 'rt1.rtfits.Fits')
            The default is None.
        preprocess : callable
            A function that will be called prior to the start of the processing
            It is called via:

            >>> preprocess(reader_arg)
        postprocess : callable
            A function that accepts a rt1.rtfits.Fits object and a dict
            as arguments and returns any desired output.

            Internally it is called after calling performfit() via:

            >>> ...
            >>> fit.performfit()
            >>> return postprocess(fit, reader_arg)
        exceptfunc : callable, optional
            A function that is called in case an exception occurs.
            It is (roughly) implemented via:

            >>> for reader_arg in reader_args:
            >>>     try:
            >>>         ...
            >>>         ... perform fit ...
            >>>         ...
            >>>     except Exception as ex:
            >>>         exceptfunc(ex, reader_arg)

            The default is None, in which case the exception will be raised.

        Returns
        -------
        The used 'rt1.rtfit.Fits' object or the output of 'postprocess()'
        """

        try:
            # call preprocess function if provided
            if callable(preprocess):
                preprocess(reader_arg)

            if fitset is None:
                fitset = self.fitset

            # if a reader (and no dataset) is provided, use the reader
            read_data = reader(**reader_arg)
            # check for multiple return values and split them accordingly
            # (any value beyond the first is appended as aux_data)
            if isinstance(read_data, pd.DataFrame):
                dataset = read_data
                aux_data = None
            elif (isinstance(read_data, (list, tuple))
                  and isinstance(read_data[0], pd.DataFrame)):
                if len(read_data) == 2:
                    dataset, aux_data = read_data
                elif  len(read_data) > 2:
                    dataset = read_data[0]
                    aux_data = read_data[1:]
            else:
                raise TypeError('the first return-value of reader function ' +
                                'must be a pandas DataFrame')


            # pre-evaluate fn-coefficients
            if preeval_fn is True:
                fit = Fits(sig0=self.sig0, dB=self.dB, dataset = dataset,
                           set_V_SRF=self.set_V_SRF, defdict=self.defdict,
                           fitset=fitset, setindex=self.setindex)
                fit.performfit(re_init=True)

                return fit

            # perform the fit
            fit = Fits(sig0=self.sig0, dB=self.dB, dataset = dataset,
                       set_V_SRF=self.set_V_SRF, defdict=self.defdict,
                       fitset=fitset, setindex=self.setindex)
            fit.performfit()

            # append auxiliary data
            if aux_data is not None:
                fit.aux_data = aux_data

            # if a post-processing function is provided, return its output,
            # else return the fit-object directly
            if callable(postprocess):
                return postprocess(fit, reader_arg)
            else:
                return fit

        except Exception as ex:
            if callable(exceptfunc):
                return exceptfunc(ex, reader_arg)
            else:
                raise ex


    def processfunc(self, ncpu=1, reader=None, reader_args=None,
                    fitset=None, preprocess=None, postprocess=None,
                    exceptfunc=None, finaloutput=None, pool_kwargs=None):
        """
        Evaluate a RT-1 model on a single core or in parallel using either
            - a list of datasets or
            - a reader-function together with a list of arguments that
              will be used to read the datasets

        Notice:
            On Windows, if multiprocessing is used, you must protect the call
            of this function via:
            (see for example: )

            >>> if __name__ == '__main__':
                    fit.processfunc(...)

            In order to allow pickling the final rt1.rtfits.Fits object,
            it is required to store ALL definitions within a separate
            file and call processfunc in the 'main'-file as follows:

            >>> from specification_file import fit reader fitset ... ...
                if __name__ == '__main__':
                    fit.processfunc(ncpu=5, reader=reader,
                                    fitset=fitset, ... ...)

        Parameters
        ----------
        ncpu : int, optional
            The number of kernels to use. The default is 1.
        reader : callable, optional
            A function whose first return-value is a pandas.DataFrame that can
            be used as a 'dataset'. (see documentation of 'rt1.rtfits.Fits'
            for details on how to properly specify a RT-1 dataset).
            Any additional (optional) return-values will be appended to the
            Fits-object in the 'aux_data' attribute. The default is None.
        reader_args : list, optional
            A list of dicts that will be passed to the reader-function.
            The default is None.
        fitset : dict, optional
            override the fitset-dict of the parent Fits-object.
            (see documentation of 'rt1.rtfits.Fits')
            The default is None.
        preprocess : callable
            A function that will be called prior to the start of the processing
            It is called via:

            >>> preprocess(reader_arg)
        postprocess : callable
            A function that accepts a rt1.rtfits.Fits object and a dict
            as arguments and returns any desired output.

            Internally it is called after calling performfit() via:

            >>> ...
            >>> fit.performfit()
            >>> return postprocess(fit, reader_arg)
        exceptfunc : callable, optional
            A function that is called in case an exception occurs.
            It is (roughly) implemented via:

            >>> for reader_arg in reader_args:
            >>>     try:
            >>>         ...
            >>>         ... perform fit ...
            >>>         ...
            >>>     except Exception as ex:
            >>>         exceptfunc(ex, reader_arg)

            The default is None, in which case the exception will be raised.
        finaloutput : callable, optional
            A function that will be called on the list of returned objects
            after the processing has been finished.
        pool_kwargs : dict
            A dict with additional keyword-arguments passed to the
            initialization of the multiprocessing-pool via:

            >>> mp.Pool(ncpu, **pool_kwargs)


        Returns
        -------
        res : list
            A list of rt1.rtfits.Fits objects or a list of outputs of the
            postprocess-function.

        """

        if fitset is None: fitset = self.fitset
        if pool_kwargs is None: pool_kwargs = dict()

        if 'int_Q' in fitset and fitset['int_Q'] is True:
            # pre-evaluate the fn-coefficients if interaction terms are used
            fit = self._evalfunc(reader=reader, reader_arg=reader_args[0],
                                 fitset={**fitset, 'max_nfev':0, 'verbose':2},
                                 preprocess=None,
                                 postprocess=None,
                                 exceptfunc=exceptfunc,
                                 preeval_fn=True)

            fitset['_fnevals_input'] = fit.R._fnevals

        if ncpu > 1:
            print('start of parallel evaluation')
            with mp.Pool(ncpu, **pool_kwargs) as pool:
                # loop over the reader_args
                res = pool.starmap(self._evalfunc,
                                   zip(repeat(reader),
                                       reader_args,
                                       repeat(fitset),
                                       repeat(preprocess),
                                       repeat(postprocess),
                                       repeat(exceptfunc)))

        else:
            print('start of single-core evaluation')
            res = []
            for reader_arg in reader_args:
                res.append(self._evalfunc(reader=reader, reader_arg=reader_arg,
                                          fitset=fitset,
                                          preprocess=preprocess,
                                          postprocess=postprocess,
                                          exceptfunc=exceptfunc))

        if callable(finaloutput):
            return finaloutput(res)
        else:
            return res


    def dump(self, path, mini=False):
        '''
        Save the rt1.rtfits.Fits object using cloudpickle.dump()

        The generated (platform and environment-specific) file can be loaded
        via:

        >>> import cloudpickle
            with open(--path-to-file--, 'rb') as file
                fit = cloudpickle.load(file)

        Parameters
        ----------
        path : str
            The path to the file that should be created.
        mini : bool, optional
            Indicator if unnecessary attributes should be removed before
            pickling or not (to avoid storing duplicated data).
            To re-create the attributes, run

            >>> fit.performfit(re_init=True)

            The default is False.
        '''

        if mini is True:
            self._rt1_dump_mini = True
        else:
            try:
                delattr(self, '_rt1_dump_mini')
            except AttributeError:
                pass

        with open(path, 'wb') as file:
            cloudpickle.dump(self, file)


    def _reinit(self):
        '''
        re-initialize a fit-object based on the provided input without
        performing the actual fit
        '''
        self.performfit(re_init=True)



class RT1_configparser(object):
    def __init__(self, configpath):
        # setup config (allow empty values -> will result in None)
        self.config = ConfigParser(allow_no_value=True)
        # avoid converting uppercase characters (see https://stackoverflow.com/a/19359720/9703451)
        self.config.optionxform = str
        # read config file
        self.cfg = self.config.read(configpath)

        # keys that will be converted to int, float or bool
        self.lsq_parse_props = dict(section = 'least_squares_kwargs',
                                    int_keys = ['verbose', 'max_nfev'],
                                    float_keys = ['ftol', 'gtol', 'xtol'])

        self.fitprop_parse_props = dict(section = 'general_RT1_properties',
                                        bool_keys = ['int_Q'])

        self.fitargs_parse_props = dict(section = 'fits_kwargs',
                                        bool_keys = ['sig0', 'dB'])

    def _parse_dict(self, section, int_keys=[], float_keys=[], bool_keys=[]):
        '''
        a function to convert the parsed string values to int, float or bool
        (any additional values will be left unchanged)

        Parameters
        ----------
        section : str
            the name of the section in the config-file.
        int_keys : list
            keys that should be converted to int.
        float_keys : list
            keys that should be converted to float.
        bool_keys : list
            keys that should be converted to bool.

        Returns
        -------
        parsed_dict : dict
            a dict with the converted values.

        '''

        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            if key in float_keys:
                val = inp.getfloat(key)
            elif key in int_keys:
                val = inp.getint(key)
            elif key in bool_keys:
                val = inp.getboolean(key)
            else:
                val = inp[key]

            parsed_dict[key] = val
        return parsed_dict

    def _parse_V_SRF(self, section):

        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            val = None
            if key == 'ncoefs':
                val = inp.getint(key)
            else:
                # try to convert floats, if it fails return the string
                try:
                    val = inp.getfloat(key)
                except:
                    val = inp[key]

            parsed_dict[key] = val

        return parsed_dict


    def _parse_defdict(self, section):
        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            val = inp[key]

            if val.startswith('[') and val.endswith(']'):
                val = val[1:-1].replace(' ', '').split(',')
            else:
                assert False, f'the passed defdict for {key} is not a list'

            # convert values
            parsed_val = []
            parsed_val += [bool(val[0])]
            parsed_val += [float(val[1])]
            if val[2] == 'None':
                parsed_val += [None]
            else:
                parsed_val += [val[2]]

            parsed_val += [([float(val[3])],
                            [float(val[4])])]

            parsed_dict[key] = parsed_val

        return parsed_dict


    def get_config(self):
        lsq_dict = self._parse_dict(**self.lsq_parse_props)
        fitprop_dict = self._parse_dict(**self.fitprop_parse_props)
        fitset = dict(**lsq_dict, **fitprop_dict)

        fits_kwargs = self._parse_dict(**self.fitargs_parse_props)
        defdict = self._parse_defdict('defdict')
        set_V_SRF = dict(V_props = self._parse_V_SRF('RT1_V'),
                         SRF_props = self._parse_V_SRF('RT1_SRF'))

        return dict(fitset=fitset,
                    defdict=defdict,
                    set_V_SRF=set_V_SRF,
                    fits_kwargs=fits_kwargs)


    def get_fitobject(self):
        cfg = self.get_config()

        rt1_fits = Fits(dataset=None, defdict=cfg['defdict'],
                        set_V_SRF=cfg['set_V_SRF'],
                        fitset=cfg['fitset'], **cfg['fits_kwargs'])

        return rt1_fits


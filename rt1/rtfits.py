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
from scipy.interpolate import interp1d

from .scatter import Scatter
from .rt1 import RT1, _init_lambda_backend
from .general_functions import meandatetime, rectangularize, \
    split_into, dt_to_hms, update_progress, groupby_unsorted

from .rtplots import plot as rt1_plots

from . import surface as rt1_s
from . import volume as rt1_v
from . import __version__ as _RT1_version

import copy
import multiprocessing as mp
import ctypes
from itertools import repeat, count, chain
from functools import lru_cache, partial
from operator import itemgetter, add

from timeit import default_timer as tick
from datetime import timedelta

try:
    import cloudpickle
except ModuleNotFoundError:
    print('cloudpickle could not be imported, .dump() will not work!')

class Fits(Scatter):
    '''
    Class to perform nonlinear least-squares fits to data.


    Parameters
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
             a pandas.DataFrame with columns `inc` and `sig` defined
             where `inc` referrs to the incidence-angle in radians, and
             `sig` referrs to the measurement value (corresponding to
             the assigned sig0 and dB values)

             - If a column `data_weights` is provided, the residuals in the
               fit-procedure will be weighted accordingly.
               (e.g. residuals = weights * calculated_residuals )
             - If columns `param_dyn` are provided where `param` is the name of
               a parameter that is intended to be fitted, the entries will be
               used to assign the dynamics of the corresponding parameter
               (see defdict 'freq' entry for further details)
             - If columns with names corresponding to parameters are provided
               and the corresponding entry in the 'val' parameter of defdict
               is set to 'auxiliary', then the provided data will be used
               as auxiliary data for the parameter.

    defdict: dict (default = None)
             a dictionary of the following structure:
             (the dict will be copied internally using copy.deepcopy(dict))

             >>> defdict = {'key1' : [fitQ, val, freq, ([min], [max]), interp],
             >>>            'key2' : [fitQ, val, freq, ([min], [max]), interp],
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
                       - if 'manual', the DataFrame column "key_dyn" provided
                         in the dataset will be used to assign the temporal
                         variability within the fit
                       - if 'index', a unique value will be fitted to each
                         unique index of the provided dataset
                       - if freq corresponds to a pandas offset-alias, it
                         will be used together with the dataset-index to
                         assign the temporal variability within the fit
                         (see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)
                       - if both a pandas offset-alias and a dataset-column
                         "key_dyn" is provided, the the provided variability
                         will be superimposed onto the variability resulting
                         form the chosen offset-alias

                min, max: float (only needed if fitQ is True)
                          the boundary-values used within the fit

                interp: bool
                        indicator if the obtained values should be interpoalted
                        (using a quadratic interpolation function) or if a
                        step-function should be used.
                        (only effects dynamic parameters)

    set_V_SRF: callable or dict (default = None)
               either a function with the following structure:

               >>> def set_V_SRF(volume-keys, surface-keys):
               >>>     from rt1.volume import 'Volume-function'
               >>>     from rt1.surface import 'Surface function'
               >>>
               >>>     V = Volume-function(volume-keys)
               >>>     SRF = Surface-function(surface-keys)
               >>>
               >>>     return V, SRF

               or a dict that will be passed to _init_V_SRF() to initialize
               the V- and SRF-objects
    lsq_kwargs: dict (default = dict())
                a dictionary with keyword-arguments passed to
                scipy.optimize.least_squares
    int_Q: bool (default = True)
           indicator if the interaction-term should be evaluated or not.
    lambda_backend: str, optional
                    select method for generating the _fnevals functions
                    if they are not provided explicitly and int_Q is True
                    The default is 'symengine' if symengine is installed and
                    'sympy' otherwise.
    _fnevals_input: callable, optional
                    pre-evaluated functions used for evaluating the
                    interaction-term
                    -> use with care! you must ensure that the provided
                    function evaluates correctly for the used definitions
    _interp_vals: list, optional
                  a list of keys corresponding to parameters whose values
                  should be quadratically interpolated over the timespan
                  instead of using a step-function that assigns the obtained
                  value equally to all observations within the timespan.
                  -> use with care! this might cause unexpected behaviour!
    verbose : int, optional
              the verbosity of the print-outputs (also passed to
              scipy.optimize.least_squares and rt1.RT1)


    Attributes
    -----------

    index: array-like
        the unique index values of the provided dataset
    res_dict: dict
        a dictionary of the obtained fit-results
    res_df: pandas.DataFrame
        a pandas DataFrame of the obtained fit-results
    param_dyn_dict: dict
        a dict of the individual parameter-dynamics
    param_dyn_df: pandas.DataFrame
        a pandas DataFrame of the individual parameter-dynamics
    meandatetimes: dict
        a dict of the mean datetime-values for each parameter (with respect to
        the corresponding fit-groups)
    fit_output: scipy.optimize.OptimizeResult
        the output of scipy.optimize.least_squares
    R: rt1.RT1 object
       the RT1 object used
    V: rt1.RT1.volume object
        the volume-scattering object used
    SRF: rt1.RT1.surface object
        the surface-scattering object used
    data: array-like
        the used measurement malues
    inc: array-like
        the used incidence-angles
    mask: array-like
        a mask that indicates the values added to "data" and "inc" in order
        to obtain a rectangular array
    dataset_used: pandas.DataFrame
        a DataFrame of the used data grouped with respect to the
        temporal variations of the parameters that have been fitted
    fixed_dict: dict
        a dict of the auxiliary datasets used

    Methods
    ---------

    performfit(clear_cache=True, intermediate_results=False,\
               print_progress=False)
        perform a fit of the defined model to the dataset

    processfunc(ncpu=1, reader=None, reader_args=None,\
                lsq_kwargs=None, preprocess=None, postprocess=None,\
                exceptfunc=None, finaloutput=None, pool_kwargs=None)
        perform multiple fits of the defined model using multiprocessing

    dump(path, mini=True)
        dump the fits-object using cloudpickle.dump() to the specified path

    calc(param, inc, return_components=True, fixed_param = None)
        evaluate the defined model based on a given set of parameters and
        incidence-angle ranges

    '''

    def __init__(self, sig0=False, dB=False, dataset=None,
                 defdict=None, set_V_SRF=None, lsq_kwargs=None, int_Q=True,
                 lambda_backend=None, _fnevals_input=None,
                 verbose=2, **kwargs):

        self.sig0 = sig0
        self.dB = dB
        self.dataset = dataset
        self.set_V_SRF = copy.deepcopy(set_V_SRF)
        self.defdict = copy.deepcopy(defdict)
        if lsq_kwargs is None:
            self.lsq_kwargs = dict()
        else:
            self.lsq_kwargs = copy.deepcopy(lsq_kwargs)

        self.int_Q = int_Q
        self.lambda_backend = lambda_backend

        if self.lambda_backend is None:
            self.lambda_backend = _init_lambda_backend

        self._fnevals_input = _fnevals_input

        self.verbose = verbose

        # add plotfunctions
        self.plot = rt1_plots(self)


    def __update__(self):
        '''
        needed for downward compatibility
        '''

        if not hasattr(self, 'plot'):
            print('... re-initializing plot-functions')
            self.plot = rt1_plots(self)

        if not hasattr(self, 'verbose'):
            self.verbose = 2

        if hasattr(self, 'fitset'):
            self.lsq_kwargs = self.fitset
            del self.fitset
        if not hasattr(self, 'lsq_kwargs'):
            self.lsq_kwargs = dict()
        if not hasattr(self, 'int_Q'):
            self.int_Q = self.lsq_kwargs.pop('int_Q', True)
        if not hasattr(self, 'lambda_backend'):
            self.lambda_backend = self.lsq_kwargs.pop('lambda_backend',
                                                      _init_lambda_backend)
        if not hasattr(self, '_fnevals_input'):
            self._fnevals_input = self.lsq_kwargs.pop('_fnevals_input',
                                                      None)

        if (hasattr(self, 'res_dict')
            and np.all([len(val)==2
                       for _, val in self.res_dict.items()])
            and np.all([isinstance(val[0], list)
                       for _, val in self.res_dict.items()])):
                print('updating res-dict to new shape...')
                self.res_dict = {key:val[0]
                                 for key, val in self.res_dict.items()}

    def __setstate__(self, d):
        # this is done to support downward-compatibility with pickled results
        self.__dict__ = d
        self.__update__()


    def __getstate__(self):
        if '_rt1_dump_mini' in self.__dict__:
            # remove the dummy-attribute that indicates that we do a mini-dump
            delattr(self, '_rt1_dump_mini')

            # remove unnecessary data to save storage
            removekeys = ['fit_output', '_fnevals_input']
            returndict = {key: val for key, val in self.__dict__.items()}
            for key in removekeys:
                returndict[key] = None

            return returndict
        else:
            return self.__dict__


    def __setattr__(self, attr, value):
        # TODO write proper setters that do the job
        # clear the cache in case it is not empty and a
        # defining variable is set
        if attr in ['sig0', 'dB', 'dataset', 'defdict', 'set_V_SRF']:
            if not all(i == 0 for i in self._cached_arg_number):
                #print(f'{attr} has been set, clearing cache')
                self._clear_cache()

        super().__setattr__(attr, value)


    @property
    def _cached_props(self):
         '''
         a list of the names of the properties that are cached
         '''
         names = ['param_dyn_dict', 'param_dyn_df', '_groupindex', '_N_groups',
                  '_dataset_used', 'index', '_orig_index',
                  '_jac_assign_rule', 'meandatetimes', 'inc', 'data',
                  'data_weights', '_idx_assigns', '_param_assigns',
                  '_param_assigns_dataset', '_val_assigns', '_order',
                  'interp_vals', ]

         for i in ['tau', 'omega', 'N']:
             names += [f'_{i}_symb', f'_{i}_func', f'_{i}_diff_func']

         return names


    def _clear_cache(self):
        '''
        clear all cached properties
        '''
        # only clear the cache if at least one property has been cached
        if not all(i == 0 for i in self._cached_arg_number):
            for name in self._cached_props:
                getattr(Fits, name).fget.cache_clear()
            #print('...cache cleared')


    def _cache_info(self):
        '''
        print the state of the lru_cache for all cached properties
        '''
        text = []
        for name in self._cached_props:
            try:
                cinfo = getattr(Fits, name).fget.cache_info()
                text += [f'{name:<18}:   ' + f'{cinfo}']
            except:
                text += [f'{name:<18}:   ' + '???']
        print(*text, sep='\n')


    @property
    def _cached_arg_number(self):
        '''
        print the state of the lru_cache for all cached properties
        '''
        nums = []
        for name in self._cached_props:
            try:
                if '.' in name:
                    nsplit = name.split('.')
                    nums.append(getattr(getattr(self, nsplit[0]),
                                        nsplit[1]).cache_info().currsize)
                else:
                    nums.append(getattr(Fits, name).fget.cache_info().currsize)
            except:
                pass
        return nums


    @property
    @lru_cache()
    def interp_vals(self):
        interpkeys= [key for key, val in self.defdict.items()
                     if val[0] is True and len(val) >= 5 and val[4] is True]

        # TODO make a proper test
        # don't use meandatetimes -> it disrupts parallel processing
        # for key in interpkeys:
        #     assert np.all(self.meandatetimes[key][:-1]
        #                   <= self.meandatetimes[key][1:]), (
        #         'interpolation of unsorted parameter-groupings not possible!')
        return interpkeys


    @property
    @lru_cache()
    def param_dyn_dict(self):
        '''
        get index to assign grouping (with respect to the dataset-index)
        '''
        if self.dataset is None:
            return dict()
        else:

            # the names of the parameters that will be fitted
            dyn_keys = [key for key, val in self.defdict.items()
                        if val[0] is True]

            # set frequencies of fitted parameters
            # (group by similar frequencies)
            grps = groupby_unsorted(self._timescaledict.items(),
                                    key=itemgetter(1), get=itemgetter(0))
            freq = list(grps.keys())
            freqkeys = list(grps.values())
            param_dyn_dict = {}
            # initialize all parameters as scalar parameters
            for key in dyn_keys:
                param_dyn_dict[key] = list(repeat(1, len(self.dataset.index)))
            if freq is not None:
                for i, f in enumerate(freq):
                    try:
                        grp_idx = self.dataset.index.to_frame().groupby(
                                                        pd.Grouper(freq=f),
                                                        sort=False)
                    except ValueError:
                        raise ValueError(f'The provided frequency ({f}) of ' +
                                         f'{freqkeys[i]} is not a valid ' +
                                         'pandas datetime-offset string. ' +
                                         'Check the assignments in defdict!')
                    # get unique group indices for each datetime-group
                    for key in freqkeys[i]:
                        grp_data = []
                        for nval, [_, val] in enumerate(grp_idx):
                            grp_data += repeat(nval, len(val))
                        param_dyn_dict[key] = grp_data

            manual_dyn_df = self._manual_dyn_df
            if manual_dyn_df is not None:
                for key, val in manual_dyn_df.astype(str).items():
                    dd1 = np.char.zfill(
                        np.array(param_dyn_dict[key], dtype='str'),
                        len(max(np.array(param_dyn_dict[key], dtype='str'),
                                key=len)))
                    dd2 = val.str.zfill(len(max(val, key=len)))
                    # generate a combined (unique) integer
                    param_dyn_dict[key] = np.array(np.char.add(dd1, dd2),
                                                   dtype=int)
            return param_dyn_dict


    @property
    @lru_cache()
    def param_dyn_df(self):
        '''
        a data-frame with the individual group-indexes for each parameter
        (with respect to the unique index values of the provided dataset)
        '''
        if self.dataset is not None:
            groupids = [i[0] for i in groupby_unsorted(
                        zip(self._groupindex, *self.param_dyn_dict.values()),
                        key=itemgetter(0),
                        ).values()]
            param_dyn_df = pd.DataFrame(groupids,
                                        columns=['group_id',
                                                 *self.param_dyn_dict.keys()])
            param_dyn_df.set_index('group_id', inplace=True)
            return param_dyn_df


    @property
    @lru_cache()
    def _groupindex(self):
        '''
        the index used to group the dataset with respect to the temporal
        dynamics of the parameters
        '''

        # find the max. length of the parameters
        maxdict = {key:len(str(max(val))) for key, val in
                   self.param_dyn_dict.items()}
        # find the max. length of the parameters
        def doit(x, N):
            return str(x).zfill(N)
        for i, [key, val] in enumerate(self.param_dyn_dict.items()):
            if i == 0:
                conclist = list(map(partial(doit, N=maxdict[key]), val))
            else:
                conclist = map(add, conclist,
                               map(partial(doit, N=maxdict[key]), val))
        return np.array(list(map(int, conclist)))


    @property
    @lru_cache()
    def _N_groups(self):
        return len(set(self._groupindex))


    @property
    @lru_cache()
    def _dataset_used(self):
        '''
        group the dataset with respect to the required parameter-groups
        '''
        if self.dataset is None:
            return
        # don't use keys that are not provided in defdict
        # (e.g. "param_dyn" keys and additional datasets irrelevant to the fit)
        usekeys = ['sig', 'inc'] + [key for key in
                                    self.defdict if key in self.dataset]
        if 'data_weights' in self.dataset:
            usekeys += ['data_weights']

        # generate new data-frame based on groups
        dataset_used = self.dataset[usekeys].reset_index()
        dataset_used.rename(columns={'index':'orig_index'}, inplace=True)
        dataset_used.set_index(self._groupindex, inplace=True)
        dataset_used = dataset_used.groupby(level=0,
                                            sort=False).agg(pd.Series.tolist)

        return dataset_used


    @property
    @lru_cache()
    def index(self):
        '''
        return the unique index-values of each fit-group
        (used to assign the fit-results)
        '''

        # get all unique values of each group in _orig_index
        return np.concatenate([list(dict.fromkeys(i))
                               for i in self._orig_index])


    @property
    @lru_cache()
    def _jac_assign_rule(self):
        '''
        a dict containing the positions of the derivative-values
        (used in the calculation of the jacobi-determinant to assign
        a scipy.sparse matrix and avoid memory-overflows)
        '''
        shape = self.inc.shape

        jac_rules = dict()
        for key, val in self.param_dyn_df.items():
            uniques = pd.unique(val)

            row_ind = []  # row-indices where jac is nonzero
            col_ind = []  # col-indices where jac is nonzero
            for n_uni, uni in enumerate(uniques):
                rule = (val == uni).values
                #where_n = np.where(np.concatenate(
                #    np.broadcast_to(rule[:,np.newaxis], shape)))[0]
                where_n = [i for i, x in enumerate(
                    chain(*(repeat(i, shape[1]) for i in rule))) if x]

                col_ind += where_n
                row_ind += list(repeat(n_uni, len(where_n)))
            jac_rules[key] = [row_ind, col_ind]
        return jac_rules




    @property
    @lru_cache()
    def meandatetimes(self):
        '''
        a dict of the mean datetime-indexes that correspond to each obtained
        parameter
        '''

        idx = self.dataset.index.to_numpy()
        dates = dict()
        for key, val in self.param_dyn_df.reindex(self._groupindex).items():
            dates[key] = list({g:meandatetime(val_g) for g, val_g in
             groupby_unsorted(zip(idx, val),
                             key=itemgetter(1),
                             get=itemgetter(0)).items()}.values())
        return dates


    @property
    def meandatetimes_group(self):
        '''
        the average datetime-index of each fit-group
        '''
        return [meandatetime(val)for key, val in
                groupby_unsorted(zip(self.dataset.index.to_numpy(),
                                     self._groupindex),
                                 key=itemgetter(1),
                                 get=itemgetter(0)).items()]


    @property
    @lru_cache()
    def inc(self):
        '''
        a rectangular array consisting of the incidence-angles as
        provided in the dataset, rectangularized by repeating the last
        values of each row to fit in length.
        '''

        return self.__get_data(prop='inc')


    @property
    def mask(self):
        '''
        a mask that indicates the artificially added values
        (see 'inc', and 'data' properties for details)
        '''

        return self.__get_data(prop='mask')


    @property
    @lru_cache()
    def data(self):
        '''
        a rectangular array consisting of the data-values as provided
        in the dataset, rectangularized by repeating the last values
        of each row to fit in length
        '''

        return self.__get_data(prop='sig')

    @property
    @lru_cache()
    def data_weights(self):
        '''
        If a column 'data_weights' has been provided in the dataset,
        the residuals in the fit-procedure will be multiplied by the
        values provided as 'data_weights'.
        '''

        if 'data_weights' in self.dataset:
            return self.__get_data(prop='data_weights')
        else:
            return 1.

    @property
    @lru_cache()
    def _order(self):
        '''
        a list of the names of the parameters that will be fitted.
        (this is necessary to ensure correct broadcasting of values since
        dictionarys do)
        '''

        return [i for i, v in self._startvaldict.items() if v is not None]


    @property
    @lru_cache()
    def _orig_index(self):
        '''
        a list of the (grouped) index-values
        '''

        orig_index = [np.array(i, dtype=self.dataset.index.dtype) for i in
             groupby_unsorted(zip(self._groupindex,
                                  self.dataset.index.to_numpy()),
                              key=itemgetter(0), get=itemgetter(1)).values()]

        return orig_index


    @property
    @lru_cache()
    def _idx_assigns(self):
        '''
        a list of ranges that is used to assign values provided as a list
        with length of the dataset to the shape needed for further processing
        (e.g. grouped with respect to the temporal dynamics of the parameters)
        '''
        return rectangularize(groupby_unsorted(range(len(self._groupindex)),
                                        key=lambda x: self._groupindex[x]
                                        ).values())


    @property
    @lru_cache()
    def _param_assigns(self):
        '''
        the indices of each parameter-group (to re-assign to to the fit-index)
        '''
        assigndict = dict()
        for key, val in self.param_dyn_df.items():
            assigndict[key] = groupby_unsorted(range(len(val)),
                                               key=lambda x: val.iloc[x])

        return assigndict


    @property
    @lru_cache()
    def _param_assigns_dataset(self):
        '''
        the indices of each parameter-group (to re-assign to to the fit-index)
        '''
        assigndict = dict()
        for key, val in self.param_dyn_df.reindex(self._groupindex).items():
            assigndict[key] = groupby_unsorted(range(len(val)),
                                               key=lambda x: val.iloc[x])

        return assigndict


    @property
    @lru_cache()
    def _val_assigns(self):
        '''
        the indices to assign the groups to the dataset-index
        '''
        return groupby_unsorted(enumerate(self._groupindex),
                                key=itemgetter(1),
                                get=itemgetter(0))

    @property
    def _setdict(self):
        '''
        Generate RT-1 specifications based on the provided "defdict".
        ... used to simplify the model-specification for 'rtfits.performfit()'
        '''

        # generate RT1 specifications based on defdict
        # initialize empty dicts
        setdict = {}
        # set parameters
        for key, val in self.defdict.items():
            # if parameter is intended to be fitted, assign a sympy-symbol
            if val[0] is True:
                setdict[key] = sp.var(key)

            elif val[0] is False:
                # treat parameters that are intended to be constants
                # if value is provided as a scalar, insert it in the definition
                if isinstance(val[1],
                              (int, float)) and not isinstance(val[1], bool):
                    setdict[key] = val[1]
                else:
                    # if value is provided as array, add it to fixed_dict
                    setdict[key] = sp.var(key)
        return setdict


    @property
    def _fixed_dict(self):
        '''
        a dictionary containing the fixed-values to be used
        '''

        _fixed_dict = {}
        for key, val in self.defdict.items():
            if val[0] is False:
                # treat parameters that are intended to be constants
                # if value is provided as a scalar, insert it in the definition
                if isinstance(val[1], str) and val[1] == 'auxiliary':
                    _fixed_dict[key] = 'auxiliary'
                else:
                    _fixed_dict[key] = val[1]

        return _fixed_dict


    @property
    def fixed_dict(self):
        # update 'fixed_dict' with timeseries provided via 'dataset'-DataFrame
        # ensure that only parameters that are explicitely mentioned in
        # fixed_dict are passed to the fit-procedure as fixed datasets
        # (additional columns of 'dataset' do not affect the fit)
        fixed_dict = dict()
        if isinstance(self.dataset, pd.DataFrame):
            for key, val in self._fixed_dict.items():
                if isinstance(val, str) and val == 'auxiliary':
                    assert key in self._dataset_used, \
                        f"auxiliary data for '{key}' is missing!"
                    fixed_dict[key] = rectangularize(self._dataset_used[key])

        return fixed_dict


    @property
    def _startvaldict(self):
        '''
        a dictionary containing the start-values to be used
        '''
        startvaldict = {}
        for key, val in self.defdict.items():
            if val[0] is True:
                startvaldict[key] = val[1]
        if self.param_dyn_df is not None:
            # re-shape param_dict and bounds_dict to fit needs
            uniques = self.param_dyn_df.nunique()
            for key, val in startvaldict.items():
                # adjust shape of startvalues
                if uniques[key] >= 1 and np.isscalar(val):
                    startvaldict[key] = [val for i in range(uniques[key])]

        return startvaldict


    @property
    def _boundsvaldict(self):
        '''
        a dictionary containing the boundary-values to be used
        '''
        boundsvaldict = {}
        for key, val in self.defdict.items():
            if val[0] is True:
                boundsvaldict[key] = val[3]
        if self.param_dyn_df is not None:
            # re-shape param_dict and bounds_dict to fit needs
            uniques = self.param_dyn_df.nunique()
            for key, val in boundsvaldict.items():
                # adjust shape of boundary conditions
                if uniques[key] >= 1 and len(val[0]) == 1:
                    boundsvaldict[key] = (val[0]*uniques[key], val[1]*uniques[key])

        return boundsvaldict


    @property
    def _timescaledict(self):
        '''
        a dictionary containing the pandas datetime offset strings to use
        '''

        timescaledict = {}
        for key, val in self.defdict.items():
            if (val[0] is True and val[2] is not None
                and val[2] != 'manual' and val[2] != 'index'):
                    timescaledict[key] = self.defdict[key][2]
        return timescaledict


    @property
    def _manual_dyn_df(self):
        '''
        a dictionary containing possible additional temporal dynamics
        (e.g. in addition to _timescaledict)
        '''

        manual_dyn_df = pd.DataFrame()
        for key, val in self.defdict.items():
            if val[0] is True:
                if val[2] == 'manual':
                    # set manual parameter dynamics
                    assert f'{key}_dyn' in self.dataset, (
                        f'{key}_dyn must be provided in the dataset' +
                        'if defdict[{key}][2] is set to "manual"')

                    manual_dyn_df[f'{key}'] = self.dataset[f'{key}_dyn']
                elif val[2] == 'index':
                    indexdyn = pd.DataFrame({key:1}, self.dataset.index
                                            ).groupby(axis=0, level=0,
                                                      sort=False).ngroup()
                    indexdyn.name = key
                    manual_dyn_df[f'{key}'] = indexdyn

                else:
                    if f'{key}_dyn' in self.dataset:
                        print(f'parameter dynamics ({val[2]}) and ' +
                              f'"dataset[{key}_dyn]" combined')
                        manual_dyn_df[f'{key}'] = self.dataset[f'{key}_dyn']
        if manual_dyn_df.empty:
            return None
        else:
            return manual_dyn_df


    @property
    def _param_R_dict(self):
        '''
        a dict containing the parameter-values that are needed to evaluate
        the fn-coefficients (as needed for evaluation of the interaction-term)
        '''

        param_R = dict(**self._startvaldict, **self.fixed_dict)
        param_R.pop('omega', None)
        param_R.pop('tau', None)
        param_R.pop('NormBRDF', None)
        param_R.pop('bsf', None)

        toNlist = set(self._tau_symb + self._omega_symb + self._N_symb)

        # check of general input-requirements
        # check if all parameters have been provided
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, self.V._func.free_symbols)) - angset
        srfsymb = set(map(str, self.SRF._func.free_symbols)) - angset

        paramset = ((set(map(str, self._startvaldict.keys()))
                     ^ set(map(str, self.fixed_dict.keys())))
                    - {'tau', 'omega', 'NormBRDF'})

        assert paramset >= (vsymb | srfsymb), (
            'the parameters ' +
            str((vsymb | srfsymb) - paramset) +
            ' must be provided in param_dict')

        # remove also other symbols that are used in the definitions of
        # tau, omega and NormBRDF
        for i in set(toNlist - vsymb - srfsymb):
            if i in param_R:
                param_R.pop(i)

        return param_R


    @property
    def V(self):
        '''
        new initialization of the rt1.volume object used
        '''

        # set V and SRF based on setter-function
        if callable(self.set_V_SRF):
            V, _ = self.set_V_SRF(**self._setdict)
        elif isinstance(self.set_V_SRF, dict):
            V = self._init_V_SRF(**self.set_V_SRF,
                                 setdict=self._setdict,
                                 V_SRF_Q='V')

        return V


    @property
    def SRF(self):
        '''
        new initialization of the rt1.surface object used
        '''

        # set V and SRF based on setter-function
        if callable(self.set_V_SRF):
            _, SRF = self.set_V_SRF(**self._setdict)
        elif isinstance(self.set_V_SRF, dict):
            SRF = self._init_V_SRF(**self.set_V_SRF,
                                   setdict=self._setdict,
                                   V_SRF_Q='SRF')
        return SRF


    @property
    def R(self):
        '''
        new initialization of the rt1.RT1 object used
        '''

        R = RT1(1., self.inc, self.inc,
                np.zeros_like(self.inc), np.full_like(self.inc, np.pi),
                V=self.V, SRF=self.SRF,
                fn_input=None, _fnevals_input=None,
                geometry='mono', bsf = self._setdict.get('bsf', 0.),
                int_Q=self.int_Q,
                lambda_backend=self.lambda_backend,
                param_dict=self._param_R_dict,
                verbosity=self.verbose)

        if self.int_Q is True and self._fnevals_input is None:
            self._fnevals_input = R._fnevals
        elif self.int_Q is True:
            # store _fnevals functions in case they have not been provided
            # as input-arguments explicitly to avoid recalculation for each step
            R._fnevals_input = self._fnevals_input
            # set fn_input to any value except None to avoid re-calculation
            R.fn_input = 1

        return R


    def __get_V_SRF_symbs(self, V_SRF, prop):
        '''
        the symbols used to define tau, omega and NormBRDF of V and SRF
        '''
        try:
            symbs = list(map(str,
                             getattr(getattr(self, V_SRF), prop).free_symbols))
        except Exception:
            symbs = list()
        return symbs


    def __get_V_SRF_funcs(self, V_SRF, prop):
        '''
        the lambdified functions used to define tau, omega and NormBRDF
        of V and SRF
        '''
        try:
            func = sp.lambdify(
                getattr(getattr(self, V_SRF), prop).free_symbols,
                getattr(getattr(self, V_SRF), prop),
                modules=['numpy'])
        except Exception:
            func = None
        return func


    def __get_V_SRF_diff_funcs(self, V_SRF, prop):
        '''
        a dict containing the lambdified partial derivatives of the functions
        used to define tau, omega and NormBRDF of V and SRF
        '''

        d_inner = dict()
        for param in (self.__get_V_SRF_symbs(V_SRF, prop) &
                  self.param_dyn_dict.keys()):
            d_inner[param] = sp.lambdify(
                self.__get_V_SRF_symbs(V_SRF, prop),
                sp.diff(getattr(getattr(self, V_SRF), prop),
                        sp.Symbol(param)),
                modules=['numpy'])
        return d_inner


    @property
    @lru_cache()
    def _tau_symb(self):
        return self.__get_V_SRF_symbs('V', 'tau')
    @property
    @lru_cache()
    def _tau_func(self):
        return self.__get_V_SRF_funcs('V', 'tau')
    @property
    @lru_cache()
    def _tau_diff_func(self):
        return self.__get_V_SRF_diff_funcs('V', 'tau')


    @property
    @lru_cache()
    def _omega_symb(self):
        return self.__get_V_SRF_symbs('V', 'omega')
    @property
    @lru_cache()
    def _omega_func(self):
        return self.__get_V_SRF_funcs('V', 'omega')
    @property
    @lru_cache()
    def _omega_diff_func(self):
        return self.__get_V_SRF_diff_funcs('V', 'omega')

    @property
    @lru_cache()
    def _N_symb(self):
        return self.__get_V_SRF_symbs('SRF', 'NormBRDF')
    @property
    @lru_cache()
    def _N_func(self):
        return self.__get_V_SRF_funcs('SRF', 'NormBRDF')
    @property
    @lru_cache()
    def _N_diff_func(self):
        return self.__get_V_SRF_diff_funcs('SRF', 'NormBRDF')


    @property
    def res_df(self):
        '''
        return a pandas DataFrame with the obtained parameters from the fit
        (performfit must be called prior to accessing this property!)
        '''
        if not hasattr(self, 'res_dict'):
            print('you must perform the fit first! ...e.g. call performfit()')
            return

        vals = self._assignvals(self.res_dict)
        for key, val in self._assignvals(self.res_dict).items():
            if key in self.interp_vals:
                # take care of interpolation-values
                # (they are provided in the shape of fit.data)
                vals[key] = vals[key][~self.mask]
            else:
                x = np.empty(len(self._groupindex), dtype=float)
                [x.put(ind, val_i) for val_i, ind in
                 zip(vals[key], self._val_assigns.values())]
                vals[key] = x

        resdf = pd.DataFrame(vals, self.dataset.index).groupby(level=0).first()

        return resdf


    @property
    def res_df_group(self):
        '''
        return a pandas DataFrame with the obtained parameters from the fit
        for each unique fit-group. The index is the mean-datetime index
        of the corresponding group.
        (performfit must be called prior to accessing this property!)
        '''
        if not hasattr(self, 'res_dict'):
            print('you must perform the fit first! ...e.g. call performfit()')
            return

        series = []
        for key, val in self.res_dict.items():
            x = np.empty(len(self.meandatetimes_group), dtype=float)
            [x.put(ind, val_i) for val_i, ind in
              zip(val, self._param_assigns[key].values())]
            series.append(pd.Series(x,
                                    self.meandatetimes_group,
                                    name=key))
        resdf = pd.concat(series, axis=1)
        return resdf



    def _assignvals(self, res_dict, interp_vals=None):
        '''
        a function to distribute the fit-values to the actual shape of the
        dataset. This is done in one of the following ways:

            - if the parameter is provided in the "interp_vals" list,
              a quadratic interpolation (with respect to the datetime-indices)
              will be used to assign the values
            - otherwise, a step-function will be used

        Parameters:
        -------------
        res_dict : dict
            a dict containing the results obtained from a run of performfit()

        '''
        if interp_vals is None:
            interp_vals = self.interp_vals

        if len(interp_vals) > 0:
            firstindex = self.dataset.index[0]
            lastindex = self.dataset.index[-1]

            # get the results and a quadratic interpolation-function
            use_res_dict = dict()
            for key, val in res_dict.items():
                if key in interp_vals:
                    # generate an interpolaion function
                    # the values at the boundaries are set to the nearest
                    # obtained values to avoid extrapolation

                    # ensure that the indexes are never the same
                    # (add a milisecond if they are...)
                    if firstindex == self.meandatetimes[key][0]:
                        firstindex += np.timedelta64(1, 'ms')
                    if lastindex == self.meandatetimes[key][-1]:
                        lastindex -= np.timedelta64(1, 'ms')

                    useindex = np.array([firstindex,
                                         *self.meandatetimes[key],
                                         lastindex],
                                        dtype='datetime64[ns]'
                                        ).astype(float, copy=False)

                    if len(val) >= 2:
                        usevals = np.take(val, [0, *range(len(val)), -1])

                        # interpolate the data to the used timestamps
                        f = interp1d(useindex.astype(float), usevals,
                                     fill_value='extrapolate', axis=0,
                                     kind='quadratic')

                        x = f(np.array(self.dataset.index,
                                                dtype='datetime64[ns]'))
                        # assign correct shape
                        use_res_dict[key] = np.take(x, self._idx_assigns)
                    else:
                        print('warning, interpolation not possible for ({key})'
                              'because there are less than 2 values')

                        x = np.empty(len(self.dataset), dtype=float)
                        [x.put(ind, val_i) for val_i, ind in
                              zip(val,
                                  self._param_assigns_dataset[key].values())]
                        # assign correct shape
                        use_res_dict[key] = np.take(x, self._idx_assigns)

                else:
                    x = np.empty(self._N_groups, dtype=float)
                    [x.put(ind, val_i) for val_i, ind in
                     zip(val, self._param_assigns[key].values())]
                    use_res_dict[key] = x[:,np.newaxis]
        else:
            use_res_dict = dict()
            for key, val in res_dict.items():
                x = np.empty(self._N_groups, dtype=float)
                [x.put(ind, val_i) for val_i, ind in
                  zip(val, self._param_assigns[key].values())]
                use_res_dict[key] = x[:,np.newaxis]

        return use_res_dict


    def _calc_model(self, R=None, res_dict=None, fixed_dict=None,
                    interp_vals=None, return_components=False,
                    assign=True):
        '''
        function to calculate the model-results based on the provided
        parameters

        Parameters:
        ------------
        R: RT1-object
           the rt1-object for which the results shall be calculated
        res_dict: dict
                  a dictionary containing all dynamic parameter-values
                  if None, `self.res_dict` is used
        fixed_dict: dict
                    a dictionary containing all fixed-values
                    if None, `self.fixed_dict` is used
        interp_vals: list
                     a list of parameter-names whose fitted values should
                     be interpolated. if None, `self.interp_vals` is used
        return_components: bool (default=False)
                           indicator if the individual components or only
                           the total backscattered radiation are returned
        assign: bool (default=True)
                indicator if the provided values should be assigned based
                on `fit.param_dyn_df` bevore evaluating the model.
                If `True`, the parameters must be provided in the
                same shape as `fit.res_dict`. For arbitrary number of
                parameters set to `False` or the dedicated method `fit.calc()`
        Returns:
        ----------
        model_calc: the evaluated backscatter (as intensity or sigma_0)
                    in linear-units or dB corresponding to the specifications
                    defined in the rtfits-class.
        '''

        if R is None:
            R = self.R
        if res_dict is None:
            res_dict = self.res_dict
        if fixed_dict is None:
            fixed_dict = self.fixed_dict
        if interp_vals is None:
            interp_vals = self.interp_vals

        # ensure correct array-processing
        # res_dict = {key:val[:,np.newaxis] for
        #             key, val in self._assignvals(res_dict).items()}
        if assign is True:
            res_dict = self._assignvals(res_dict, interp_vals)
        res_dict.update(fixed_dict)

        # update the numeric representations of omega, tau and NormBRDF
        # based on the values for the used symbols provided in res_dict
        if self._omega_func is None:
            if 'omega' in res_dict:
                R.V.omega = res_dict['omega']
        else:
            R.V.omega = self._omega_func(
                **{key:res_dict[key] for key in self._omega_symb})

        if self._tau_func is None:
            if 'tau' in res_dict:
                R.V.tau = res_dict['tau']
        else:
            R.V.tau = self._tau_func(
                **{key:res_dict[key] for key in self._tau_symb})

        if self._N_func is None:
            if 'NormBRDF' in res_dict:
                R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = self._N_func(
                **{key:res_dict[key] for key in self._N_symb})

        if 'bsf' in res_dict:
            R.bsf = res_dict['bsf']

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input. (i.e. "omega", "tau", "NormBRDF"
        # and the symbols used to define them must be removed)

        # symbols used to define the functions
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, self.V._func.free_symbols)) - angset
        srfsymb = set(map(str, self.SRF._func.free_symbols)) - angset

        # a list of all symbols used to define tau, omega and NormBRDF
        toNlist = set(self._tau_symb + self._omega_symb + self._N_symb)

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

        return model_calc


    def _calc_jac(self, R=None, res_dict=None, fixed_dict=None,
                  param_dyn_dict=None, order=None):
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

        if R is None:
            R = self.R
        if res_dict is None:
            res_dict = self.res_dict
        if fixed_dict is None:
            fixed_dict = self.fixed_dict
        if param_dyn_dict is None:
            param_dyn_dict = self.param_dyn_df
        if order is None:
            order = self._order

        # ensure correct array-processing
        # res_dict = {key:val[:,np.newaxis] for
        #             key, val in self._assignvals(res_dict).items()}
        res_dict = self._assignvals(res_dict)
        res_dict.update(fixed_dict)


        # update the numeric representations of omega, tau and NormBRDF
        # based on the values for the used symbols provided in res_dict
        if self._omega_func is None:
            if 'omega' in res_dict:
                R.V.omega = res_dict['omega']
        else:
            R.V.omega = self._omega_func(
                **{key:res_dict[key] for key in self._omega_symb})

        if self._tau_func is None:
            if 'tau' in res_dict:
                R.V.tau = res_dict['tau']
        else:
            R.V.tau = self._tau_func(
                **{key:res_dict[key] for key in self._tau_symb})

        if self._N_func is None:
            if 'NormBRDF' in res_dict:
                R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = self._N_func(
                **{key:res_dict[key] for key in self._N_symb})

        if 'bsf' in res_dict:
            R.bsf = res_dict['bsf']

        # remove all unwanted symbols that are NOT needed for evaluation
        # of the fn-coefficients from res_dict to generate a dict that
        # can be used as R.param_dict input (i.e. "omega", "tau", "NormBRDF",
        # "bsf" and the symbols used to define them must be removed)

        # symbols used in the definitions of the functions
        angset = {'phi_ex', 'phi_0', 'theta_0', 'theta_ex'}
        vsymb = set(map(str, self.V._func.free_symbols)) - angset
        srfsymb = set(map(str, self.SRF._func.free_symbols)) - angset

        # a list of all symbols used to define tau, omega and NormBRDF
        toNlist = set(self._tau_symb + self._omega_symb + self._N_symb)

        # exclude all keys that are not needed to calculate the fn-coefficients
        # vsymb and srfsymb must be subtracted in case the same symbol is used
        # for omega, tau or NormBRDF definition and in the function definiton
        excludekeys = ['omega', 'tau', 'NormBRDF', 'bsf',
                       *[str(i) for i in set(toNlist - vsymb - srfsymb)]]

        strparam_fn = {str(key) : val for key, val in res_dict.items()
                       if key not in excludekeys}

        # set the param-dict to the newly generated dict
        R.param_dict = strparam_fn

        # if tau, omega or NormBRDF have been provided in terms of symbols,
        # remove the symbols that are intended to be fitted (that are also
        # in param_dyn_dict) and replace them by 'omega', 'tau' and 'NormBRDF'
        # so that calling R.jacobian will calculate the "outer" derivative
        neworder = [o for o in order]
        if len(self._tau_symb) != 0:
            for i in set(self._tau_symb) & set(param_dyn_dict.keys()):
                neworder[neworder.index(i)] = 'tau'
        if len(self._omega_symb) != 0:
            for i in set(self._omega_symb) & set(param_dyn_dict.keys()):
                neworder[neworder.index(i)] = 'omega'
        if len(self._N_symb) != 0:
            for i in set(self._N_symb) & set(param_dyn_dict.keys()):
                neworder[neworder.index(i)] = 'NormBRDF'

        # calculate the jacobian based on neworder
        # (evaluating only "outer" derivatives with respect to omega,
        # tau and NormBRDF)
        jac = R.jacobian(sig0=self.sig0, dB=self.dB, param_list=neworder)
        #self.jacshape = jac[0].shape

        # generate a scipy.sparse matrix that represents the jacobian for all
        # the individual parameters according to jac_dyn_dict
        # (this is needed to avoid memory overflows for very large jacobians)
        jac_size = jac[0].size
        newjacdict = {}
        for i, key in enumerate(order):
            uniques = pd.unique(param_dyn_dict[key])
            # provide unique values based on original occurence

            if len(uniques) == 1:
                my_array = np.empty(jac_size)
                #np.array([np.concatenate(jac[i], axis=0)])
                newjacdict[key] = np.expand_dims(
                    np.fromiter(chain(*jac[i]), dtype=float, count=jac_size),0)
            else:
                # if too many unique values occur, use scipy sparse matrix
                # to avoid memory-overflow due to the large number of zeroes...
                # (this will reduce speed since scipy.sparse does not fully
                # supprot BLAS and so no proper parallelization is performed)
                #data = np.concatenate(jac[i])
                data = np.fromiter(chain(*jac[i]), dtype=float, count=jac_size)
                row_ind, col_ind = self._jac_assign_rule[key]
                # generate a sparse matrix
                m = csr_matrix((data, (row_ind, col_ind)),
                               shape=(max(row_ind) + 1,
                                      max(col_ind) + 1))
                newjacdict[key] = m

        # evaluate jacobians of the functional representations of tau
        # and add them to newjacdict
        for i in set(self._tau_symb) & set(param_dyn_dict.keys()):
            # generate a function that evaluates the 'inner' derivative, i.e.:
            # df/dx = df/dtau * dtau/dx = df/dtau * d_inner
            d_inner = self._tau_diff_func[i]
            # evaluate the inner derivative
            dtau_dx = d_inner(**{key:res_dict[key]
                                   for key in self._tau_symb})
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
        for i in set(self._omega_symb) & set(param_dyn_dict.keys()):
            d_inner = self._omega_diff_func[i]
            domega_dx = d_inner(**{key:res_dict[key]
                                   for key in self._omega_symb})

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
        for i in set(self._N_symb) & set(param_dyn_dict.keys()):
            d_inner = self._N_diff_func[i]
            dN_dx = d_inner(**{key:res_dict[key] for key in self._N_symb})

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

        sparse = False
        for key, val in newjacdict.items():
            if isspmatrix(val):
                newjacdict[key] = val.tocsc()[:,np.ravel(~self.mask)]
                sparse = True
            else:
                newjacdict[key] = val[:,np.ravel(~self.mask)]

        # return the transposed jacobian as needed by scipy's least_squares
        if sparse:#np.any([isspmatrix(newjacdict[key]) for key in order]):
            # in case sparse matrices have been used, use scipy to vstack them
            jac_lsq = vstack([newjacdict[key] for key in order]).transpose()
        else:
            jac_lsq = np.vstack([newjacdict[key] for key in order]).transpose()

        return jac_lsq


    def __get_data(self, prop):
        '''
        a function to retrieve properties from the provided dataset

        Parameters
        ----------
        prop : str
            the name of the property.

        Returns
        -------
        np.array
            the values of the requested property

        '''
        if isinstance(self._dataset_used, pd.DataFrame):
            if prop in ['inc', 'sig', 'data_weights']:
                return rectangularize(self._dataset_used[prop].values)
            elif prop == 'mask':
                _, mask = rectangularize(self._dataset_used.inc.values,
                                         return_mask=True)
                if prop == 'mask':
                    return mask
        elif isinstance(self._dataset_used, list):
            if prop == 'inc':
                return rectangularize([i[0] for i in self._dataset_used])
            elif prop == 'sig':
                return rectangularize([i[1] for i in self._dataset_used])
            elif prop == 'mask':
                _, mask = rectangularize([i[0] for i in self._dataset_used]
                                         , return_mask=True)
                if prop == 'mask':
                    return mask


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
        res_dict = self._assignvals(res_dict)
        res_dict.update(fixed_dict)


        # update the numeric representations of omega, tau and NormBRDF
        # based on the values for the used symbols provided in res_dict
        if self._omega_func is None:
            if 'omega' in res_dict:
                R.V.omega = res_dict['omega']
        else:
            R.V.omega = self._omega_func(
                **{key:res_dict[key] for key in self._omega_symb})

        if self._tau_func is None:
            if 'tau' in res_dict:
                R.V.tau = res_dict['tau']
        else:
            R.V.tau = self._tau_func(
                **{key:res_dict[key] for key in self._tau_symb})

        if self._N_func is None:
            if 'NormBRDF' in res_dict:
                R.SRF.NormBRDF = res_dict['NormBRDF']
        else:
            R.SRF.NormBRDF = self._N_func(
                **{key:res_dict[key] for key in self._N_symb})

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

        # a list of all symbols used to define tau, omega and NormBRDF
        toNlist = set(self._tau_symb + self._omega_symb + self._N_symb)

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

        return {'slope' : model_slope,
                'curv' : model_curv}


    def _init_V_SRF(self, V_props, SRF_props, setdict=dict(),
                    V_SRF_Q='V'):
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

        if V_SRF_Q == 'V': props = V_props
        elif V_SRF_Q == 'SRF': props = SRF_props

        set_dict = dict()
        for key, val in props.items():
            if key == 'V_name' or key == 'SRF_name': continue

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

            set_dict[key] = useval

        if V_SRF_Q == 'V':
            # initialize the volume-scattering function
            V = getattr(rt1_v, V_props['V_name'])(**set_dict)
            return V

        elif V_SRF_Q == 'SRF':
            # initialize the surface-scattering function
            SRF = getattr(rt1_s, SRF_props['SRF_name'])(**set_dict)
            return SRF

    def performfit(self, clear_cache=True, intermediate_results=False,
                   print_progress=False):
        '''
        Perform least-squares fitting of omega, tau, NormBRDF and any
        parameter used to define V and SRF to sets of monostatic measurements.


        Parameters
        ----------
        clear_cache : bool, optional
            indicator if the cache should be cleared prior to performing
            the fit. Disable this only if you know exactly what you're doing!.
            The default is True.
        intermediate_results : bool, optional
            indicator if intermedite results should be stored or not.
            This might generate a lot of additional output and is only
            required for the plotfunction `fit.plot.intermediate_results()`
            The default is False.
        print_progress : bool, optional
            indicator if a progress-bar should be printed to stdout or not.
            The default is False.
        '''

        # clear the cache (to avoid issues in case re-processing is applied)
        if clear_cache is True:
            self._clear_cache()
        # maintain R object during fit
        R = self.R

        if clear_cache is True:
            R._clear_cache()

        # set the number of repetitions (e.g. the max. number of values
        # encountered in a group)

        # set up the dictionary for storing intermediate results
        if intermediate_results is True:
            self.intermediate_results = {'parameters':[],
                                         'residuals':[],
                                         'jacobian':[]}

        # will be used to assign the values to the individual parameters
        # (the returned parameters are given as a concatenated array
        # of the shape [*p0, *p1, *p2, ...]) where p1,p2,p3 are a list of
        # values for each parameter
        splitpos = self.param_dyn_df.nunique()[self._order].to_list()

        if print_progress:
            update_cnt = count(1)
            if 'max_nfev' in self.lsq_kwargs:
                max_cnt = self.lsq_kwargs['max_nfev']
            else:
                max_cnt = 1000

        # define a function that evaluates the model in the shape as needed
        # for scipy's least_squares function
        def fun(params):
            if print_progress:
                update_progress(next(update_cnt), max_cnt,
                                title="function evaluations: ",
                                finalmsg=f"max_nfev ({max_cnt}) reached!")

            # generate a dictionary to assign values based on input
            split_vals = split_into(params, splitpos)
            newdict = dict(zip(self._order, split_vals))
            # calculate the residuals and incorporate data-weighting
            errs = (self.data_weights *
                    (self._calc_model(R=R, res_dict=newdict) - self.data)
                    )[~self.mask]

            if intermediate_results is True:
                self.intermediate_results['parameters'] += [newdict]
                errdict = {'abserr' : errs,
                           'relerr' : errs/self.data[~self.mask]}
                self.intermediate_results['residuals'] += [errdict]

            return errs


        # function to evaluate the jacobian
        def dfun(params):
            # generate a dictionary to assign values based on input
            split_vals = split_into(params, splitpos)
            newdict = dict(zip(self._order, split_vals))

            # calculate the jacobian
            jac = self._calc_jac(R=R, res_dict=newdict)

            return jac


        # generate list of boundary conditions as needed for the fit
        bounds = [[], []]
        for key in self._order:
            bounds[0] = bounds[0] + list(self._boundsvaldict[key][0])
            bounds[1] = bounds[1] + list(self._boundsvaldict[key][1])

        # setup the start-value array as needed for the fit
        startvals = []
        for key in self._order:
            if self._startvaldict[key] is not None:
                if np.isscalar(self._startvaldict[key]):
                    startvals = startvals + [self._startvaldict[key]]
                else:
                    startvals = startvals + list(self._startvaldict[key])

        # perform the actual fit
        res_lsq = least_squares(fun, startvals, bounds=bounds,
                                jac=dfun, **self.lsq_kwargs)

        # generate a dictionary to assign values based on fit-results
        # split the obtained result with respect to the individual parameters
        if res_lsq is not None:
            # split concatenated parameter-results given in res_lsq.x
            # (don't use the last array since it is empty)

            split_vals = split_into(res_lsq.x, splitpos)
            res_dict = dict(zip(self._order, split_vals))
            self.fit_output = res_lsq
            self.res_dict = res_dict
        else:
            self.res_dict = dict()
            self.fit_output = None

        # clear the cache (to avoid issues in case re-processing is applied)
        if clear_cache is True:
            self._clear_cache()
            R._clear_cache()

        if print_progress:
            if next(update_cnt) - 1 < max_cnt:
                update_progress(max_cnt, max_cnt,
                                finalmsg=f'Done! ({res_lsq.message})')



    def _evalfunc(self, reader=None, reader_arg=None, lsq_kwargs=None,
                  preprocess=None, postprocess=None, exceptfunc=None,
                  process_cnt=None):
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
        lsq_kwargs : dict, optional
            override the lsq_kwargs-dict of the parent Fits-object.
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
        if process_cnt is not None:
            start = tick()
        try:
            # call preprocess function if provided
            if callable(preprocess):
                preprocess(reader_arg)

            if lsq_kwargs is None:
                lsq_kwargs = self.lsq_kwargs

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
            # perform the fit
            fit = Fits(sig0=self.sig0, dB=self.dB, dataset = dataset,
                       defdict=self.defdict, set_V_SRF=self.set_V_SRF,
                       lsq_kwargs=lsq_kwargs, int_Q=self.int_Q,
                       lambda_backend=self.lambda_backend,
                       _fnevals_input=self._fnevals_input,
                       interp_vals=self.interp_vals)
            fit.performfit()

            # append auxiliary data
            if aux_data is not None:
                fit.aux_data = aux_data

            # if a post-processing function is provided, return its output,
            # else return the fit-object directly
            if callable(postprocess):
                ret = postprocess(fit, reader_arg)
            else:
                ret = fit

            if process_cnt is not None:
                p_totcnt, p_meancnt, p_max, p_time, p_ncpu = process_cnt
                end = tick()
                # increase the total counter
                p_totcnt.value += 1

                # update the estimate of the mean time needed to process a site
                p_time.value = (p_meancnt.value * p_time.value
                                + (end - start)) / (p_meancnt.value + 1)
                # increase the mean counter
                p_meancnt.value += 1
                # get the remaining time and update the progressbar
                remain = timedelta(
                    seconds = (p_max - p_totcnt.value) / p_ncpu * p_time.value)
                d,h,m,s = dt_to_hms(remain)
                update_progress(
                    p_totcnt.value, p_max,
                    title=f"approx. {d} {h:02}:{m:02}:{s:02} remaining",
                    finalmsg=f"finished! ({p_max} [{p_meancnt.value}] fits)",
                    progress2=p_totcnt.value - p_meancnt.value)

            return ret

        except Exception as ex:
            if process_cnt is not None:
                p_totcnt, p_meancnt, p_max, p_time, p_ncpu = process_cnt
                # only increase the total counter
                p_totcnt.value += 1
                if p_meancnt.value == 0:
                    title=f"{'estimating time ...':<28}"
                else:
                    # get the remaining time and update the progressbar
                    remain = timedelta(
                        seconds = (p_max - p_totcnt.value
                                   ) / p_ncpu * p_time.value)
                    d,h,m,s = dt_to_hms(remain)
                    title=f"approx. {d} {h:02}:{m:02}:{s:02} remaining"

                update_progress(
                    p_totcnt.value, p_max,
                    title=title,
                    finalmsg=f"finished! ({p_max} [{p_meancnt.value}] fits)",
                    progress2=p_totcnt.value - p_meancnt.value)

            if callable(exceptfunc):
                return exceptfunc(ex, reader_arg)
            else:
                raise ex


    def processfunc(self, ncpu=1, reader=None, reader_args=None,
                    lsq_kwargs=None, preprocess=None, postprocess=None,
                    exceptfunc=None, finaloutput=None, pool_kwargs=None,
                    print_progress=True):
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

            >>> from specification_file import fit reader lsq_kwargs ... ...
                if __name__ == '__main__':
                    fit.processfunc(ncpu=5, reader=reader,
                                    lsq_kwargs=lsq_kwargs, ... ...)

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
        lsq_kwargs : dict, optional
            override the lsq_kwargs-dict of the parent Fits-object.
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
        print_progress : bool
            indicator if a progress-bar should be printed to stdout or not
            that looks like this:

            >>> approx. 0 00:00:02 remaining ################------ 3 (2) / 4
            >>>
            >>> (estimated time day HH:MM:SS)(     progress bar   )( counts )
            >>> ( counts ) = finished fits [actually fitted] / total

        Returns
        -------
        res : list
            A list of rt1.rtfits.Fits objects or a list of outputs of the
            postprocess-function.

        """

        if lsq_kwargs is None: lsq_kwargs = self.lsq_kwargs
        if pool_kwargs is None: pool_kwargs = dict()

        if self.int_Q is True:
            # pre-evaluate the fn-coefficients if interaction terms are used
            self._fnevals_input = self.R._fnevals

        if print_progress is True:
            # initialize shared values that will be used to track the number
            # of completed processes and the mean time to complete a process
            manager = mp.Manager()
            p_totcnt = manager.Value(ctypes.c_ulonglong, 0)
            p_meancnt = manager.Value(ctypes.c_ulonglong, 0)
            p_time = manager.Value(ctypes.c_float, 0)
            process_cnt = [p_totcnt, p_meancnt, len(reader_args),
                           p_time, ncpu]
        else:
            process_cnt = None

        if ncpu > 1:
            print('start of parallel evaluation')
            with mp.Pool(ncpu, **pool_kwargs) as pool:
                # loop over the reader_args
                res_async = pool.starmap_async(self._evalfunc,
                                               zip(repeat(reader),
                                                   reader_args,
                                                   repeat(lsq_kwargs),
                                                   repeat(preprocess),
                                                   repeat(postprocess),
                                                   repeat(exceptfunc),
                                                   repeat(process_cnt)))
                pool.close()  # Marks the pool as closed.
                pool.join()   # Waits for workers to exit.
                res = res_async.get()
        else:
            print('start of single-core evaluation')
            res = []
            for reader_arg in reader_args:
                res.append(self._evalfunc(reader=reader,
                                          reader_arg=reader_arg,
                                          lsq_kwargs=lsq_kwargs,
                                          preprocess=preprocess,
                                          postprocess=postprocess,
                                          exceptfunc=exceptfunc,
                                          process_cnt=process_cnt))

        if callable(finaloutput):
            return finaloutput(res)
        else:
            return res


    def dump(self, path, mini=True):
        '''
        Save the rt1.rtfits.Fits object using cloudpickle.dump()

        The generated file can be loaded via:

        >>> import cloudpickle
            with open(--path-to-file--, 'rb') as file
                fit = cloudpickle.load(file)

        In order to avoid platform and environment-specific issues, the
        "mini=True" option removes any pre-evaluated "fit._fnevals_input"
        functions (since symengine LLVM lambdas are platform-specific)
        as well as "fit.fit_output" (it contains the residuals, the final
        jacobian etc. which might take up a lot of space)


        Parameters
        ----------
        path : str
            The path to the file that should be created.
        mini : bool, optional
            Indicator if unnecessary attributes should be removed before
            pickling or not (to avoid storing duplicated data).

            The default is True.
        '''
        # add version number to dump
        self._RT1_version = _RT1_version

        if mini is True:
            self._rt1_dump_mini = True
        else:
            try:
                delattr(self, '_rt1_dump_mini')
            except AttributeError:
                pass

        with open(path, 'wb') as file:
            cloudpickle.dump(self, file)


    def calc(self, param, inc, return_components=True,
             fixed_param = None):
        '''
        evaluate the model with respect to a given set of parameters
        and incidence-angles.

        Parameters
        ----------
        param : dict
            a dictionary with the parameter-values to be used, provided in
            the shape

            >>> params = dict(p1 = [1,2], p2 = [3,4], ...)

        inc : array-like
            a 1d numpy-array of the incidence-angles (in radians) to use

        return_components : bool, optional
            indicator if only the total backscatter or also the individual
            backscatter-contributions should be returned.
            The default is True.
        fixed_param : dict, optional
            a dictionary in the same shape as `params` that will be used to
            assign the values of parameters that are provided as 'auxiliary'
            datasets. The default is `None`.

        Returns
        -------
        res : array-like
            the calculated backscatter with respect to the parameters
            if return_components is True, a set of arrays is return
            corresponding to the backscatter-contributions:

            >>> res = [total, surface, volume, (interaction)]

            where the interaction-contribution is only returned if the `int_Q`
            parameter is set to `True`.

            if return_components is False, only the total-contribution is
            returned
        '''

        R = self.R
        R.t_0 = np.atleast_2d(inc)
        R.p_0 = np.full_like(R.t_0, 0.)
        if isinstance(param,
                      pd.DataFrame) and (fixed_param is None
                                         or isinstance(fixed_param,
                                                       pd.DataFrame)):
            res_dict = {key:np.atleast_1d(val.values)[:,np.newaxis]
                        for key, val in param.items()}
            if fixed_param is None:
                fixed_param = dict()
            else:
                fixed_param = {key:np.atleast_1d(val)[:,np.newaxis] for
                               key, val in fixed_param.loc[param.index].items()}

        else:
            res_dict = {key:np.atleast_1d(val)[:,np.newaxis]
                        for key, val in param.items()}

            if fixed_param is None:
                fixed_param = dict()
            else:
                fixed_param = {key:np.atleast_1d(val)[:,np.newaxis] for
                               key, val in fixed_param.items()}

        res = self._calc_model(R=R, res_dict=res_dict, fixed_dict=fixed_param,
                               interp_vals=[],
                               return_components=return_components,
                               assign=False)

        return res

    from functools import wraps
    @wraps(_calc_model)
    def calc_model(self, *args, **kwargs):
        # check if components have been calculated

        # get the index of each fit-value
        # use the concatenated orig_index in case unsorted grouping is used!
        # (only different from fit.dataset.index in case a unsorted grouping
        # has been used)
        calcindex = np.concatenate(self._orig_index)

        res = self._calc_model(*args, **kwargs)
        if len(res.shape) == 3:
            return pd.DataFrame(
                dict(zip(('tot','surf','vol','inter'),
                         [i[~self.mask] for i in res])),
                index=calcindex)#.groupby(level=0).mean()
        else:
            return pd.DataFrame(
                dict(tot=res[~self.mask]),
                index= calcindex)#.groupby(level=0).mean()


    @property
    def model_definition(self):

        fitted, auxiliary, fixed = '', [], []

        fitted += ('     NAME'.ljust(14) + '|     START'.ljust(15) +
                   '|  VARIABILITY'.ljust(16) + '|    BOUNDS'.ljust(15) +
                   '| INTERPOLATION'.ljust(15)) + ' |'
        fitted += '\n'

        for key, val in self.defdict.items():
            if val[0] is True:
                name = f'{key:<13}'
                star = f'{val[1]:.10}'.ljust(13)

                if (val[2] != 'manual' and self.dataset is not None and
                    f'{key}_dyn' in self.dataset):
                    vari = f'{str(val[2])} & manual'.ljust(14)
                elif val[2] is not None:
                    vari = f'{str(val[2]):<14}'
                else:
                    vari = '      -       '


                boun = (f'{val[3][0][0]:.5}' + "-" +
                        f'{val[3][1][0]:.5}').ljust(13)
                try:
                    inte = f'{str(val[4]):<14}'
                except:
                    IndexError
                    inte = 'False'.ljust(14)

                fitted += (f' {name}| {star}| {vari}| {boun}| {inte}|\n')

            if val[0] is False:
                if val[1] == 'auxiliary':
                    auxiliary += [f'| {key}']
                elif isinstance(val[1], (int, float)):
                    fixed += [f' {key:<13}= {val[1]}']
        try:
            vname = self.V.__class__.__name__
        except Exception:
            vname = '?'
        try:
            srfname = self.SRF.__class__.__name__
        except Exception:
            srfname = '?'

        outstr = '-'*77 + '\n'
        outstr += '# SCATTERING FUNCTIONS ' + '\n'

        outstr += f' Volume: {vname}'.ljust(37) + '|'
        outstr += f' Surface: {srfname}' + '\n\n'
        outstr += f'# Interaction-contribution?      {self.int_Q}'+'\n\n'

        outstr += '-'*29 + ' FITTED PARAMETERS ' + '-'*29 + '\n'
        outstr += fitted + '\n'
        outstr += ('-'*9 +' FIXED PARAMETERS ' + '-'*10 + '|' +
                   '-'*10 + ' AUXILIARY DATASETS ' + '-'*9) + '\n'

        naux = len(max([auxiliary, fixed], key=len))
        for i in range(naux):
            try:
                outstr += fixed[i].ljust(37)
            except IndexError:
                outstr += ''.ljust(37)

            try:
                outstr += auxiliary[i].ljust(38)
            except IndexError:
                outstr += ''.ljust(38)
            outstr += '\n'
        outstr += '-'*77

        # write least-squares keywords (maintain order)
        lsqkw = self.lsq_kwargs
        keys = [*lsqkw.keys()]
        if len(lsqkw) > 0:
            outstr += '\n# LSQ PARAMETERS ' + '\n'
            for key1, key2 in zip(keys[:(len(keys) + 1)//2],
                                  keys[(len(keys) + 1)//2:]):
                outstr += f' {key1:<15}= {lsqkw[key1]}'.ljust(37) + \
                          f'  {key2:<15}= {lsqkw[key2]}' + '\n'
            outstr += '-'*77

        print(outstr)
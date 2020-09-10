"""a set of common performance metrics"""
from scipy import stats
from functools import lru_cache

from itertools import chain, repeat, permutations
from operator import itemgetter
from functools import wraps

from .general_functions import groupby_unsorted


class _metric_keys(object):
    """
    a class to get all available variable-keys that can be used to
    calculate metrics
    """

    def __init__(self, fit, d1=None, d2=None, auxdat=None):

        self._datakeys = fit.dataset.select_dtypes(include='number').keys()
        self._modelkeys = ['tot', 'surf', 'vol']
        self._retrievalkeys = fit.res_dict.keys()
        if auxdat is not None:
            self._auxkeys = auxdat.keys()
        else:
            self._auxkeys = []

        if fit.int_Q is True:
            self._modelkeys += ['inter']

        self._check_keys()

        if d1 is not None:
            assert isinstance(d1, str), 'd1 must be a string!'
            setattr(self, d1, _RTmetrics0())

            if d2 is None:
                for k2 in self._all_keys:
                    setattr(getattr(self, d1), k2,
                            _RTmetrics1(d1=d1, d2=k2, fit=fit, auxdat=auxdat,
                                        all_keys=self._all_keys))
            elif isinstance(d2, str):
                setattr(getattr(self, d1), d2,
                        _RTmetrics1(d1=d1, d2=d2, fit=fit))
            else:
                try:
                    d2name = d2.name
                except AttributeError:
                    d2name = 'aux'
                setattr(getattr(self, d1), d2name,
                        _RTmetrics1(d1=d1, d2=d2, fit=fit, auxdat=auxdat,
                                    all_keys=self._all_keys))
        else:
            for k1, k2 in permutations(self._all_keys, 2):
                if not hasattr(self, k1):
                    setattr(self, k1, _RTmetrics0())

                setattr(getattr(self, k1), k2,
                        _RTmetrics1(d1=k1, d2=k2, fit=fit, auxdat=auxdat,
                                    all_keys=self._all_keys))

    def _check_keys(self):
        all_keys = list(chain(self._datakeys,
                              self._modelkeys,
                              self._retrievalkeys,
                              self._auxkeys))

        if len(all_keys) != len(set(all_keys)):
            print('warning, the following keys are present in multiple ' +
                  'sources!')

        suffix = chain(repeat('dataset', len(self._datakeys)),
                       repeat('calc_model',
                              len(self._modelkeys)),
                       repeat('res_df', len(self._retrievalkeys)),
                       repeat('auxdat', len(self._auxkeys)))

        grps = groupby_unsorted(zip(suffix, all_keys),
                                key=itemgetter(1), get=itemgetter(0))

        new_all_keys = []
        src = []
        for key, val in grps.items():
            if len(val) > 1:
                print(f'"{key}": '.ljust(15) + '[' + ', '.join(val) + ']')
                for suffix in val:
                    new_all_keys += [key + '__' + suffix]
                    src += [suffix]
            else:
                new_all_keys += [key]
                src += val

        self._all_keys = dict(zip(new_all_keys, src))


class _RTmetrics0(object):
    """a dummy class to pass variable names"""

    def __init__(self):
        pass


class _RTmetrics1(object):
    def __init__(self, d1, d2, fit, auxdat, all_keys):

        assert d1 in all_keys, f'the key "{d1}" could not be found'
        assert d2 in all_keys, f'the key "{d2}" could not be found'

        self._s1 = all_keys[d1]
        self._s2 = all_keys[d2]

        if d1.endswith(f'__{self._s1}'):
            self._d1 = d1[:-len(f'__{self._s1}')]
        else:
            self._d1 = d1

        if d2.endswith(f'__{self._s2}'):
            self._d2 = d2[:-len(f'__{self._s2}')]
        else:
            self._d2 = d2

        self.fit = fit
        self.auxdat = auxdat

    def _get_data(self, source, key):
        if source == 'auxdat':
            return self.auxdat[key]
        elif source == 'dataset':
            return self.fit.dataset[key]
        elif source == 'calc_model' and key == 'tot':
            return self.fit.calc_model(return_components=False)[key]
        elif source == 'calc_model':
            return self.fit.calc_model(return_components=True)[key]
        elif source == 'res_df':
            return self.fit.res_df[key]

    @property
    @lru_cache()
    def d1(self):
        d1 = self._get_data(self._s1, self._d1)
        return d1

    @property
    @lru_cache()
    def d2(self):
        d2 = self._get_data(self._s2, self._d2)

        assert len(self.d1) == len(d2), ('the length of the 2 datasets is' +
                                         'not the same!' +
                                         f'({len(self.d1)} != {len(d2)})')
        return d2

    @property
    def pearsson(self):
        return RTmetrics.pearsson(self.d1, self.d2)

    @property
    def spearman(self):
        return RTmetrics.spearman(self.d1, self.d2)



class RTmetrics(object):
    def __init__(self):
        pass

    @staticmethod
    def pearsson(d1, d2):
        """
        evaluate pearsson

        Parameters
        ----------
        d1 : TYPE
            DESCRIPTION.
        d2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return stats.pearsonr(d1, d2)

    @staticmethod
    def spearman(d1, d2):
        return stats.spearmanr(d1, d2)

    @staticmethod
    def linregress(d1, d2):
        return dict(zip(['slope', 'intercept', 'pearson', 'pvalue', 'stderr'],
                        stats.linregress(d1, d2)))

    # TODO metrics to implement
    #
    # RMSD
    # ubRMSD
    # bias
    # standard-deviation ratio
    # MAE
    # MAPE
    # ... any additional ideas?

    # a convenience method to get all metrics in one step
    # (just the numbers without additional information)

    # a convenience method to print a text-summary of the metrics

    # a scatterplot function that shows the data as well as the metrics

    # proper docstrings for the metric-functions


    @property
    def allmetrics(self):
        return dict(pearson=self.pearsson[0],
                    spearman=self.spearman[0])

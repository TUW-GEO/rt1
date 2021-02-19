"""a set of common performance metrics"""
from scipy import stats
from functools import lru_cache

from itertools import chain, repeat, permutations
from operator import itemgetter
from decimal import Decimal

from .general_functions import groupby_unsorted
from . import log

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class _metric_keys(object):
    """
    a class to get all available variable-keys that can be used to
    calculate metrics
    """

    def __init__(self, fit, d1=None, d2=None, auxdat=None):

        self._datakeys = fit.dataset.select_dtypes(include="number").keys()
        self._modelkeys = ["tot", "surf", "vol"]
        self._retrievalkeys = fit.res_dict.keys()

        if auxdat is not None:
            self._auxkeys = auxdat.keys()
        else:
            self._auxkeys = []

        if fit.int_Q is True:
            self._modelkeys += ["inter"]

        self._check_keys()

        if d1 is not None:
            assert isinstance(d1, str), "d1 must be a string!"
            setattr(self, d1, _RTmetrics0())

            if d2 is None:
                for k2 in self._all_keys:
                    setattr(
                        getattr(self, d1),
                        k2,
                        _RTmetrics1(
                            d1=d1,
                            d2=k2,
                            fit=fit,
                            auxdat=auxdat,
                            all_keys=self._all_keys,
                        ),
                    )
            elif isinstance(d2, str):
                setattr(getattr(self, d1), d2, _RTmetrics1(d1=d1, d2=d2, fit=fit))
            else:
                try:
                    d2name = d2.name
                except AttributeError:
                    d2name = "aux"
                setattr(
                    getattr(self, d1),
                    d2name,
                    _RTmetrics1(
                        d1=d1,
                        d2=d2,
                        fit=fit,
                        auxdat=auxdat,
                        all_keys=self._all_keys,
                    ),
                )
        else:
            for k1, k2 in permutations(self._all_keys, 2):
                if not hasattr(self, k1):
                    setattr(self, k1, _RTmetrics0())

                setattr(
                    getattr(self, k1),
                    k2,
                    _RTmetrics1(
                        d1=k1,
                        d2=k2,
                        fit=fit,
                        auxdat=auxdat,
                        all_keys=self._all_keys,
                    ),
                )

    def _check_keys(self):
        # a list of all possible keys that can be used for metric calculations
        all_keys = chain(
            self._datakeys, self._modelkeys, self._retrievalkeys, self._auxkeys
        )
        # a list of the "sources" that belong to the keys
        suffix = chain(
            repeat("dataset", len(self._datakeys)),
            repeat("calc_model", len(self._modelkeys)),
            repeat("res_df", len(self._retrievalkeys)),
            repeat("auxdat", len(self._auxkeys)),
        )
        # group by the sources to check if any key is defined more than once
        grps = groupby_unsorted(
            zip(suffix, all_keys), key=itemgetter(1), get=itemgetter(0)
        )

        # make all keys unique (e.g. add a suffix if there are multiple
        # appearances of the same key -> also warn the user of multiple keys!)
        newgrps = dict()
        warnmsg = ""
        for key, val in grps.items():
            if len(val) > 1:
                warnmsg += f'"{key}": '.ljust(15) + "[" + ", ".join(val) + "]"
                warnmsg += "\n"
                for i in val:
                    newgrps[key + "__" + i] = i
            else:
                newgrps[key] = val[0]

        if len(warnmsg) > 0:
            log.warning(
                "the following keys are present in multiple sources!\n" + warnmsg
            )

        self._all_keys = newgrps


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

        if d1.endswith(f"__{self._s1}"):
            self._d1 = d1[: -len(f"__{self._s1}")]
        else:
            self._d1 = d1

        if d2.endswith(f"__{self._s2}"):
            self._d2 = d2[: -len(f"__{self._s2}")]
        else:
            self._d2 = d2

        self.fit = fit
        self.auxdat = auxdat

    def _get_data(self, source, key):
        if source == "auxdat":
            return self.auxdat[key]
        elif source == "dataset":
            return self.fit.dataset[key]
        elif source == "calc_model" and key == "tot":
            return self.fit.calc_model(return_components=False)[key]
        elif source == "calc_model":
            return self.fit.calc_model(return_components=True)[key]
        elif source == "res_df":
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
        return d2

    @property
    @lru_cache()
    def _unify_idx_data(self):
        if len(self.d1) != len(self.d2):
            log.warning(
                f'index of "{self._d1}" and "{self._d2}" is not '
                + "the same! -> a concatenation is performed!"
            )

            # try to unify the index
            df = pd.concat([self.d1, self.d2], axis=1, copy=False)
            d1 = df[self._d1].dropna()
            d2 = df[self._d2].dropna()

            assert len(d1) == len(d2), (
                "the length of the 2 datasets is "
                + "not the same!"
                + f"({len(self.d1)} != {len(d2)})"
            )
            return d1, d2
        else:
            return self.d1, self.d2

    @property
    def pearson(self):
        return RTmetrics.pearson(*self._unify_idx_data)

    @property
    def spearman(self):
        return RTmetrics.spearman(*self._unify_idx_data)

    @property
    def linregress(self):
        return RTmetrics.linregress(*self._unify_idx_data)

    @property
    def rmsd(self):
        return RTmetrics.rmsd(*self._unify_idx_data)

    @property
    def ub_rmsd(self):
        return RTmetrics.ub_rmsd(*self._unify_idx_data)

    @property
    def bias(self):
        return RTmetrics.bias(*self._unify_idx_data)

    @property
    def mae(self):
        return RTmetrics.mae(*self._unify_idx_data)

    @property
    def mape(self):
        return RTmetrics.mape(*self._unify_idx_data)

    @property
    def std_ratio(self):
        return RTmetrics.std_ratio(*self._unify_idx_data)

    @property
    def allmetrics(self):
        return RTmetrics.allmetrics(*self._unify_idx_data)

    @property
    def metrics_table(self):
        return RTmetrics.metrics_table(*self._unify_idx_data)

    def scatterplot(self):
        RTmetrics.scatterplot(*self._unify_idx_data, self._d1, self._d2)


class RTmetrics(object):

    # registry of metric methods used for allmetrics and metrics_table
    # enter the function name of a new metric in here
    # functions listed in here must have two pandas series d1, d2 as parameters
    metrics_registry = [
        "pearson",
        "spearman",
        "linregress",
        "rmsd",
        "ub_rmsd",
        "bias",
        "mae",
        "mape",
        "std_ratio",
    ]

    def __init__(self):
        pass

    @staticmethod
    def pearson(d1, d2):
        """
        evaluates pearson correlation coefficient of given series d1 and d2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            pearson correlation coefficient

        """
        return d1.corr(d2, method="pearson")

    @staticmethod
    def spearman(d1, d2):
        """
        evaluates spearman's rank correlation coefficient of given series d1 and d2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            spearman's rank correlation coefficient

        """
        return d1.corr(d2, method="spearman")

    @staticmethod
    def linregress(d1, d2):
        """
        evaluates pearson correlation coefficient of given series d1 and d2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        dictionary {string: float}
            float values for slope, intercept, pearson, pvalue, stderr

        """
        return dict(
            zip(
                ["slope", "intercept", "pearson", "pvalue", "stderr"],
                stats.linregress(d1, d2),
            )
        )

    @staticmethod
    def rmsd(d1, d2):
        """
        evaluates root mean square deviation of given series d1 and d2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            root mean square deviation

        """
        diff_sq = d1.subtract(d2).pow(2)
        return np.sqrt(diff_sq.mean())

    @staticmethod
    def ub_rmsd(d1, d2):
        """
        evaluates unbiased root mean square deviation of given series d1 and d2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            unbiased root mean square deviation

        """
        d1_corr = d1 - d1.mean()
        d2_corr = d2 - d2.mean()
        diff_sq = d1_corr.subtract(d2_corr).pow(2)
        return np.sqrt(diff_sq.mean())

    @staticmethod
    def bias(d1, d2):
        """
        evaluates bias of given series d1 and d2: mu_1 - mu_2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            bias

        """
        return d1.mean() - d2.mean()

    @staticmethod
    def mae(d1, d2):
        """
        evaluates mean absolute error of given Series d1 and d2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            mean absolute error

        """
        abs_diff = d1.subtract(d2).abs()
        return abs_diff.mean()

    @staticmethod
    def mape(d1, d2):
        """
        evaluates mean absolute percentage error of given Series d1 and d2 with respect to d1

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            mean absolute percentage error

        """
        abs_rel_diff = d1.subtract(d2).div(d1).abs()
        return abs_rel_diff.mean()

    @staticmethod
    def std_ratio(d1, d2):
        """
        evaluates standard deviation ratio of given Series d1 and d2: sigma_1 / sigma_2

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        float
            standard deviation ratio

        """
        return d1.std() / d2.std()

    @classmethod
    def allmetrics(cls, d1, d2):
        """
        run all metrics specified in RTmetrics.metrics_registry of given Series d1 and d2
        metrics have to be specified by function name in metrics_registry

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        dictionary {string: (float or dictionary)}
            function/metric name and corresponding value

        """
        return {func: getattr(cls, func)(d1, d2) for func in cls.metrics_registry}

    @classmethod
    def scatterplot(cls, d1, d2, d1_name, d2_name):
        """
        draws scatterplot of two given series d1 and d2
        list all metrics in a table beside the scatterplot

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2
        d1_name : string
            name of time series 1
        d2_name : string
            name of time series 2

        Returns
        -------
        scatterplot figure object

        """

        # create plot and axes
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [5, 1]}
        )

        # create scatterplot
        ax1.scatter(d1, d2)
        ax1.set_xlabel(d1_name)
        ax1.set_ylabel(d2_name)

        # get all metrics and define lists for table data
        metrics_dict = cls.allmetrics(d1, d2)
        metric_names = []
        metric_values = []

        # flatten metrics array and format float values
        for key, val in cls._flatten_dictionary(metrics_dict).items():
            metric_names.append(key)
            if isinstance(val, float):
                metric_values.append("%1.3f" % val)
            else:
                metric_values.append(val)

        # add another dimension for usage in table
        two_dim_metric_values = [[metric_value] for metric_value in metric_values]

        # remove border to only show table itself
        ax2.axis("off")

        # plot and create table object
        metrics_table = ax2.table(
            cellText=two_dim_metric_values,
            rowLabels=metric_names,
            colLabels=["Value"],
            loc="center",
        )

        # scale for higher cells
        metrics_table.scale(1, 1.5)

        plt.show()
        return fig

    @classmethod
    def _flatten_dictionary(cls, dictionary, depth=0):
        """
        recursively flattens a dictionary, only returns float values from that dictionary
        keys of sub-dictionaries are prefixed with a '-' according to the depth

        Parameters
        ----------
        dictionary : dict
            dictionary that should be flattened, should contain float or dict values
        depth : integer
            recursion depth used for prefixing

        Returns
        -------
        dict
            flat dictionary containing numbers or strings

        """
        items = []
        for key, val in dictionary.items():
            new_key = "-" * depth + " " + key
            if isinstance(val, dict):
                items.append((new_key, ""))
                items.extend(cls._flatten_dictionary(val, depth + 1).items())
            elif isinstance(val, float):
                items.append((new_key, val))

        return dict(items)

    @classmethod
    def metrics_table(cls, d1, d2):
        """
        prints a table with all metrics and values returned from the allmetrics method for given series d1 and d2
        dictionaries returned by allmetrics are handled recursively by _metrics_table_dict_entry

        Parameters
        ----------
        d1 : pandas.Series
            time series 1
        d2 : pandas.Series
            time series 2

        Returns
        -------
        void

        """
        metrics_dict = cls.allmetrics(d1, d2)

        header = "-" * 11 + " METRICS " + "-" * 11 + "\n"
        columns = "     METRIC".ljust(14) + "|     VALUE".ljust(15) + " |" + "\n"

        entries = cls._metrics_table_dict_entry(metrics_dict)

        outstr = header + columns + entries
        print(outstr)

    @classmethod
    def _metrics_table_dict_entry(cls, metrics_dict, depth=0):
        """
        recursively generates entries string for dictionaries in a metrics table
        entries of sub-dictionaries are indented

        Parameters
        ----------
        metrics_dict : dictionary
            metrics dictionary to generate string from
        depth : integer
            recursion depth

        Returns
        -------
        string
            multiline string containing all floating point entries of a dictionary and its sub-dictionaries

        """
        entries = ""
        for key, val in metrics_dict.items():
            depth_offset = 2 * depth
            metric = f"{key:<{13 - depth_offset}}"

            if isinstance(val, (float, int)):
                valstr = (
                    f"{val:.6f}".ljust(14)
                    if abs(val) < 1e5
                    else f"{Decimal(val):.6E}".ljust(14)
                )
                entries += " " + "--" * depth + f"{metric}| {valstr}|\n"

            elif isinstance(val, dict):
                entries += (
                    " " + "--" * depth + f"{metric}|".ljust(29 - depth_offset) + "|\n"
                )
                entries += cls._metrics_table_dict_entry(val, depth + 1)

        return entries

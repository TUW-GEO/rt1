import unittest
import os
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rt1.rtmetrics import RTmetrics
from rt1.rtresults import HDFaccessor


class TestRTMetrics(unittest.TestCase):
    @staticmethod
    def mock_series():
        d1 = pd.Series(
            [43.04, 20.55, 8.98, -15.27, 29.18, -12.48, 78.35, 92.73, -23.31, 14.12]
        )
        d2 = pd.Series(
            [11.52, 116.34, 60.88, 9.73, 35.51, 26.69, 119.53, -16.41, 25.05, -68.33]
        )
        results = {
            "pearson": 0.13745925602259895,
            "spearman": 0.10303030303030303,
            "linregress": {
                "slope": 0.2014594067161711,
                "intercept": 27.29877405497224,
                "rvalue": 0.13745925602259895,
                "pvalue": 0.7049253466715144,
                "stderr": 0.513246861349088,
            },
            "rmsd": 61.48676800743392,
            "ub_rmsd": 60.901701092826634,
            "bias": -8.462000000000003,
            "mae": 53.084,
            "mape": 2.578235110085792,
            "std_ratio": 0.682317387225608,
        }
        return (d1, d2, results)

    @staticmethod
    def mock_fit():
        fit_dB_path = os.path.dirname(__file__) + os.sep + "test_fit_db.h5"

        with HDFaccessor(fit_dB_path) as fit_db:
            fit = fit_db.load_fit("sig0_dB")

        return fit

    def test_metrics(self):
        d1, d2, expected_values = self.mock_series()
        # loop through all possible metrics
        for metric in RTmetrics.metrics_registry:
            if metric == "linregress":
                continue

            res = getattr(RTmetrics, metric)(d1, d2)
            self.assertAlmostEqual(res, expected_values[metric])

    def test_linregress(self):
        d1, d2, expected_values = self.mock_series()
        linregress = RTmetrics.linregress(d1, d2)
        linregress_expected = expected_values["linregress"]

        self.assertAlmostEqual(linregress["slope"], linregress_expected["slope"])
        self.assertAlmostEqual(
            linregress["intercept"], linregress_expected["intercept"]
        )
        self.assertAlmostEqual(linregress["rvalue"], linregress_expected["rvalue"])
        self.assertAlmostEqual(linregress["pvalue"], linregress_expected["pvalue"])
        self.assertAlmostEqual(linregress["stderr"], linregress_expected["stderr"])

    def test_fit_metric(self):

        fit = self.mock_fit()

        # loop through all possible parameter combinations
        for [p1, s1], [p2, s2] in combinations(fit.metric._all_keys.items(), 2):

            metric_fit_params = getattr(getattr(fit.metric, p1), p2)

            # loop through all possible metrics
            for metric in RTmetrics.metrics_registry:
                fit_metric = getattr(metric_fit_params, metric)

                # remove suffix if present
                if p1.endswith(f"__{s1}"):
                    p1 = p1[: -len(f"__{s1}")]
                if p2.endswith(f"__{s2}"):
                    p2 = p2[: -len(f"__{s2}")]

                d1 = getattr(fit, s1)
                if s1 == "calc_model":
                    d1 = getattr(d1(return_components=True), p1)
                else:
                    d1 = getattr(d1, p1)

                d2 = getattr(fit, s2)
                if s2 == "calc_model":
                    d2 = getattr(d2(return_components=True), p2)
                else:
                    d2 = getattr(d2, p2)

                df = pd.concat([d1, d2], axis=1)
                fit_func = getattr(RTmetrics, metric)(df[p1], df[p2])

                assertmsg = (
                    "there's been something wrong during metric-evaluation!\n"
                    f'    p1="{p1}", p2="{p2}", metric="{metric}"\n'
                    + f"    fit_metric={fit_metric}\n"
                    + f"    fit_func={fit_func}\n"
                )

                if metric == "linregress":
                    for key, val in fit_metric.items():
                        if np.isnan(val):
                            assert np.isnan(fit_func[key]), assertmsg
                        else:
                            assert val == fit_func[key], assertmsg
                else:
                    if np.isnan(fit_metric):
                        assert np.isnan(fit_func), assertmsg
                    else:
                        assert fit_metric == fit_func, assertmsg

            # check additional methods
            f = metric_fit_params.scatterplot()
            plt.close(f)

            _ = metric_fit_params.allmetrics

            _ = metric_fit_params.metrics_table

    def test_scatterplot(self):
        plt.ion()
        fit = self.mock_fit()
        _ = RTmetrics.scatterplot(fit.res_df.SM, fit.res_df.omega, "SM", "omega")


if __name__ == "__main__":
    unittest.main()

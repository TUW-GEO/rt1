import unittest
import pandas as pd
import matplotlib.pyplot as plt

from rt1.rtmetrics import RTmetrics
from rt1 import rtfits


class TestRTMetrics(unittest.TestCase):  

    @staticmethod
    def mock_series():
        d1 = pd.Series([43.04, 20.55, 8.98, -15.27, 29.18, -12.48, 78.35, 92.73, -23.31, 14.12])
        d2 = pd.Series([11.52, 116.34, 60.88, 9.73, 35.51, 26.69, 119.53, -16.41, 25.05, -68.33])
        results = {
            'pearson': 0.13745925602259895,
            'spearman': 0.10303030303030303,
            'linregress': {
                'slope': 0.2014594067161711,
                'intercept': 27.29877405497224,
                'pearson': 0.13745925602259895,
                'pvalue': 0.7049253466715144,
                'stderr': 0.513246861349088},
             'rmsd': 61.48676800743392,
             'ub_rmsd': 60.901701092826634,
             'bias': -8.462000000000003,
             'mae': 53.084,
             'mape': 2.578235110085792,
             'std_ratio': 0.682317387225608
             }
        return (d1, d2, results)
    
    @staticmethod
    def mock_fit():
        return rtfits.load('./tests/sig0_dB.dump')

    def test_pearson(self):     
        d1, d2, expected_values = self.mock_series()
        pearson = RTmetrics.pearson(d1, d2)
        
        self.assertAlmostEqual(pearson, expected_values['pearson'])

    def test_spearman(self):  
        d1, d2, expected_values = self.mock_series()
        spearman = RTmetrics.spearman(d1, d2)

        self.assertAlmostEqual(spearman, expected_values['spearman'])
        
    def test_linregress(self):  
        d1, d2, expected_values = self.mock_series()
        linregress = RTmetrics.linregress(d1, d2)
        linregress_expected = expected_values['linregress']
        
        self.assertAlmostEqual(linregress['slope'], linregress_expected['slope'])
        self.assertAlmostEqual(linregress['intercept'], linregress_expected['intercept'])
        self.assertAlmostEqual(linregress['pearson'], linregress_expected['pearson'])
        self.assertAlmostEqual(linregress['pvalue'], linregress_expected['pvalue'])
        self.assertAlmostEqual(linregress['stderr'], linregress_expected['stderr'])
        
    def test_rmsd(self):  
        d1, d2, expected_values = self.mock_series()
        rmsd = RTmetrics.rmsd(d1, d2)
        
        self.assertAlmostEqual(rmsd, expected_values['rmsd'])
        
    def test_ub_rmsd(self):  
        d1, d2, expected_values = self.mock_series()
        ub_rmsd = RTmetrics.ub_rmsd(d1, d2)
        
        self.assertAlmostEqual(ub_rmsd, expected_values['ub_rmsd'])
        
    def test_bias(self):  
        d1, d2, expected_values = self.mock_series()
        bias = RTmetrics.bias(d1, d2)
        
        self.assertAlmostEqual(bias, expected_values['bias'])
        
    def test_mae(self):  
        d1, d2, expected_values = self.mock_series()
        mae = RTmetrics.mae(d1, d2)
        
        self.assertAlmostEqual(mae, expected_values['mae'])
        
    def test_mape(self):  
        d1, d2, expected_values = self.mock_series()
        mape = RTmetrics.mape(d1, d2)
        
        self.assertAlmostEqual(mape, expected_values['mape'])
        
    def test_std_ratio(self):  
        d1, d2, expected_values = self.mock_series()
        std_ratio = RTmetrics.std_ratio(d1, d2)
        
        self.assertAlmostEqual(std_ratio, expected_values['std_ratio'])
        
    def test_pearson_fit_metric(self):
        fit = self.mock_fit()
        pearson_fit = fit.metric.SM.omega.pearson
        pearson_func = RTmetrics.pearson(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(pearson_fit, pearson_func)
        
    def test_spearman_fit_metric(self):
        fit = self.mock_fit()
        spearman_fit = fit.metric.SM.omega.spearman
        spearman_func = RTmetrics.spearman(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(spearman_fit, spearman_func)
        
    def test_linregress_fit_metric(self):
        fit = self.mock_fit()
        linregress_fit = fit.metric.SM.omega.linregress
        linregress_func = RTmetrics.linregress(fit.res_df.SM, fit.res_df.omega)
        
        self.assertDictEqual(linregress_fit, linregress_func)
        
    def test_rmsd_fit_metric(self):
        fit = self.mock_fit()
        rmsd_fit = fit.metric.SM.omega.rmsd
        rmsd_func = RTmetrics.rmsd(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(rmsd_fit, rmsd_func)
        
    def test_ub_rmsd_fit_metric(self):
        fit = self.mock_fit()
        _, _, expected_values = self.mock_series()
        ub_rmsd_fit = fit.metric.SM.omega.ub_rmsd
        ub_rmsd_func = RTmetrics.ub_rmsd(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(ub_rmsd_fit, ub_rmsd_func)
        
    def test_mae_fit_metric(self):
        fit = self.mock_fit()
        mae_fit = fit.metric.SM.omega.mae
        mae_func = RTmetrics.mae(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(mae_fit, mae_func)
        
    def test_mape_fit_metric(self):
        fit = self.mock_fit()
        mape_fit = fit.metric.SM.omega.mape
        mape_func = RTmetrics.mape(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(mape_fit, mape_func)
        
    def test_std_ratio_fit_metric(self):
        fit = self.mock_fit()
        std_ratio_fit = fit.metric.SM.omega.std_ratio
        std_ratio_func = RTmetrics.std_ratio(fit.res_df.SM, fit.res_df.omega)
        
        self.assertEquals(std_ratio_fit, std_ratio_func)
        
    def test_scatterplot(self):
        fit = self.mock_fit()
        pl = RTmetrics.scatterplot(fit.res_df.SM, fit.res_df.omega, "SM", "omega")
        plt.close(pl)
        

if __name__ == "__main__":
    unittest.main()



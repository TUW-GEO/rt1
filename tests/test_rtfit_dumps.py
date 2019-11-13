"""
Test the fits-module by loading a dumped rtfits result and performing
all actions again

"""

import unittest
import numpy as np
import cloudpickle
import matplotlib.pyplot as plt
import copy
import os

class TestDUMPS(unittest.TestCase):
    def setUp(self):
        self.sig0_dB_path = os.path.dirname(__file__) + os.sep + "sig0_dB.dump"
        self.sig0_linear_path = os.path.dirname(__file__) + os.sep + "sig0_linear.dump"

    def load_data(self, path):
        with open(path, 'rb') as file:
            fit = cloudpickle.load(file)
        return fit


#        self.assertTrue(
#            err < errdict[key],
#            msg='derived error' + str(err) + 'too high for ' + str(key))

    def test_rtplots(self):

        for path, msg in zip([self.sig0_dB_path, self.sig0_linear_path],
                             ['dB', 'linear']):

            print(f'testing plotfunctions for {msg} fit')
            fit = self.load_data(path)

            fitset = {
                'int_Q': True,
                '_fnevals_input': None,
                'verbose': 0,
                'ftol': 1e-3,
                'gtol': 1e-3,
                'xtol': 1e-3,
                'max_nfev': 1,
                'method': 'trf',
                'tr_solver': 'lsmr',
                'x_scale': 'jac'}
            # call performfit to re-initialize _fnevals functions
            # (they might have been removed if symeninge has been used)
            fit.performfit(fitset=fitset)

            # get list of available plot-methods
            method_list = [func for func in dir(fit.plot) if
                           callable(getattr(fit.plot, func)) and not func.startswith("__")]

            for function_name in method_list:
                print(f'... {function_name}')
                f = fit.plot.__getattribute__(function_name)()
                if function_name == 'printsig0analysis':
                    plt.close(f[0])
                else:
                    plt.close(f)

    def test_performfit(self):
        # settings with whome the fit has been performed
        fitset = {
            'int_Q': True,
            '_fnevals_input': None,
            'verbose': 0,
            'ftol': 1e-3,
            'gtol': 1e-3,
            'xtol': 1e-3,
            'max_nfev': 50,
            'method': 'trf',
            'tr_solver': 'lsmr',
            'x_scale': 'jac'}

        for path, msg in zip([self.sig0_dB_path, self.sig0_linear_path],
                             ['dB', 'linear']):

            print(f'testing plotfunctions for {msg} fit')
            fit = self.load_data(path)

            old_results = copy.deepcopy(fit.res_dict)
            print('testing performfit')
            fit.performfit(fitset=fitset)

            for key, val in old_results.items():
                self.assertTrue(np.allclose(fit.res_dict[key], val, atol=1e-4, rtol=1e-4),
                                msg=f'fitted values for {msg} fit of {key} ' +
                                     f'differ by {np.mean(fit.res_dict[key] - val)}')



if __name__ == "__main__":
    unittest.main()
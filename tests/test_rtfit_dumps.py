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
            lsq_kwargs = {
                'ftol': 1e-3,
                'gtol': 1e-3,
                'xtol': 1e-3,
                'max_nfev': 1,
                'method': 'trf',
                'tr_solver': 'lsmr',
                'x_scale': 'jac'}
            fit.lsq_kwargs = lsq_kwargs

            # call performfit to re-initialize _fnevals functions
            # (they might have been removed if symeninge has been used)
            fit.performfit()

            # get list of available plot-methods
            method_list = [func for func in dir(fit.plot) if
                           callable(getattr(fit.plot, func)) and not func.startswith("__")]

            for function_name in method_list:
                print(f'... {function_name}')
                if function_name == 'printsig0analysis':
                    f, s1, s2 = fit.plot.__getattribute__(function_name)(
                            secondslider=True, dayrange2=1)
                    # check update functions
                    s1.set_val(1)
                    s2.set_val(1)

                    plt.close(f)
                elif function_name == 'analyzemodel':
                    f, sliders, txt_but = fit.plot.__getattribute__(
                        function_name)()

                    # check update functions
                    for key, s in sliders.items():
                        s.set_val((s.valmax - s.valmin)/2.)

                    for key, b in txt_but.items():
                        if key == 'buttons':
                            # the initial status is ALL OFF
                            stat = b.get_status()
                            for i in range(len(stat)):
                                b.set_active(i)
                            # now all should be ON
                            self.assertTrue(np.all(b.get_status()))
                            for i in range(len(stat)):
                                b.set_active(i)
                            # now all should be OFF again
                            self.assertTrue(~np.all(b.get_status()))
                        else:
                            # set the boundaries of the parameters
                            if 'min' in key:
                                b.set_val(0.02)
                            if 'max' in key:
                                b.set_val(0.99)

                    plt.close(f)
                else:
                    f = fit.plot.__getattribute__(function_name)()
                    plt.close(f)

    def test_performfit(self):
        # settings with whome the fit has been performed
        lsq_kwargs = {
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
            fit.lsq_kwargs = lsq_kwargs
            old_results = copy.deepcopy(fit.res_dict)

            # print model definition
            fit.model_definition

            print('testing performfit')
            fit.performfit()

            for key, val in old_results.items():
                self.assertTrue(np.allclose(np.repeat(*fit.res_dict[key]), val, atol=1e-4, rtol=1e-4),
                                msg=f'fitted values for {msg} fit of {key} ' +
                                     f'differ by {np.mean(np.repeat(*fit.res_dict[key]) - val)}')



if __name__ == "__main__":
    unittest.main()
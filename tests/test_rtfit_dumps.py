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

            # call performfit to re-initialize _fnevals functions
            # and evaluate intermediate results
            # (they might have been removed if symeninge has been used)
            fit.lsq_kwargs['verbose'] = 0
            fit.performfit(intermediate_results=True,
                           print_progress=True)

            # get list of available plot-methods
            method_list = [func for func in dir(fit.plot) if
                           callable(getattr(fit.plot, func)) and not func.startswith("__")]

            for function_name in method_list:
                print(f'... {function_name}')
                if function_name == 'printsig0analysis':
                    # check 'dataset' index slider
                    f, s1, s2 = fit.plot.__getattribute__(function_name)(
                            range2=2, range1=1, use_index='dataset')
                    # check update functions
                    s1.set_val(1)
                    s2.set_val(1)
                    plt.close(f)

                    # check 'groups' index slider
                    f, s1, s2 = fit.plot.__getattribute__(function_name)(
                            range2=2, range1=1, use_index='groups')
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

                elif function_name == 'intermediate_residuals':
                    # check default (e.g. pandas datetime-offset)
                    f = fit.plot.__getattribute__(function_name)(fmt='%d.%b %Y')
                    plt.close(f)
                    # check grouping with respect to incidence angles and
                    # convert the labels to degrees
                    f = fit.plot.__getattribute__(function_name)(
                        grp=('inc', 10),
                        label_formatter=lambda x,y:round(np.rad2deg(x),2))
                    plt.close(f)
                    # check grouping with respect to datetimes
                    f = fit.plot.__getattribute__(function_name)(grp='groups')
                    plt.close(f)
                    # check grouping with respect to the dataset index
                    f = fit.plot.__getattribute__(function_name)(
                        grp='dataset', plottype='2D', fmt='%Y %b %d (%H:%M)')
                    plt.close(f)

                else:
                    f = fit.plot.__getattribute__(function_name)()
                    plt.close(f)

    def test_performfit(self):
        for path, msg in zip([self.sig0_dB_path, self.sig0_linear_path],
                             ['dB', 'linear']):

            print(f'testing plotfunctions for {msg} fit')
            fit = self.load_data(path)

            old_results = fit.res_dict
            # print model definition
            fit.model_definition
            print('testing performfit')
            fit.lsq_kwargs['verbose'] = 0
            fit.performfit(intermediate_results=True,
                           print_progress=True)

            # call _cache_info() to make coveralls happy
            fit._cache_info()
            fit.R._cache_info()

            # try to dump the file again (without fit-details)
            fit.dump(os.path.join(os.path.dirname(__file__), 'testdump1.dump'),
                     mini=True)
            # try to dump the file again (with fit-details)
            fit.dump(os.path.join(os.path.dirname(__file__), 'testdump2.dump'),
                     mini=False)

            for key, val in old_results.items():
                self.assertTrue(np.allclose(fit.res_dict[key],
                                            old_results[key], atol=1e-4, rtol=1e-4),
                                msg=f'fitted values for {msg} fit of {key} ' +
                                     f'differ by {np.subtract(fit.res_dict[key], old_results[key]).mean()}')



if __name__ == "__main__":
    unittest.main()
# -*- coding: utf-8 -*-
"""
test configparser

a quick check if a standard config-file is read correctly
"""

import unittest
import os
from rt1.rtfits import RT1_configparser
import sympy as sp


class TestCONFIGPARSER(unittest.TestCase):
    def setUp(self):
        self.configpath = os.path.join(os.path.dirname(__file__),
                                       'test_config.ini')


    def test_load_cfg(self):
        cfg = RT1_configparser(self.configpath)

        lsq_kwargs = {'verbose': 2,
                      'ftol': 0.0001,
                      'gtol': 0.0001,
                      'xtol': 0.0001,
                      'max_nfev': 20,
                      'method': 'trf',
                      'tr_solver': 'lsmr',
                      'x_scale': 'jac'}

        defdict = {'omega': [True, 0.05, '2M', ([0.01], [0.5])],
                   't_v': [True, 0.25, None, ([0.01], [0.5])],
                    't_s': [True, 0.25, None, ([0.01], [0.5])],
                    'N': [True, 0.1, 'index', ([0.01], [0.2])],
                    'tau': [True, 0.5, '3M', ([0.01], [1.5])],
                    'bsf': [True, 0.05, 'A', ([0.01], [1.0])],
                    'tau_multip': [False, 0.5]}

        set_V_SRF = {'V_props': {'V_name': 'HenyeyGreenstein',
                                 'tau': 'tau * tau_multip',
                                 'omega': 'omega',
                                 't': 't_v',
                                 'ncoefs': 10},
                     'SRF_props': {'SRF_name': 'HG_nadirnorm',
                                   'NormBRDF': 'N',
                                   't': 't_s',
                                   'ncoefs': 10}}

        fits_kwargs = {'sig0': True,
                       'dB': False,
                       'setindex': 'first',
                       'int_Q': False,
                       'lambda_backend': 'symengine',
                       'interp_vals': ['tau', 'omega'],
                       '_fnevals_input': None}


        configdicts = cfg.get_config()

        for key, val in lsq_kwargs.items():
            assert key in configdicts['lsq_kwargs'], f'error in lsq_kwargs {key}'
            assert val == configdicts['lsq_kwargs'][key], f'error in lsq_kwargs {key}'

        for key, val in defdict.items():
            assert key in configdicts['defdict'], f'error in defdict {key}'
            for i, val_i in enumerate(val[:3]):
                if len(val) <= 2: continue
                assert val_i == configdicts['defdict'][key][i], f'error in defdict {key}'
            if len(val) == 4:
                assert val[-1][0][0] == configdicts['defdict'][key][-1][0][0], f'error in defdict {key}'
                assert val[-1][1][0] == configdicts['defdict'][key][-1][1][0], f'error in defdict {key}'

        for key, val in set_V_SRF.items():
            assert key in configdicts['set_V_SRF'], f'error in set_V_SRF {key}'

            for key_i, val_i in val.items():
                assert key_i in configdicts['set_V_SRF'][key], f'error in set_V_SRF["{key}"]["{key_i}"]'

                assert val_i == configdicts['set_V_SRF'][key][key_i], f'error in set_V_SRF["{key}"]["{key_i}"]'

        for key, val in fits_kwargs.items():
            assert key in configdicts['fits_kwargs'], f'error in fits_kwargs {key}'
            assert val == configdicts['fits_kwargs'][key], f'error in fits_kwargs {key}'


        fit = cfg.get_fitobject()
        V, SRF = fit._init_V_SRF(**fit.set_V_SRF)

        assert V.t == sp.Symbol('t_v'), 'V.t assigned incorrectly'
        assert V.tau == sp.Symbol('tau') * sp.Symbol('tau_multip'), 'V.tau assigned incorrectly'
        assert V.omega == sp.Symbol('omega'), 'V.omega assigned incorrectly'
        assert SRF.t == sp.Symbol('t_s'), 'SRF.t assigned incorrectly'
        assert SRF.NormBRDF == sp.Symbol('N'), 'SRF.NormBRDF assigned incorrectly'


if __name__ == "__main__":
    unittest.main()
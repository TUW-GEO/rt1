# -*- coding: utf-8 -*-
"""
test configparser

a quick check if a standard config-file is read correctly
"""
import unittest
import os
from rt1.rtparse import RT1_configparser
import sympy as sp
from pathlib import Path
from datetime import datetime


class TestCONFIGPARSER(unittest.TestCase):
    def setUp(self):
        self.configpath = os.path.join(os.path.dirname(__file__),
                                       'test_config.ini')


    def test_load_cfg(self):
        cfg = RT1_configparser(self.configpath)

        #----------------------------------------- check parsed configdicts
        lsq_kwargs = {'verbose': 0,
                      'ftol': 0.0001,
                      'gtol': 0.0001,
                      'xtol': 0.0001,
                      'max_nfev': 20,
                      'method': 'trf',
                      'tr_solver': 'lsmr',
                      'x_scale': 'jac'}

        defdict = {'omega': [True, 0.05, '2M', ([0.01], [0.5]), True],
                   't_v': [True, 0.25, None, ([0.01], [0.5]), False],
                    't_s': [True, 0.25, None, ([0.01], [0.5]), False],
                    'N': [True, 0.1, 'index', ([0.01], [0.2]), False],
                    'tau': [True, 0.5, '3M', ([0.01], [1.5]), True],
                    'bsf': [True, 0.05, 'A', ([0.01], [1.0]), False],
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
                       'int_Q': True,
                       'lambda_backend': 'symengine',
                       '_fnevals_input': None,
                       'verbose' : 1}

        configdicts = cfg.get_config()

        for key, val in lsq_kwargs.items():
            assert key in configdicts['lsq_kwargs'], f'error in lsq_kwargs {key}'
            assert val == configdicts['lsq_kwargs'][key], f'error in lsq_kwargs {key}'

        for key, val in defdict.items():
            assert key in configdicts['defdict'], f'{key} not in defdict'
            for i, val_i in enumerate(val):
                assert val_i == configdicts['defdict'][key][i], f'error in defdict {key}'

        for key, val in set_V_SRF.items():
            assert key in configdicts['set_V_SRF'], f'error in set_V_SRF {key}'

            for key_i, val_i in val.items():
                assert key_i in configdicts['set_V_SRF'][key], f'error in set_V_SRF["{key}"]["{key_i}"]'

                assert val_i == configdicts['set_V_SRF'][key][key_i], f'error in set_V_SRF["{key}"]["{key_i}"]'

        for key, val in fits_kwargs.items():
            assert key in configdicts['fits_kwargs'], f'error in fits_kwargs {key}'
            assert val == configdicts['fits_kwargs'][key], f'error in fits_kwargs {key}'

        #----------------------------------------- check parsed process_specs

        process_specs = {'finalout_name' : 'results.h5',
                         'dumpfolder' : 'dump01',
                         'save_path' : Path('tests/proc_test'),
                         'f0' : 1245.,
                         'f1' : 5.4,
                         'i0' : 1,
                         'i1' : 5,
                         'b0' : False,
                         'b1' : True,
                         'd0' : datetime(2020,3,23),
                         'd1' : datetime(2017,1,22,12,34),
                         'lf' : [.1,.2,.3,.4,.5],
                         'li' : [1,2,3,4,5],
                         'lb' : [True, False, True, False],
                         'ls' : ['a','B','c','AbCd#'],
                         'ldt': [datetime(2017,1,22,12,34),
                                 datetime(2019,3,24,11,13)]}

        process_specs_parsed = cfg.get_process_specs()

        for key, val in process_specs.items():

            parsedval = process_specs_parsed[key]

            assert key in process_specs_parsed, f'error in PROCESS_SPECS {key}'
            assert val == parsedval, f'error in PROCESS_SPECS {key} ({val} != {parsedval})'

            if key in ['f0', 'f1']:
                assert isinstance(parsedval, float), f'{key} = {parsedval} is not parsed as float!'
            if key in ['i0', 'i1']:
                assert isinstance(parsedval, int), f'{key} = {parsedval} is not parsed as int!'
            if key in ['b0', 'b1']:
                assert isinstance(parsedval, bool), f'{key} = {parsedval} is not parsed as bool!'
            if key in ['d0', 'd1']:
                assert isinstance(parsedval, datetime), f'{key} = {parsedval} is not parsed as datetime!'



        #----------------------------------------- check fitobjects

        fit = cfg.get_fitobject()
        V = fit._init_V_SRF(**fit.set_V_SRF, V_SRF_Q='V')
        SRF = fit._init_V_SRF(**fit.set_V_SRF, V_SRF_Q='SRF')

        assert V.t == sp.Symbol('t_v'), 'V.t assigned incorrectly'
        assert V.tau == sp.Symbol('tau') * sp.Symbol('tau_multip'), 'V.tau assigned incorrectly'
        assert V.omega == sp.Symbol('omega'), 'V.omega assigned incorrectly'
        assert SRF.t == sp.Symbol('t_s'), 'SRF.t assigned incorrectly'
        assert SRF.NormBRDF == sp.Symbol('N'), 'SRF.NormBRDF assigned incorrectly'



        #----------------------------------------- check imported module
        cfg_module_dict = cfg.get_modules()
        assert 'processfuncs' in cfg_module_dict, 'modules not correctly parsed'
        cfg_module = cfg_module_dict['processfuncs']

        assert hasattr(cfg_module, 'processing_cfg'), 'modules not correctly parsed'
        assert hasattr(cfg_module, 'run'), 'modules not correctly parsed'


        cfg_module_direct = cfg.get_module('processfuncs')
        assert hasattr(cfg_module_direct, 'processing_cfg'), 'direct-module load not working'
        assert hasattr(cfg_module_direct, 'run'), 'direct-module load not working'



        #----------------------------------------- check if files have been copied
        assert Path('tests/proc_test/dump01/cfg').exists(), 'copying did not work'
        assert Path('tests/proc_test/dump01/cfg/test_config.ini').exists(), 'copying did not work'
        assert Path('tests/proc_test/dump01/cfg/parallel_processing_config.py').exists(), 'copying did not work'


if __name__ == "__main__":
    unittest.main()
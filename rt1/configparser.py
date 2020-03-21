from configparser import RawConfigParser
from datetime import datetime
import sys
import importlib.util
from pathlib import Path
from .rtfits import Fits


class RT1_configparser(object):
    def __init__(self, configpath):
        self.configpath = configpath
        # setup config (allow empty values -> will result in None)
        self.config = RawConfigParser(allow_no_value=True)
        # avoid converting uppercase characters
        # (see https://stackoverflow.com/a/19359720/9703451)
        self.config.optionxform = str

        # read config file
        self.cfg = self.config.read(self.configpath)

        # keys that will be converted to int, float or bool
        self.lsq_parse_props = dict(section = 'least_squares_kwargs',
                                    int_keys = ['verbose', 'max_nfev'],
                                    float_keys = ['ftol', 'gtol', 'xtol'])

        self.fitargs_parse_props = dict(section = 'fits_kwargs',
                                        bool_keys = ['sig0', 'dB', 'int_Q'],
                                        int_keys= ['verbose'],
                                        list_keys = [])


    def _parse_dict(self, section, int_keys=[], float_keys=[], bool_keys=[],
                    list_keys=[]):
        '''
        a function to convert the parsed string values to int, float or bool
        (any additional values will be left unchanged)

        Parameters
        ----------
        section : str
            the name of the section in the config-file.
        int_keys : list
            keys that should be converted to int.
        float_keys : list
            keys that should be converted to float.
        bool_keys : list
            keys that should be converted to bool.

        Returns
        -------
        parsed_dict : dict
            a dict with the converted values.

        '''

        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            if key in float_keys:
                val = inp.getfloat(key)
            elif key in int_keys:
                val = inp.getint(key)
            elif key in bool_keys:
                val = inp.getboolean(key)
            elif key in list_keys:
                #assert inp[key].startswith('['), f'{key}  must start with "[" '
                #assert inp[key].endswith(']'), f'{key} must end with "]" '
                #val = inp[key][1:-1].replace(' ', '').split(',')
                if inp[key] is None:
                    val = []
                else:
                    val = inp[key].replace(' ', '').split(',')
                    val = [i for i in val if len(i) > 0]
            else:
                val = inp[key]

            parsed_dict[key] = val
        return parsed_dict


    def _parse_V_SRF(self, section):

        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            val = None
            if key == 'ncoefs':
                val = inp.getint(key)
            else:
                # try to convert floats, if it fails return the string
                try:
                    val = inp.getfloat(key)
                except:
                    val = inp[key]

            parsed_dict[key] = val

        return parsed_dict


    def _parse_defdict(self, section):
        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            val = inp[key]

            if val.startswith('[') and val.endswith(']'):
                val = val[1:-1].replace(' ', '').split(',')
            else:
                assert False, f'the passed defdict for {key} is not a list'

            # convert values
            parsed_val = []
            if val[0] == 'True':
                parsed_val += [True]
            elif val[0] == 'False':
                parsed_val += [False]
            else:
                assert False, (f'the passed first defdict-argument of {key} ' +
                               f'({val[0]}) must be either "True" or "False"')

            if val[1] == 'auxiliary':
                parsed_val += [val[1]]
            else:
                parsed_val += [float(val[1])]

            if len(val) > 2:
                if val[2] == 'None':
                    parsed_val += [None]
                else:
                    parsed_val += [val[2]]

                parsed_val += [([float(val[3])],
                                [float(val[4])])]

                try:
                    interp_Q = val[5]
                    if interp_Q == 'True':
                        parsed_val += [True]
                    else:
                        parsed_val += [False]
                except:
                    parsed_val += [False]


            parsed_dict[key] = parsed_val

        return parsed_dict


    def get_config(self):
        lsq_kwargs = self._parse_dict(**self.lsq_parse_props)

        fits_kwargs = self._parse_dict(**self.fitargs_parse_props)
        defdict = self._parse_defdict('defdict')
        set_V_SRF = dict(V_props = self._parse_V_SRF('RT1_V'),
                         SRF_props = self._parse_V_SRF('RT1_SRF'))

        return dict(lsq_kwargs=lsq_kwargs,
                    defdict=defdict,
                    set_V_SRF=set_V_SRF,
                    fits_kwargs=fits_kwargs)


    def get_fitobject(self):
        cfg = self.get_config()

        rt1_fits = Fits(dataset=None, defdict=cfg['defdict'],
                        set_V_SRF=cfg['set_V_SRF'],
                        lsq_kwargs=cfg['lsq_kwargs'], **cfg['fits_kwargs'])

        return rt1_fits


    def get_modules(self):
        '''
        programmatically import a module as described here:
        https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

        Returns
        -------
        processmodules : dict
            a dict with the imported moduels

        '''

        processmodules = dict()
        for key, val in self.config['LOAD_MODULES'].items():
            if key.startswith('relative__'):
                dict_key = key[10:]
                if '..' in val.strip():
                    try:
                        location = (Path(self.configpath).parent
                                    / Path(val.strip())).resolve()
                    except Exception as ex:
                        print(f'{key} could not be resolved:', ex)
                        location = (Path(self.configpath).parent
                                    / Path(val.strip()))
                else:
                        location = (Path(self.configpath).parent
                                    / Path(val.strip()))
            else:
                dict_key = key
                location = Path(val.strip())

            assert location.suffix == '.py', f'{location} is not a .py file!'
            spec = importlib.util.spec_from_file_location(name=location.stem,
                                                          location=location)

            foo = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = foo
            spec.loader.exec_module(foo)

            processmodules[dict_key] = foo

        return processmodules


    def get_process_specs(self):
        inp = self.config['PROCESS_SPECS']

        process_specs = dict()
        for key, val in inp.items():
            if key.startswith('datetime__'):
                try:
                    date = dict(zip(['t0', 'fmt'],
                                    [i.strip() for i in val.split(',')]))
                    if 'fmt' not in date:
                        date['fmt'] = '%Y-%m-%d %H:%M:%S.%f'
                    process_specs[key[10:]] = datetime.strptime(date['t0'],
                                                           date['fmt'])
                except Exception:
                    print('date could not be converted... ',
                          'original string-value returned')
                    process_specs[key[10:]] = val.strip()

            elif key.startswith('path__'):
                process_specs[key[6:]] = Path(val.strip())
            elif key.startswith('path_relative__'):
                # resolve the path in case .. syntax is used to traverse dirs
                if '..' in val.strip():
                    try:
                        process_specs[key[15:]] = (Path(self.configpath).parent
                                                   / Path(val.strip())
                                                   ).resolve()
                    except Exception as ex:
                        print(f'{key} could not be resolved:' ,
                              ex)
                        process_specs[key[15:]] = (Path(self.configpath).parent
                                                   / Path(val.strip())
                                                   )
                else:
                        process_specs[key[15:]] = (Path(self.configpath).parent
                                                   / Path(val.strip()))



            elif key.startswith('bool__'):
                process_specs[key[6:]] = inp.getboolean(key)
            elif key.startswith('float__'):
                process_specs[key[7:]] = inp.getfloat(key)
            elif key.startswith('int__'):
                process_specs[key[5:]] = inp.getint(key)
            else:
                process_specs[key] = val


        return process_specs
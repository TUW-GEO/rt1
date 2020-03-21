import os
# disable numpy's internal multithreading to avoid parallel overhead
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from rt1.configparser import RT1_configparser
from rt1.processing_config import rt1_processing_config

import pandas as pd
import numpy as np


# subclass the configuration and add a reader
class processing_cfg(rt1_processing_config):
    def __init__(self, config_path, **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path


    def reader(self, **reader_arg):
        # initialize a reader
        if reader_arg['gpi'] in [1,2,3]:
            index = pd.date_range('1.1.2017', '1.1.2018', freq='D')
            ndata = len(index)
            df = pd.DataFrame(dict(sig=np.random.rand(ndata),
                                   inc=np.random.rand(ndata)),
                              index)
        if reader_arg['gpi'] == 4:
            df = pd.DataFrame()

        return df


    def run_procesing(self, reader_args, ncpu):

        # get fit object by using a config-file
        config = RT1_configparser(self.config_path)
        rt1_fits = config.get_fitobject()

        res = rt1_fits.processfunc(ncpu=ncpu,
                                   reader_args=reader_args,
                                   lsq_kwargs=None,
                                   pool_kwargs=None,
                                   reader=self.reader,
                                   preprocess=self.preprocess,
                                   postprocess=self.postprocess,
                                   exceptfunc=self.exceptfunc,
                                   finaloutput=self.finaloutput
                                   )


def run(config_path, reader_args, ncpu):

    cfg = RT1_configparser(config_path)
    spec = cfg.get_process_specs()

    proc = processing_cfg(config_path=config_path,
                          save_path=spec['save_path'],
                          dumpfolder=spec['dumpfolder'],
                          error_dumpfolder=spec['dumpfolder'],
                          finalout_name=spec['finalout_name'])

    proc.run_procesing(reader_args=reader_args, ncpu=ncpu)
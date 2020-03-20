# disable numpy's internal multithreading to avoid parallel overhead
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from rt1.configparser import RT1_configparser
from rt1.processing_config import rt1_processing_config

import pandas as pd
import numpy as np
import multiprocessing as mp
import unittest


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

class TestRTfits(unittest.TestCase):
    def test_parallel_processing(self):
        ncpu = 4
        config_path = os.path.join(os.path.dirname(__file__), 'test_config.ini')
        finalout_name = 'results.h5'
        save_path = os.path.join(os.path.dirname(__file__), 'proc_test')
        dumpfolder='dump01'

        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

        proc = processing_cfg(config_path=config_path,
                              save_path=save_path,
                              dumpfolder=dumpfolder,
                              error_dumpfolder=dumpfolder,
                              finalout_name=finalout_name)

        proc.run_procesing(reader_args=reader_args,
                           ncpu=ncpu)

if __name__ == "__main__":
    unittest.main()




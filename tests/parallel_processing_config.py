import os
# disable numpy's internal multithreading to avoid parallel overhead
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from rt1.processing_config import rt1_processing_config

import pandas as pd
import numpy as np


# subclass the configuration and add a reader
class processing_cfg(rt1_processing_config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reader(self, **reader_arg):

        self.check_dump_exists(reader_arg)

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
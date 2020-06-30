import sys
from pathlib import Path
sys.path.append(str(Path('H:/python_modules/rt_model_python/rt1')))
import unittest
from pathlib import Path
from rt1.rtprocess import RTprocess, RTresults
import shutil

import warnings
warnings.simplefilter('ignore')

# use "test_0_---"   "test_1_---"   to ensure test-order

class TestRTfits(unittest.TestCase):
    def test_0_parallel_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        proc = RTprocess(config_path, copy=True, autocontinue=True)

        proc.run_processing(ncpu=4, reader_args = reader_args)
        # run again to check what happens if files already exist
        proc.run_processing(ncpu=4, reader_args = reader_args)

        #----------------------------------------- check if files have been copied
        assert Path('tests/proc_test/dump01/cfg').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/results').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/dumps').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/cfg/test_config.ini').exists(), 'copying did not work'
        assert Path('tests/proc_test/dump01/cfg/parallel_processing_config.py').exists(), 'copying did not work'


    def test_1_rtresults(self):

        results = RTresults('tests/proc_test')
        assert hasattr(results, 'dump01'), 'dumpfolder not found by RTresults'
        fit = results.dump01.load_fit()
        cfg = results.dump01.load_cfg()
        dumpfiles = [i for i in results.dump01.dump_files]

        # remove the save_path directory
        print('deleting save_path directory...')
        #print('\n'.join([str(i) for i in cfg.get_process_specs()["save_path"].rglob('*')]))
        shutil.rmtree(results._parent_path)


    def test_2_single_core_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        proc = RTprocess(config_path, copy=True, autocontinue=True)
        proc.run_processing(ncpu=1, reader_args = reader_args)

        #----------------------------------------- check if files have been copied
        assert Path('tests/proc_test/dump01/cfg').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/results').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/dumps').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/cfg/test_config.ini').exists(), 'copying did not work'
        assert Path('tests/proc_test/dump01/cfg/parallel_processing_config.py').exists(), 'copying did not work'

        # remove the save_path directory
        print('deleting save_path directory...')
        #print('\n'.join([str(i) for i in cfg.get_process_specs()["save_path"].rglob('*')]))
        shutil.rmtree(proc.dumppath.parent)


if __name__ == "__main__":
    unittest.main()




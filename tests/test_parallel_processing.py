import os
from pathlib import Path
import unittest
import unittest.mock as mock
from rt1.rtprocess import RTprocess, RTresults

import warnings
warnings.simplefilter('ignore')


# use "test_0_---"   "test_1_---"   to ensure test-order


class TestRTfits(unittest.TestCase):

    def test_0_parallel_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        proc = RTprocess(config_path, autocontinue=True)

        proc.run_processing(ncpu=4, reader_args=reader_args)

        # run again to check what happens if files already exist
        proc.run_processing(ncpu=4, reader_args=reader_args)

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

        with self.assertRaises(AssertionError):
            # TODO implement netcdf export and properly test netcdf's !
            ncfile = results.dump01.load_nc()
        with self.assertRaises(AssertionError):
            # TODO implement netcdf export and properly test netcdf's !
            ncfile = results.dump01.NetCDF_variables

        # remove the save_path directory
        #print('deleting save_path directory...')
        #shutil.rmtree(results._parent_path)


    def test_2_single_core_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        # mock inputs as shown here: https://stackoverflow.com/a/37467870/9703451
        with self.assertRaises(SystemExit):
            with mock.patch('builtins.input', side_effect=['N']):
                proc = RTprocess(config_path, autocontinue=False)
                proc.setup()

        with self.assertRaises(SystemExit):
            with mock.patch('builtins.input', side_effect=['REMOVE', 'N']):
                proc = RTprocess(config_path, autocontinue=False)
                proc.setup()

        with mock.patch('builtins.input', side_effect=['REMOVE', 'Y']):
            proc = RTprocess(config_path, autocontinue=False)
            proc.setup()
        assert len(list(Path('tests/proc_test/dump01/dumps').iterdir())) == 0, 'user-input REMOVE did not work'

        with mock.patch('builtins.input', side_effect=['REMOVE', 'Y']):
            proc.run_processing(ncpu=1, reader_args = reader_args, copy=False)

        #----------------------------------------- check if files have been copied
        assert Path('tests/proc_test/dump01/cfg').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/results').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test/dump01/dumps').exists(), 'folder-generation did not work'
        assert not Path('tests/proc_test/dump01/cfg/test_config.ini').exists(), 'NOT copying did not work'
        assert not Path('tests/proc_test/dump01/cfg/parallel_processing_config.py').exists(), 'NOT copying did not work'

        # remove the save_path directory
        #print('deleting save_path directory...')
        #shutil.rmtree(proc.dumppath.parent)


    def test_3_parallel_processing_init_kwargs(self):
        # test overwriting keyword-args from .ini file
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        proc = RTprocess(config_path, autocontinue=True,
                         init_kwargs=dict(
                             path__save_path = 'tests/proc_test2',
                             dumpfolder='dump02')
                         )

        proc.run_processing(ncpu=4, reader_args = reader_args)

        #----------------------------------------- check if files have been copied
        assert Path('tests/proc_test2/dump02/cfg').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test2/dump02/results').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test2/dump02/dumps').exists(), 'folder-generation did not work'
        assert Path('tests/proc_test2/dump02/cfg/test_config.ini').exists(), 'copying did not work'
        assert Path('tests/proc_test2/dump02/cfg/parallel_processing_config.py').exists(), 'copying did not work'


if __name__ == "__main__":
    unittest.main()

    #stop_log_to_file()

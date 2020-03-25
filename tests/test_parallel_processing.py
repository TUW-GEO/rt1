import unittest
from pathlib import Path
from rt1.rtparse import RT1_configparser
import shutil

class TestRTfits(unittest.TestCase):
    def test_parallel_processing(self):
        ncpu = 4
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        cfg = RT1_configparser(config_path)
        run = cfg.get_modules()['processfuncs'].run

        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

        run(config_path=config_path,
            reader_args=reader_args,
            ncpu=ncpu)

        # run again to check what happens if files already exist
        run(config_path=config_path,
            reader_args=reader_args,
            ncpu=ncpu)

        # remove the save_path directory
        print('deleting save_path directory...')
        #print('\n'.join([str(i) for i in cfg.get_process_specs()["save_path"].rglob('*')]))
        shutil.rmtree(cfg.get_process_specs()['save_path'])


    def test_single_core_processing(self):
        ncpu = 1
        config_path = Path(__file__).parent.absolute() / 'test_config.ini'

        cfg = RT1_configparser(config_path)
        run = cfg.get_modules()['processfuncs'].run

        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

        run(config_path=config_path,
            reader_args=reader_args,
            ncpu=ncpu)

        # run again to check what happens if files already exist
        run(config_path=config_path,
            reader_args=reader_args,
            ncpu=ncpu)

        # remove the save_path directory
        print('deleting save_path directory...')
        #print('\n'.join([str(i) for i in cfg.get_process_specs()["save_path"].rglob('*')]))
        shutil.rmtree(cfg.get_process_specs()['save_path'])


if __name__ == "__main__":
    unittest.main()




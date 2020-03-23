import unittest
from pathlib import Path
from rt1.rtparse import RT1_configparser

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


if __name__ == "__main__":
    unittest.main()




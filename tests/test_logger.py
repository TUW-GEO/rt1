# -*- coding: utf-8 -*-
"""
test configparser

a quick check if a standard config-file is read correctly
"""
import unittest
from pathlib import Path
from rt1 import log, set_log_handler_level, start_log_to_file, stop_log_to_file


class TestCONFIGPARSER(unittest.TestCase):

    def test_logging(self):
        assert 'rt1_consolehandler' in [i.name for i in log.handlers], (
            "there's no rt1_consolehandler")

        set_log_handler_level(1)

        assert {i.name:
                i for i in log.handlers}['rt1_consolehandler'].level == 1, (
                    'setting log-level for consolehandler did not work')

        # start logging to a file
        logfilepath = Path().absolute() / 'log_test.log'
        start_log_to_file(logfilepath, level=10)

        log.info('a testlog')
        log.debug('a debuglog')

        with open(logfilepath, 'r') as file:
            l = file.readlines()
            assert 'INFO' in l[-2] and l[-2].strip().endswith("a testlog"), (
                f'the log-message {l} is not correct!')

            assert 'DEBUG' in l[-1] and l[-1].strip().endswith("a debuglog"), (
                f'the log-message {l} is not correct!')

        assert logfilepath.exists(), 'the logfile has not been generated!'

        assert 'rt1_filehandler' in [i.name for i in log.handlers], (
            "there's no rt1_filehandler")

        # the file can not be removed if it is not closed!
        with self.assertRaises(PermissionError):
            logfilepath.unlink()

        # close the file, now removing should be possible
        stop_log_to_file()
        logfilepath.unlink()

        assert not logfilepath.exists(), 'the logfile has not been removed!'




if __name__ == "__main__":
    unittest.main()
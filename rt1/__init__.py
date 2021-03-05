"""Import module for RT1 module"""

import logging
import logging.handlers
import sys
from textwrap import indent
import multiprocessing as mp
from .general_functions import groupby_unsorted

__version__ = "1.3"
__author__ = "Raphael Quast"


# add a logging-level "Prog." that is used to report progess (1 above INFO)
_PROGRESS_LEVEL_NUM = 21
logging.addLevelName(_PROGRESS_LEVEL_NUM, "PROG.")


def progress(self, message, *args, **kws):
    if self.isEnabledFor(_PROGRESS_LEVEL_NUM):
        self._log(_PROGRESS_LEVEL_NUM, message, args, **kws)


logging.Logger.progress = progress


def set_log_handler_level(level, name="rt1_consolehandler"):
    """
    set the level of the logger
    Parameters
    ----------
    level : the desired logging-level
        50 CRITICAL
        40 ERROR
        30 WARNING
        21 PROGRESS
        20 INFO
        10 DEBUG

    name : str, optional
        The name of the handler to address.

            - 'rt1_consolehandler'
            - 'rt1_filehandler'

        The default is 'rt1_consolehandler'.

    """

    log = setup_logger()

    h = [val for val in log.handlers if val.name == name][0]
    h.setLevel(level)


def start_log_to_file(path, name="rt1_filehandler", level=logging.INFO):
    """
    forward all log-messages to a file
    to close the file use the `stop_log_to_file()` method!

    Notice: this is not process-save and therefore extensive multiprocessing
    can lead to an unexpected scrambling of the messges!
    -> rtprocess.run_processing writes logs to the "cfg" folder in a
    process-save way!

    Parameters
    ----------
    path : str or pathlib.Path
        the file-path where the file should be stored (including filename!)
    name : str, optional
        the name of the filehandler. The default is 'rt1_filehandler'.
    level : int, optional
        the log-level attached to the handler. The default is logging.INFO.
    """
    try:
        # check if file-handler already exists, and if yes stop and remove it
        stop_log_to_file(name=name)

        log = setup_logger()
        # get formatting from consolehandler (always present)
        hc = [val for val in log.handlers if val.name == "rt1_consolehandler"][0]

        # setup a new filehandler
        fh = logging.FileHandler(path, "a")
        fh.setFormatter(hc.formatter)
        fh.set_name(name)
        # initialize the file-handler with level 1 to get all infos
        fh.setLevel(level)

        log.addHandler(fh)

        log.debug(
            f"log-file for handler {name} added at location" + f' "{fh.baseFilename}"!'
        )

    except IndexError as err:
        log.exception(err)


def stop_log_to_file(name="rt1_filehandler"):
    """
    stop and remove an existing logging handler

    Parameters
    ----------
    name : str, optional
        the name of the handler. The default is 'rt1_filehandler'.
    """
    log = setup_logger()
    handlers = groupby_unsorted(log.handlers, key=lambda x: x.name)

    # remove existing rt1_filehandler
    if name in handlers:
        for i, h in enumerate(handlers[name]):
            h.close()
            log.removeHandler(h)
            log.debug(
                f'{i + 1} existing logfile-handler "{name}" at'
                + f' "{h.baseFilename}" closed and removed!'
            )


class WrappedFixedIndentingLog(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", indent=4):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

        self._indent = indent

    def format(self, record):
        return indent(super().format(record), " " * self._indent).strip()


def _get_logger_formatter(simple=True):

    if simple is False:
        logfmt = (
            "%(asctime)s - "
            + mp.current_process().name
            + ": "
            + "%(levelname)-8s ("
            + "%(filename)s:%(lineno)d - "
            + "%(funcName)s)"
            + "\n"
            + "%(message)s"
        )
    else:
        logfmt = (
            "%(asctime)s - "
            +
            # (mp.current_process().name + ':').ljust(21) +
            "%(processName)-21s"
            + "%(levelname)-8s"
            + "%(message)s"
        )

    formatter = WrappedFixedIndentingLog(logfmt, indent=51, datefmt="%Y-%m-%d %H:%M:%S")
    return formatter


def setup_logger(
    log_name="rt1",
    console_out=True,
    console_level=logging.WARNING,
    simple=True,
):

    logger = logging.getLogger(log_name)

    # check if the logger has a Handler, if yes, it's an already existing
    # one and so we don't want to re-create it!!
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = _get_logger_formatter(simple)

    if console_out is True:
        # setup a console-handler that prints to sys.stdout
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(console_level)
        sh.setFormatter(formatter)
        sh.set_name("rt1_consolehandler")
        logger.addHandler(sh)

    logger.debug(f"Setup logger: {log_name}")

    return logger


# initialize a logger for rt1
log = setup_logger(console_out=True, simple=True, log_name="rt1")

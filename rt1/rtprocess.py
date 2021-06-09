import multiprocessing as mp
from timeit import default_timer
from datetime import datetime, timedelta
from itertools import repeat, islice
import ctypes
import sys
import os
import traceback
from textwrap import dedent

from pathlib import Path
import shutil

import pandas as pd
from numpy.random import choice

from .general_functions import dt_to_hms, update_progress, groupby_unsorted
from .rtparse import RT1_configparser
from .rtfits import load, MultiFits
from . import log, _get_logger_formatter
import logging


try:
    import xarray as xar
except ModuleNotFoundError:
    log.info(
        "xarray could not be imported, "
        + "NetCDF-features of RT1_results will not work!"
    )


def _confirm_input(msg="are you sure?", stopmsg="STOP", callbackdict=None):
    """
    Parameters
    ----------
    msg : str, optional
        The prompt message. The default is 'are you sure?'.
    stopmsg : str, optional
        The message displayed if the answer is False. The default is 'STOP'.
    callbackdict : dict, optional
        A dict of the form {key: [msg, callback]} . The default is None.
    """
    default_answers = {"YES": True, "Y": True, "NO": False, "N": False}

    # prompt input
    inp = str(input(msg)).upper()

    # in case the input is found in the callback-dict, ask for confirmation
    # and execute the callback
    if inp in callbackdict:
        cbmsg, cb = callbackdict[inp]
        inp2 = str(input(cbmsg)).upper()
        print()
        answer2 = default_answers.get(inp2, False)
        if answer2 is False:
            print(stopmsg)
            sys.exit(0)
            return
        cb()
    else:
        answer = default_answers.get(inp, False)
        if answer is False:
            print(stopmsg)
            print()
            sys.exit(0)
            return


def _make_folderstructure(save_path, subfolders):
    if save_path is not None:
        # generate "save_path" directory if it does not exist
        if not save_path.exists():
            log.info(f'"{save_path}"\ndoes not exist... creating directory')
            save_path.mkdir(parents=True, exist_ok=True)

        for folder in subfolders:
            # generate subfolder if it does not exist
            mkpath = save_path / folder
            if not mkpath.exists():
                log.info(f'"{mkpath}"\ndoes not exist... creating directory')
                mkpath.mkdir(parents=True, exist_ok=True)


def _setup_cnt(N_items, ncpu):
    if ncpu > 1:
        manager = mp.Manager()
        lock = manager.Lock()
        p_totcnt = manager.Value(ctypes.c_ulonglong, 0)
        p_meancnt = manager.Value(ctypes.c_ulonglong, 0)
        p_time = manager.Value(ctypes.c_float, 0)
    else:
        # don't use a manager if single-core processing is used
        p_totcnt = mp.Value(ctypes.c_ulonglong, 0)
        p_meancnt = mp.Value(ctypes.c_ulonglong, 0)
        p_time = mp.Value(ctypes.c_float, 0)
        lock = None

    process_cnt = [p_totcnt, p_meancnt, N_items, p_time, ncpu, lock]
    return process_cnt


def _start_cnt():
    return default_timer()


def _increase_cnt(process_cnt, start, err=False):
    if process_cnt is None:
        return

    p_totcnt, p_meancnt, p_max, p_time, p_ncpu, lock = process_cnt
    if lock is not None:
        # ensure that only one process is allowed to write simultaneously
        lock.acquire()
    try:
        if err is False:
            end = default_timer()
            # increase the total counter
            p_totcnt.value += 1

            # update the estimate of the mean time needed to process a site
            p_time.value = (p_meancnt.value * p_time.value + (end - start)) / (
                p_meancnt.value + 1
            )
            # increase the mean counter
            p_meancnt.value += 1
            # get the remaining time and update the progressbar
            remain = timedelta(seconds=(p_max - p_totcnt.value) / p_ncpu * p_time.value)
            d, h, m, s = dt_to_hms(remain)

            msg = update_progress(
                p_totcnt.value,
                p_max,
                title=f"approx. {d} {h:02}:{m:02}:{s:02} remaining",
                finalmsg=(
                    "finished! "
                    + f"({p_max} [{p_totcnt.value - p_meancnt.value}] fits)"
                ),
                progress2=p_totcnt.value - p_meancnt.value,
            )

        else:
            # only increase the total counter
            p_totcnt.value += 1
            if p_meancnt.value == 0:
                title = f"{'estimating time ...':<28}"
            else:
                # get the remaining time and update the progressbar
                remain = timedelta(
                    seconds=(p_max - p_totcnt.value) / p_ncpu * p_time.value
                )
                d, h, m, s = dt_to_hms(remain)
                title = f"approx. {d} {h:02}:{m:02}:{s:02} remaining"

            msg = update_progress(
                p_totcnt.value,
                p_max,
                title=title,
                finalmsg=(
                    "finished! "
                    + f"({p_max} [{p_totcnt.value - p_meancnt.value}] fits)"
                ),
                progress2=p_totcnt.value - p_meancnt.value,
            )

            # log to file if an error occured during processing
            log.debug(msg.strip())

        if lock is not None:
            # release the lock
            lock.release()
    except Exception:
        msg = "???"
        if lock is not None:
            # release the lock in case an error occured
            lock.release()
        pass

    sys.stdout.write(msg)
    sys.stdout.flush()


class RTprocess(object):
    def __init__(
        self,
        config_path=None,
        autocontinue=False,
        copy=True,
        proc_cls=None,
        parent_fit=None,
        init_kwargs=None,
    ):
        """
        A class to perform parallelized processing.

        Parameters
        ----------
        config_path : str, optional
            The path to the config-file to be used. The default is None.
        autocontinue : bool, optional
            indicator if user-input should be raised (True) in case the
            dump-folder already exists. The default is False.
        copy : bool
            indicator if '.ini' files and modules should be copied to
            (and imported from) the dumppath/cfg folder or not.
            The default is True.
        proc_cls : class, optional
            the processing-class. (if None it will be imported use the
            'module__processing_cfg' from the 'CONFIGFILES' section of
            the .ini file).

            NOTICE:
                All parsed arguments from the 'PROCESS_SPECS' section of
                the config-file will be used in the initialization of
                the processing class! The call-signature is equivalent to:

                    >>> from rt1.rtparse import RT1_configparser
                    ... cfg = RT1_configparser(config_path)
                    ... proc_cls = proc_cls(**cfg.get_process_specs(),
                                            **init_kwargs)

            For details on how to specify a processing-class, have look at
            the `rt1_processing_config` class in `rt1.processing_config`.
            The default is None.
        parent_fit : rt1.rtfits.Fits, optional
            a parent fit-object. (if None it will be set-up
            using the specifications of the .ini file) The default is None.
        init_kwargs : dict, optional
            Additional keyword-arguments passed to the initialization of the
            'proc_cls' class. (used to append-to or partially overwrite
            definitions passed via the config-file). The default is None.
        """
        assert config_path is not None, "Please provide a config-path!"
        self.config_path = Path(config_path)

        assert self.config_path.exists(), (
            f"the file {self.config_path} " + "does not exist!"
        )

        self.autocontinue = autocontinue

        self._postprocess = True

        self.copy = copy

        self._proc_cls = proc_cls
        self._parent_fit = parent_fit
        if init_kwargs is None:
            self.init_kwargs = dict()
        else:

            assert all(
                [isinstance(i, str) for i in init_kwargs.values()]
            ), 'the values of "init_kwargs" MUST be strings !'
            self.init_kwargs = init_kwargs

    def _listener_process(self, queue):

        # adapted from https://docs.python.org/3.7/howto/logging-cookbook.html
        # logging-to-a-single-file-from-multiple-processes
        if not hasattr(self, "dumppath"):
            log.warning("no dumppath specified, log is not saved!")
            return

        datestring = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        path = self.dumppath / "cfg" / f"RT1_process({datestring}).log"

        log.debug("listener process started for path: " + str(path))

        log2 = logging.getLogger()
        log2.setLevel(1)
        h = logging.FileHandler(path)
        h.setFormatter(_get_logger_formatter())
        h.setLevel(1)
        log2.addHandler(h)

        while True:
            try:
                record = queue.get()
                if record is None:  # quit listener if the message is None
                    break
                log2.handle(record)  # ...No level or filter logic applied
            except Exception:
                print("Whoops! Problem:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    @staticmethod
    def _worker_configurer(queue, loglevel=10):

        log.debug("configuring worker... ")
        log.setLevel(loglevel)
        h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
        h.setLevel(loglevel)
        h.name = "rtprocessing_queuehandler"
        log.addHandler(h)

        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger
        warnings_logger.addHandler(h)

    def override_config(self, override=True, **kwargs):
        """
        set the init_kwargs to change parts of the specifications provided in the
        used ".ini"-file.

        by default, already present `init_kwargs` will be overritten!
        (use override=False to update `init_kwargs`)

        kwargs must be passed in the form:

        >>> section = dict( key1 = val1,
        >>>                 key2 = val2, ... )

        Parameters
        ----------
        override : bool, optional
            indicator if "init_kwargs" should be overritten (True) or updated (False).
            The default is True
        """
        cfg = RT1_configparser(self.config_path)

        init_kwargs = dict()
        for section, init_defs in kwargs.items():
            assert section in cfg.config, (
                f"the section '{section}' you provided in init_kwargs "
                + " is not present in the .ini file!"
            )

            init_kwargs[section] = {
                str(key): str(val) for key, val in init_defs.items()
            }

        if override is False:
            for section, init_defs in init_kwargs.items():
                init_kwargs[section] = {
                    **self.init_kwargs.get(section, {}),
                    **init_defs,
                }

        self.init_kwargs = init_kwargs

    def _remove_dumpfolder(self):
        # try to close existing filehandlers to avoid
        # errors if the files are located in the folder
        # if they are actually in the cfg folder, delete
        # the corresponding handler as well!
        for h in log.handlers:
            try:
                h.close()
                if Path(h.baseFilename).parent.samefile(self.dumppath / "cfg"):
                    log.removeHandler(h)
            except Exception:
                pass

        # remove folderstructure
        shutil.rmtree(self.dumppath)

        log.info(f'"{self.dumppath}"' + "\nhas successfully been removed.\n")

    def setup(self):
        """
        perform necessary tasks to run a processing-routine
          - initialize the folderstructure (only from MainProcess!)
          - copy modules and .ini files (if copy=True) (only from MainProcess!)
          - load modules and set parent-fit-object
        """
        try:
            self.cfg = RT1_configparser(self.config_path)
        except Exception:
            if mp.current_process().name == "MainProcess":
                log.warning("no valid config file was found")
            return


        # update specs with init_kwargs
        for section, init_defs in self.init_kwargs.items():
            assert (
                section in self.cfg.config
            ), f"the init_kwargs section {section} is not present in the .ini file!"

            for key, val in init_defs.items():
                if key in self.cfg.config[section]:
                    log.warning(
                        f'"{key} = {self.cfg.config[section][key]}" '
                        + "will be overwritten by the definition provided via "
                        + f'"init_kwargs[{section}]": "{key} = {val}" '
                    )
                    # update the parsed config (for import of modules etc.)
                    self.cfg.config[section][key] = val

        specs = self.cfg.get_process_specs()
        self.dumppath = specs["save_path"] / specs["dumpfolder"]

        if mp.current_process().name == "MainProcess":
            # init folderstructure and copy files only from main process
            if self.autocontinue is False:

                if self.dumppath.exists():
                    _confirm_input(
                        msg=(
                            f'the path \n "{self.dumppath}"\n'
                            + " already exists..."
                            + "\n- to continue type YES or Y"
                            + "\n- to abort type NO or N"
                            + "\n- to remove the existing directory and "
                            + "all subdirectories type REMOVE \n \n"
                        ),
                        stopmsg='aborting due to manual input "NO"',
                        callbackdict={
                            "REMOVE": [
                                (
                                    f'\n"{self.dumppath}"\n will be removed!'
                                    + " are you sure? (y, n): "
                                ),
                                self._remove_dumpfolder,
                            ]
                        },
                    )

            # initialize the folderstructure
            _make_folderstructure(
                specs["save_path"] / specs["dumpfolder"],
                ["results", "cfg", "dumps"],
            )

        # copy modules and config-files and ensure that they are loaded from the
        # right direction (NO files will be overwritten!!)
        # copying is only performed in the main-process!
        if self.copy is True:
            self._copy_cfg_and_modules()

        # get the name of the processing-module and the processing-class
        proc_module_name = specs.get("processing_cfg_module", "processing_cfg")
        proc_class_name = specs.get("processing_cfg_class", "processing_cfg")

        # load ALL modules to ensure that the importer finds them
        procmodule = self.cfg.get_all_modules(load_copy=self.copy)[proc_module_name]

        # only import proc_cls if it is not yet set
        # (to allow overriding already set properties during runtime)
        if self._proc_cls is None:

            log.debug(
                f'processing config class "{proc_class_name}"'
                + f'  will be imported from \n"{procmodule}"'
            )

            self.proc_cls = getattr(procmodule, proc_class_name)(**specs)
        else:
            self.proc_cls = self._proc_cls

        # get the parent fit-object
        if self._parent_fit is None:
            self.parent_fit = self.cfg.get_fitobject()
        else:
            self.parent_fit = self._parent_fit

        # check if all necessary functions are defined in the  processing-class
        for key in [
            "preprocess",
            "reader",
            "postprocess",
            "finaloutput",
            "exceptfunc",
        ]:
            assert hasattr(
                self.proc_cls, key
            ), f"a function {key}() MUST be provided in the config-class!"

        assert (
            self.parent_fit is not None
        ), "you MUST provide a valid config-file or a parent_fit-object!"

    def _copy_cfg_and_modules(self):
        # if copy is True, copy the config-file and re-import the cfg
        # from the copied file
        copypath = self.dumppath / "cfg" / self.cfg.configpath.name
        if mp.current_process().name == "MainProcess":
            if copypath.exists() and len(self.init_kwargs) == 0:

                log.warning(
                    f'the file "{Path(*copypath.parts[-3:])} "'
                    + "already exists... NO copying is performed and the "
                    + "existing one is used!\n"
                )

                log.debug(
                    f'{copypath.name} imported from \n"{copypath}"\n'
                )
            else:
                if len(self.init_kwargs) == 0:
                    # if no init_kwargs have been provided, copy the
                    # original file
                    shutil.copy(self.cfg.configpath, copypath.parent)
                    log.info(
                        f'"{self.cfg.configpath.name}" copied to\n'
                        + f'"{copypath.parent}"'
                    )
                else:
                    # if init_kwargs have been provided, write the updated
                    # config to the folder
                    with open(copypath, "w") as file:
                        self.cfg.config.write(file)

                    log.warning(
                        f'the config-file "{self.cfg.configpath}" has been'
                        + " updated with the init_kwargs and saved to"
                        + f'"{copypath}"'
                    )

            # copy modules
            for key, val in self.cfg.config["CONFIGFILES"].items():
                if key.startswith("module__"):
                    modulename = key[8:]

                    module_path = self.cfg.config["CONFIGFILES"][
                        f"module__{modulename}"
                    ]

                    location = Path(module_path.strip())

                    module_copypath = self.dumppath / "cfg" / location.name

                    if module_copypath.exists():
                        log.warning(
                            f'the file "{Path(*module_copypath.parts[-3:])} "'
                            + "already exists... NO copying is performed and the "
                            + "existing one is used!\n"
                        )

                        log.debug(
                            f'{module_copypath.name} imported from \n"{module_copypath}"\n'
                        )

                    else:
                        shutil.copy(location, module_copypath)
                        log.info(f'"{location.name}" copied to \n"{module_copypath}"')
        # remove the config and re-read the config from the copied path
        del self.cfg
        self.cfg = RT1_configparser(copypath)

    def _evalfunc(self, reader_arg=None, process_cnt=None):
        """
        Initialize a Fits-instance and perform a fit.
        (used for parallel processing)

        Parameters
        ----------
        reader_arg : dict
            A dict of arguments passed to the reader.
        process_cnt : list
            A list of shared-memory variables that are used to update the
            progressbar
        Returns
        -------
        The used 'rt1.rtfit.Fits' object or the output of 'postprocess()'
        """

        if process_cnt is not None:
            start = _start_cnt()
        try:
            # if a reader (and no dataset) is provided, use the reader
            read_data = self.proc_cls.reader(reader_arg)
            # check for multiple return values and split them accordingly
            # (any value beyond the first is appended as aux_data)
            if isinstance(read_data, pd.DataFrame):
                dataset = read_data
                aux_data = None
            elif isinstance(read_data, (list, tuple)) and isinstance(
                read_data[0], pd.DataFrame
            ):
                if len(read_data) == 2:
                    dataset, aux_data = read_data
                elif len(read_data) > 2:
                    dataset = read_data[0]
                    aux_data = read_data[1:]
            else:
                raise TypeError(
                    "the first return-value of reader function "
                    + "must be a pandas DataFrame"
                )

            # take proper care of MultiFit objects
            if isinstance(self.parent_fit, MultiFits):
                # set the dataset
                self.parent_fit.set_dataset(dataset)
                # set the aux_data
                if aux_data is not None:
                    self.parent_fit.set_aux_data(aux_data)
                # append reader_arg to all configs
                self.parent_fit.set_reader_arg(reader_arg)

                do_performfit = self.parent_fit.accessor.performfit()
                _ = [doit() for doit in do_performfit.values()]

                # if a post-processing function is provided, return its output,
                # else return None
                if self._postprocess and callable(self.proc_cls.postprocess):
                    ret = dict(
                        self.parent_fit.apply(
                            self.proc_cls.postprocess, reader_arg=reader_arg
                        )
                    )
                else:
                    ret = dict()

                if self._dump_fit and hasattr(self.proc_cls, "dump_fit_to_file"):
                    self.proc_cls.dump_fit_to_file(
                        self.parent_fit, reader_arg, mini=True
                    )

                if process_cnt is not None:
                    _increase_cnt(process_cnt, start, err=False)

            # # initialize a new fits-object and perform the fit
            # if len(self.cfg.config_names) > 0:
            #     # in case multiple configs are provided, evaluate them one by one
            #     ret = []
            #     for cfg_name, parent_fit in self.parent_fit:

            #         if self.dumppath is not None:
            #             self.proc_cls.rt1_procsesing_dumppath = (
            #                 self.dumppath / "dumps" / cfg_name
            #             )

            #         fit = parent_fit.reinit_object(dataset=dataset)
            #         fit.performfit()

            #         # append auxiliary data
            #         if aux_data is not None:
            #             fit.aux_data = aux_data
            #         # append reader_arg
            #         fit.reader_arg = reader_arg

            #         # if a post-processing function is provided, return its output,
            #         # else return None
            #         if self._postprocess and callable(self.proc_cls.postprocess):
            #             ret.append(self.proc_cls.postprocess(fit, reader_arg))
            #         else:
            #             ret.append(None)

            #         # dump a fit-file
            #         # save dumps AFTER prostprocessing has been performed
            #         # (it might happen that some parts change during postprocessing)
            #         if self._dump_fit and hasattr(self.proc_cls, "dump_fit_to_file"):
            #             self.proc_cls.dump_fit_to_file(fit, reader_arg, mini=True)

            #     if process_cnt is not None:
            #         _increase_cnt(process_cnt, start, err=False)

            else:
                fit = self.parent_fit.reinit_object(dataset=dataset)
                fit.performfit()

                # append auxiliary data
                if aux_data is not None:
                    fit.aux_data = aux_data

                # append reader_arg
                fit.reader_arg = reader_arg

                if process_cnt is not None:
                    _increase_cnt(process_cnt, start, err=False)

                # if a post-processing function is provided, return its output,
                # else return None
                if self._postprocess and callable(self.proc_cls.postprocess):
                    ret = self.proc_cls.postprocess(fit, reader_arg)
                else:
                    ret = None

                # dump a fit-file
                # save dumps AFTER prostprocessing has been performed
                # (it might happen that some parts change during postprocessing)
                if self._dump_fit and hasattr(self.proc_cls, "dump_fit_to_file"):
                    self.proc_cls.dump_fit_to_file(fit, reader_arg, mini=True)
            return ret

        except Exception as ex:

            if callable(self.proc_cls.exceptfunc):
                ex_ret = self.proc_cls.exceptfunc(ex, reader_arg)
                if ex_ret is None or ex_ret is False:
                    _increase_cnt(process_cnt, start, err=True)
                else:
                    _increase_cnt(process_cnt, start, err=False)

                return ex_ret
            else:
                raise ex

    def _initializer(self, initializer, queue, *args):
        """
        decorate a provided initializer with the "_worker_configurer()" to
        enable logging from subprocesses while keeping the initializer a
        pickleable function (e.g. avoid using an actual decorator for this)
        (>> returns from provided initializer are returned)
        """

        if queue is not None:
            self._worker_configurer(queue)

        if not mp.current_process().name == "MainProcess":
            # call setup() on each worker-process to ensure that the importer loads
            # all required modules from the desired locations
            log.progress("setting up RTprocess-worker")
            self.setup()

        if initializer is not None:
            res = initializer(*args)
            return res

    def processfunc(
        self,
        ncpu=1,
        print_progress=True,
        reader_args=None,
        pool_kwargs=None,
        preprocess_kwargs=None,
        queue=None,
        dump_fit=True,
        postprocess=True,
    ):
        """
        Evaluate a RT-1 model on a single core or in parallel using

            - a list of datasets or
            - a reader-function together with a list of arguments that
              will be used to read the datasets

        Notice:
            On Windows, if multiprocessing is used, you must protect the call
            of this function via:
            (see for example: )

            >>> if __name__ == '__main__':
                    fit.processfunc(...)

            In order to allow pickling the final rt1.rtfits.Fits object,
            it is required to store ALL definitions within a separate
            file and call processfunc in the 'main'-file as follows:

            >>> from specification_file import fit reader lsq_kwargs ... ...
                if __name__ == '__main__':
                    fit.processfunc(ncpu=5, reader=reader,
                                    lsq_kwargs=lsq_kwargs, ... ...)

        Parameters
        ----------
        ncpu : int, optional
            The number of kernels to use. The default is 1.
        print_progress : bool, optional
            indicator if a progress-bar should be printed to stdout or not
            that looks like this:

            >>> approx. 0 00:00:02 remaining ################------ 3 (2) / 4
            ...
            ... (estimated time day HH:MM:SS)(     progress bar   )( counts )
            ... ( counts ) = finished fits [actually fitted] / total

            The default is True.
        reader_args : list, optional
            A list of dicts that will be passed to the reader-function.
            I `None`, the `reader_args` will be taken from the return of the
            `preprocess()`-function via:

            >>> reader_args = preprocess(**preprocess_kwargs)['reader_args']

            The default is None.
        pool_kwargs : dict, optional
            A dict with additional keyword-arguments passed to the
            initialization of the multiprocessing-pool via:

            >>> mp.Pool(ncpu, **pool_kwargs)

            The default is None.
        preprocess_kwargs : dict, optional
            A dict with keyword-arguments passed to the call of the preprocess
            function via:

            >>> preprocess(**preprocess_kwargs)

            The default is None.
        queue : multiprocessing.Manager().queue
            the queue used for logging
        dump_fit : bool, optional
            indicator if a fit-object should be dumped or not.
            (e.g. by invoking processing_config.dump_fit_to_file() )
            The default is True
        postprocess bool, optional
            indicator if postprocess() and finalout() workflows should be
            executed or not

        Returns
        -------
        res : list
            A list of rt1.rtfits.Fits objects or a list of outputs of the
            postprocess-function.

        """
        self._postprocess = postprocess
        self._dump_fit = dump_fit

        if callable(self.proc_cls.preprocess):
            setupdict = self.proc_cls.preprocess(**preprocess_kwargs)
            if setupdict is None:
                setupdict = dict()
            assert isinstance(
                setupdict, dict
            ), "the preprocess() function must return a dict!"
        else:
            setupdict = dict()

        # check if reader args is provided in setupdict
        if reader_args is None:
            assert "reader_args" in setupdict, (
                'if "reader_args" is not passed directly to processfunc() '
                + ', the preprocess() function must return a key "reader_args"!'
            )

            reader_args = setupdict["reader_args"]
        else:
            assert "reader_args" not in setupdict, (
                '"reader_args" is provided as argument to processfunc() '
                + "AND via the return-dict of the preprocess() function!"
            )

        log.info(f"processing {len(reader_args)} features")

        if "pool_kwargs" in setupdict:
            pool_kwargs = setupdict["pool_kwargs"]

        if pool_kwargs is None:
            pool_kwargs = dict()

        # add provided initializer and queue (used for subprocess-logging)
        # to the initargs and use "self._initializer" as initializer-function
        # Note: this is a pickleable way for decorating the initializer!
        pool_kwargs["initargs"] = [
            pool_kwargs.pop("initializer", None),
            queue,
            *pool_kwargs.pop("initargs", []),
        ]
        pool_kwargs["initializer"] = self._initializer

        # pre-evaluate the fn-coefficients if interaction terms are used
        if isinstance(self.parent_fit, MultiFits):
            for name, parent_fit in self.parent_fit.accessor.config_fits.items():
                if parent_fit.int_Q is True:
                    parent_fit._fnevals_input = parent_fit.R._fnevals
        else:
            if self.parent_fit.int_Q is True:
                self.parent_fit._fnevals_input = self.parent_fit.R._fnevals

        if print_progress is True:
            # initialize shared values that will be used to track the number
            # of completed processes and the mean time to complete a process
            process_cnt = _setup_cnt(N_items=len(reader_args), ncpu=ncpu)
        else:
            process_cnt = None

        if ncpu > 1:
            log.info(f"start of parallel evaluation on {ncpu} cores")
            with mp.Pool(ncpu, **pool_kwargs) as pool:
                # loop over the reader_args
                res_async = pool.starmap_async(
                    self._evalfunc, zip(reader_args, repeat(process_cnt))
                )

                pool.close()  # Marks the pool as closed.
                pool.join()  # Waits for workers to exit.
                res = res_async.get()
        else:
            log.info("start of single-core evaluation")
            # force autocontinue=True when calling the initializer since setup() has
            # already been called in the main process so all folders will already exist!
            init_autocontinue = self.autocontinue
            self.autocontinue = True

            # call the initializer if it has been provided
            if "initializer" in pool_kwargs:
                if "initargs" in pool_kwargs:
                    pool_kwargs["initializer"](*pool_kwargs["initargs"])
                else:
                    pool_kwargs["initializer"]()

            self.autocontinue = init_autocontinue

            res = []
            for reader_arg in reader_args:
                res.append(
                    self._evalfunc(reader_arg=reader_arg, process_cnt=process_cnt)
                )

        # TODO incorporate this in run_finalout
        # call finaloutput if post-processing has been performed and a finaloutput
        # function is provided
        if self._postprocess and callable(self.proc_cls.finaloutput):
            # in case of a multi-config, results will be dicts!
            if isinstance(self.parent_fit, MultiFits):
                # store original finalout_name
                finalout_name, ending = self.proc_cls.finalout_name.split(".")

                for name in self.parent_fit.config_names:
                    self.proc_cls.finalout_name = f"{finalout_name}__{name}.{ending}"
                    self.proc_cls.finaloutput(
                        (i.get(name) for i in res if i is not None)
                    )
                # reset initial finalout_name
                self.proc_cls.finalout_name = f"{finalout_name}.{ending}"

            # if len(self.cfg.config_names) > 0:
            #     for i, res_i in enumerate(
            #         zip_longest(*(i for i in res if i is not None))
            #     ):
            #         self.proc_cls.rt1_procsesing_respath = (
            #             self.dumppath / "results" / self.parent_fit[i][0]
            #         )
            #         self.proc_cls.finaloutput(res_i)
            else:
                return self.proc_cls.finaloutput(res)
        else:
            return res

    def run_processing(
        self,
        ncpu=1,
        print_progress=True,
        reader_args=None,
        pool_kwargs=None,
        preprocess_kwargs=None,
        logfile_level=1,
        dump_fit=True,
        postprocess=True,
    ):
        """
        Start the processing

        Parameters
        ----------
        ncpu : int
            The number of cpu's to use. The default is 1.
        print_progress : bool, optional
            Indicator if a progress-bar should be printed or not.
            If True, it might be wise to suppress warnings during runtime
            to avoid unwanted outputs. This can be achieved by using:

                >>> import warnings
                ... warnings.simplefilter('ignore')

            The default is True.
        reader_args : list, optional
            A list of dicts that will be passed to the reader-function.
            I `None`, the `reader_args` will be taken from the return of the
            `preprocess()`-function via:

            >>> reader_args = preprocess(**preprocess_kwargs)['reader_args']

            The default is None.
        pool_kwargs : dict, optional
            A dict with additional keyword-arguments passed to the
            initialization of the multiprocessing-pool via:

            >>> mp.Pool(ncpu, **pool_kwargs)

            The default is None.
        preprocess_kwargs : dict, optional
            A dict with keyword-arguments passed to the call of the preprocess
            function via:

            >>> preprocess(**preprocess_kwargs)

            The default is None.
        logfile_level : int
            the log-level of the log-file that will be generated as
            "processing folder / cfg / RT1_processing.log"

            If None no log-file will be generated.

            for information on the level values see:
                https://docs.python.org/3/library/logging.html#logging-levels
        dump_fit: bool, optional
            indicator if `processing_config.dump_fit_to_file()` should be
            called after finishing the fit.
            The default is True
        postprocess : bool, optional
            indicator if postprocess() and finaloutput() workflows should be
            executed or not.
            The default is True.
        """

        try:
            if logfile_level is not None and ncpu > 1:
                # start a listener-process that takes care of the logs from
                # multiprocessing workers
                queue = mp.Manager().Queue(-1)
                listener = mp.Process(target=self._listener_process, args=[queue])

                # make the queue listen also to the MainProcess start the
                # listener-process that writes the file to disc AFTER
                # initialization of the  folder-structure!
                # (otherwise the log-file can not be generated!)
                self._worker_configurer(queue)
            else:
                queue = None

            # initialize all necessary properties if setup was not yet called
            self.setup()

            if logfile_level is not None and ncpu > 1:
                # start the listener after the setup-function completed, since
                # otherwise the folder-structure does not yet exist and the
                # file to which the process is writing can not be generated!
                listener.start()

            if preprocess_kwargs is None:
                preprocess_kwargs = dict()

            # save the used model-definition string to a file
            if self.dumppath is not None:
                if isinstance(self.parent_fit, MultiFits):
                    # if multiple configs are provided, save an individual file for each
                    for (
                        cfg_name,
                        parent_fit,
                    ) in self.parent_fit.accessor.config_fits.items():
                        with open(
                            self.dumppath / "cfg" / f"model_definition__{cfg_name}.txt",
                            "w",
                        ) as file:

                            outtxt = ""
                            if hasattr(self.proc_cls, "description"):
                                outtxt += dedent(self.proc_cls.description)
                                outtxt += "\n\n"
                                outtxt += "_" * 77
                                outtxt += "\n\n"

                            outtxt += parent_fit._model_definition
                            print(outtxt, file=file)

                            log.info(outtxt)

                else:
                    with open(
                        self.dumppath / "cfg" / "model_definition.txt", "w"
                    ) as file:

                        outtxt = ""
                        if hasattr(self.proc_cls, "description"):
                            outtxt += dedent(self.proc_cls.description)
                            outtxt += "\n\n"
                            outtxt += "_" * 77
                            outtxt += "\n\n"

                        outtxt += self.parent_fit._model_definition
                        print(outtxt, file=file)

                        log.info(outtxt)

            _ = self.processfunc(
                ncpu=ncpu,
                print_progress=print_progress,
                reader_args=reader_args,
                pool_kwargs=pool_kwargs,
                preprocess_kwargs=preprocess_kwargs,
                queue=queue,
                dump_fit=dump_fit,
                postprocess=postprocess,
            )

        except Exception as err:
            log.exception("there has been something wrong during processing!")
            raise err

        finally:
            if hasattr(self, "proc_cls"):
                if hasattr(self.proc_cls, "finalizer"):
                    self.proc_cls.finalizer()

            # turn off capturing warnings
            logging.captureWarnings(False)

            if logfile_level is not None:
                if ncpu > 1:

                    # tell the queue to stop
                    queue.put_nowait(None)

                    # stop the listener process
                    listener.join()

                # remove any remaining file-handler and queue-handlers after
                # the processing is done
                handlers = groupby_unsorted(log.handlers, key=lambda x: x.name)
                for h in handlers.get("rtprocessing_queuehandler", []):
                    # close the handler (removing it does not close it!)
                    h.close()
                    # remove the file-handler from the log
                    log.handlers.remove(h)

    def run_finaloutput(
        self,
        ncpu=1,
        use_N_files=None,
        use_config=None,
        finalout_name=None,
        finaloutput=None,
        postprocess=None,
        print_progress=True,
        logfile_level=1,
    ):
        """
        run postprocess and finaloutput for available .dump files

        Parameters
        ----------
        ncpu : int
            The number of cpu's to use. The default is 1.
        use_N_files : int, optional
            The number of files to process (e.g. a subset of all files)
        use_config : str or list, optional
            Only relevant if a multi-config .ini file has been used.
            The names of the configs to use.
            The default is None, in which case all configs are used!
        finalout_name : str, optional
            override the finalout_name provided in the ".ini" file
        finaloutput : callable, optional
            override the finaloutput function provided via the
            "processing_config.py" script
            NOTICE: the call-signature is
            `finaloutput(proc_cls, ...)` where proc_cls is the
            used instance of the "processing_cfg" class!
            -> this way you have access to all arguments specified in
            proc_cls such as ".save_path", ".dumpfolder" etc.
        postprocess : callable, optional
            override the postprocess function provided via the
            "processing_config.py" script
            NOTICE: the call signature is similar to finaloutput above!
        print_progress : bool, optional
            Indicator if a progress-bar should be printed or not.
            If True, it might be wise to suppress warnings during runtime
            to avoid unwanted outputs. This can be achieved by using:

                >>> import warnings
                ... warnings.simplefilter('ignore')

            The default is True.
        logfile_level : int
            the log-level of the log-file that will be generated as
            "processing folder / cfg / RT1_processing.log"

            If None no log-file will be generated.

            for information on the level values see:
                https://docs.python.org/3/library/logging.html#logging-levels

        """
        try:
            if logfile_level is not None and ncpu > 1:
                # start a listener-process that takes care of the logs from
                # multiprocessing workers
                queue = mp.Manager().Queue(-1)
                listener = mp.Process(target=self._listener_process, args=[queue])

                # make the queue listen also to the MainProcess start the
                # listener-process that writes the file to disc AFTER
                # initialization of the  folder-structure!
                # (otherwise the log-file can not be generated!)
                self._worker_configurer(queue)
            else:
                queue = None

            # override finalout_name
            # do this BEFORE self.setup() is called !!
            if finalout_name is not None:
                assert isinstance(finalout_name, str), "finalout_name must be a string"
                # self.proc_cls.finalout_name = finalout_name
                self.override_config(
                    PROCESS_SPECS=dict(finalout_name=finalout_name), override=False
                )

            # initialize all necessary properties with autocontinue=True
            initial_autocontinue = self.autocontinue
            self.autocontinue = True
            self.setup()
            self.autocontinue = initial_autocontinue

            if logfile_level is not None and ncpu > 1:
                # start the listener after the setup-function completed, since
                # otherwise the folder-structure does not yet exist and the
                # file to which the process is writing can not be generated!
                listener.start()

            fitlist = self._get_files(use_N_files=use_N_files)

            return self._run_finalout(
                ncpu,
                fitlist=fitlist,
                use_config=use_config,
                finaloutput=finaloutput,
                postprocess=postprocess,
                queue=queue,
                print_progress=print_progress,
            )

        except Exception as err:
            log.exception("there was an error during finalout generation!")
            raise err

        finally:

            # turn off capturing warnings
            logging.captureWarnings(False)

            if logfile_level is not None:
                if ncpu > 1:
                    # tell the queue to stop
                    queue.put_nowait(None)

                    # stop the listener process
                    listener.join()

                # remove any remaining file-handler and queue-handlers after
                # the processing is done
                handlers = groupby_unsorted(log.handlers, key=lambda x: x.name)
                for h in handlers.get("rtprocessing_queuehandler", []):
                    # close the handler (removing it does not close it!)
                    h.close()
                    # remove the file-handler from the log
                    log.handlers.remove(h)

    def _run_postprocess(
        self, fitpath, use_config=None, process_cnt=None, postprocess=None
    ):
        if process_cnt is not None:
            start = _start_cnt()

        # load fitobjects based on fit-paths
        try:
            fit = load(fitpath)
            if use_config is None:
                use_config = fit.config_names

            # assign pre-evaluated fn-coefficients
            for config_name in use_config:
                config_fit = self.parent_fit.accessor.config_fits[config_name]
                if config_fit.int_Q is True:
                    try:
                        getattr(fit.configs, config_name)._fnevals_input = config_fit._fnevals_input
                    except AttributeError:
                        log.error("could not assign pre-evaluated fn-coefficients to " +
                                  f"the config {config_name}... " +
                                  "available configs of the loaded Fits object:" +
                                  f" {fit.config_names}")


        except Exception:
            log.error(f"there was an error while loading {fitpath}")
            _increase_cnt(process_cnt, start, err=True)
            return
        # TODO fix reader_arg argument of postprocess
        try:
            if isinstance(fit, MultiFits):
                if postprocess is None:
                    ret = dict(
                        fit.apply(
                            self.proc_cls.postprocess,
                            use_config=use_config,
                            reader_arg=fit.reader_arg,
                        )
                    )
                else:
                    ret = dict(
                        fit.apply(
                            postprocess,
                            use_config=use_config,
                            reader_arg=fit.reader_arg,
                        )
                    )
            else:
                # run postprocessing
                if postprocess is None:
                    ret = self.proc_cls.postprocess(fit, reader_arg=fit.reader_arg)
                else:
                    ret = postprocess(fit, reader_arg=fit.reader_arg)

            if process_cnt is not None:
                _increase_cnt(process_cnt, start, err=False)

            return ret

        except Exception as ex:
            if callable(self.proc_cls.exceptfunc):
                ex_ret = self.proc_cls.exceptfunc(ex, fit.reader_arg)
                if ex_ret is None or ex_ret is False:
                    _increase_cnt(process_cnt, start, err=True)
                else:
                    _increase_cnt(process_cnt, start, err=False)
                return ex_ret
            else:
                raise ex

    def _get_files(self, use_N_files=None):
        # get the relevant files that have to be processed for `run_finaloutput`

        useres = getattr(RTresults(self.dumppath), self.proc_cls.dumpfolder)
        # get dump-files
        dumpiter = useres.dump_files

        if use_N_files is not None:
            fitlist = list(islice(dumpiter, use_N_files))
        else:
            fitlist = list(dumpiter)
        return fitlist

    def _run_finalout(
        self,
        ncpu,
        fitlist=None,
        use_config=None,
        finaloutput=None,
        postprocess=None,
        queue=None,
        print_progress=True,
    ):

        # override postprocess function
        if postprocess is not None:
            assert callable(postprocess), "postprocess must be callable!"

        # don't call additional initializers, only use the default initializer
        # (to enable subprocess-logging)
        pool_kwargs = dict(initializer=self._initializer, initargs=[None, queue])

        # pre-evaluate the fn-coefficients if interaction terms are used
        # since finalout might involve calling "calc_model()" or similar functions
        # that require a re-evaluation of the fn_coefficients
        if isinstance(self.parent_fit, MultiFits):
            if use_config is None:
                use_config = self.parent_fit.config_names

            for fit_name in use_config:
                parent_fit = self.parent_fit.accessor.config_fits[fit_name]
                if parent_fit.int_Q is True:
                    parent_fit._fnevals_input = parent_fit.R._fnevals
        else:
            if self.parent_fit.int_Q is True:
                self.parent_fit._fnevals_input = self.parent_fit.R._fnevals

        if print_progress is True:
            # initialize shared values that will be used to track the number
            # of completed processes and the mean time to complete a process
            process_cnt = _setup_cnt(N_items=len(fitlist), ncpu=ncpu)
        else:
            process_cnt = None

        if ncpu > 1:
            log.progress(f"start of finalout generation on {ncpu} cores")
            with mp.Pool(ncpu, **pool_kwargs) as pool:
                # loop over the reader_args
                res_async = pool.starmap_async(
                    self._run_postprocess,
                    zip(
                        fitlist,
                        repeat(use_config),
                        repeat(process_cnt),
                        repeat(postprocess),
                    ),
                )

                pool.close()  # Marks the pool as closed.
                pool.join()  # Waits for workers to exit.
                res = res_async.get()
        else:
            log.progress("start of single-core finalout generation")
            # force autocontinue=True when calling the initializer since setup() has
            # already been called in the main process so all folders will already exist!
            init_autocontinue = self.autocontinue
            self.autocontinue = True

            # call the initializer if it has been provided
            if "initializer" in pool_kwargs:
                if "initargs" in pool_kwargs:
                    pool_kwargs["initializer"](*pool_kwargs["initargs"])
                else:
                    pool_kwargs["initializer"]()

            self.autocontinue = init_autocontinue
            res = []
            for fitpath in fitlist:
                res.append(
                    self._run_postprocess(
                        fitpath=fitpath,
                        use_config=use_config,
                        process_cnt=process_cnt,
                        postprocess=postprocess,
                    )
                )
        log.progress("... generating finaloutput")


        # in case of a multi-config, results will be dicts!
        if isinstance(self.parent_fit, MultiFits):
            # store original finalout_name
            finalout_name, ending = self.proc_cls.finalout_name.split(".")

            for name in use_config:
                self.proc_cls.finalout_name = f"{finalout_name}__{name}.{ending}"
                self.proc_cls.config_name = name
                if callable(finaloutput):
                    finaloutput((i.get(name) for i in res if i is not None))
                elif hasattr(self.proc_cls, "finaloutput"):
                    self.proc_cls.finaloutput(
                        (i.get(name) for i in res if i is not None)
                    )

            # reset initial finalout_name
            self.proc_cls.finalout_name = f"{finalout_name}.{ending}"
        else:
            if callable(finaloutput):
                return finaloutput(res)
                log.info("finished generation of finaloutput!")
            elif hasattr(self.proc_cls, "finaloutput"):
                return self.proc_cls.finaloutput(res)
            else:
                return res


class RTresults(object):
    """
    A class to provide easy access to processed results.
    On initialization the class will traverse the provided "parent_path"
    and recognize any sub-folder that matches the expected folder-structure
    as a sub-result.

    NOTE: only the first 100 entries of each directory will be checked
    to avoid performance issues in case a folder with a lot of sub-files is selected

    Assuming a folder-structure as indicated below, the class can be used via:

        >>> ../../RESULTS      (parent_path)
        ...     results/..     (.nc files)
        ...     dumps/..       (.dump files)
        ...     cfg/..         (.ini files)
        ...
        ...     sub_RESULT1
        ...         results/.. (.nc files)
        ...         dumps/..   (.dump files)
        ...         cfg/..     (.ini files)
        ...
        ...     sub_RESULT2_with_multiple_configs
        ...         results/.. (.nc files)
        ...             config1/.. (.nc files)
        ...             config2/.. (.nc files)
        ...         dumps/..
        ...             config1/.. (.dump files)
        ...             config2/.. (.dump files)
        ...         cfg/..     (.ini files)


        >>> results = RT1_results(parent_path)
        ... # print available NetCDF files and variables
        ... x.RESULTS.NetCDF_variables
        ... x.sub_RESULT_1.NetCDF_variables
        ...
        ... # load some dump-files
        ... fit_random = results.sub_RESULT_1.load_fit()
        ... fit1_0 = results.RESULT.load_fit('id_of_fit_1')
        ... fit1_1 = results.sub_RESULT_1.load_fit('id_of_fit_1')
        ... fit1_2 = results.sub_RESULT_2.load_fit('id_of_fit_1')
        ...
        ... # access a NetCDF file
        ... with results.sub_RESULT_2.load_nc() as ncfile:
        ...     --- read something from the ncfie ---
        ...
        ... # get a generator for the paths of all available dump-files
        ... dump-files = results.sub_RESULT_1.dump_files
        ...
        ... # load the configfile of a given fit
        ... cfg_01 = results.sub_RESULT_1.load_cfg()


    Parameters
    ----------
    parent_path : str
        the parent-path where the results are stored.
    """

    def __init__(self, parent_path):
        self._parent_path = Path(parent_path)

        self._paths = dict()
        if self._check_folderstructure(self._parent_path):
            # check if the path is already a valid folder-structure
            self._addresult(Path(self._parent_path))
        else:
            # otherwise, check if a subfolder has a valid folder-structure
            # (only 1 level is checked)
            for p in islice(os.scandir(self._parent_path), 100):
                if p.is_dir():
                    self._addresult(Path(p.path))

    @staticmethod
    def _check_folderstructure(path):
        return all(
            folder in [i.name for i in islice(os.scandir(path), 100) if i.is_dir()]
            for folder in [
                "cfg",
                "results",
                "dumps",
            ]
        )

    def _addresult(self, path):
        if self._check_folderstructure(path):
            self._paths[path.stem] = path
            log.info(f"... adding result {path.stem}")
            setattr(
                self,
                path.stem,
                self._RT1_fitresult(path.stem, path),
            )

    class _RT1_fitresult(object):
        def __init__(self, name, path):
            self.name = name
            self.path = Path(path)
            self._result_path = self.path / "results"
            self._dump_path = self.path / "dumps"
            self._cfg_path = self.path / "cfg"

            self._nc_paths = self._get_results(".nc")

        @staticmethod
        def _check_dump_filename(p):
            return p.endswith(".dump") and "error" not in p.split(os.sep)[-1]

        def _get_results(self, ending):
            assert self._result_path.exists(), (
                f"{self._result_path}" + " does not exist"
            )

            results = dict()
            for i in os.scandir(self._result_path):
                if i.is_file() and i.path.endswith(ending):
                    res = Path(i)
                    results[res.stem] = res

            if len(results) == 0:
                log.info(f'there is no "{ending}" file in "{self._result_path}"')

            return results

        def load_nc(self, result_name=None, use_xarray=True):
            """
            open a NetCDF file stored in the "results"-folder

            can be used as context-manager, e.g.:

                >>> with result.load_nc() as ncfile:
                ...     --- do something ---


            Parameters
            ----------
            result_name : str, optional
                The name of the NetCDF-file (without a .nc extension).
                If None, and only 1 file is available, the available file
                will be laoded. The default is None.
            use_xarray : bool, optional
                Indicator if NetCDF4 or xarray should be used to
                laod the NetCDF file. The default is True.

            Returns
            -------
            file : a file-handler for the NetCDF file
                the return of either xarray.Dataset or NetCDF4.Dataset
            """
            results = self._get_results(".nc")
            assert len(results) > 0, "no NetCDF file in the results folder!"

            assert len(results) == 1 or result_name in results, (
                "there is more than 1 result... "
                + 'provide "result_name":\n    - '
                + "\n    - ".join(results.keys())
            )

            if result_name is None:
                result_name = list(results.keys())[0]

            log.info(f"loading nc-file for {result_name}")
            file = xar.open_dataset(results[result_name])
            return file

        def load_fit(self, ID=0, return_ID=False):
            """
            load one of the available .dump-files located in the "dumps"
            folder.  (using rt1.rtfits.load() )

            Notice: the dump-files are generated using cloudpickle.dump()
            and might be platform and environment-specific!

            Parameters
            ----------
            ID : str or int, or pathli.Path
                If str:  The name of the dump-file to be loaded
                         (without the .dump extension).
                If int:  load the nth file found in the dumpfolder
                if Path: the full path to the .dump file
                If None: load a random file from the folder
                         NOTE: in order to load a random file, the folder
                         must be indexed first! (this might take some time for
                         very large amounts of files)
                         > call `scan_folder()` to re-scan the folder contents
                The default is 0.
            return_ID : bool, optional
                If True, a tuple (fit, ID) is returned, otherwise only
                the fit is returned

            Returns
            -------
            fit : rt1.rtfits.Fits
                the loaded rt1.rtfits.Fits result.
            """

            if isinstance(ID, Path):
                filepath = ID
            else:
                if ID is None:
                    if not hasattr(self, "_dump_file_list"):
                        self.scan_folder()

                    filepath = Path(choice(self._dump_file_list))

                    log.info(
                        f"loading random ID ({filepath.stem}) from "
                        + f" {self._n_dump_files} available files"
                    )
                elif isinstance(ID, int):
                    filepath = next(islice(self.dump_files, ID, None))
                elif isinstance(ID, str):
                    filepath = self._dump_path / (ID + ".dump")

            fit = load(filepath)

            if return_ID is True:
                return (fit, ID)
            else:
                return fit

        def scan_folder(self):
            """
            (re)-scan the contents of the dump-folder for ".dump" files
            """
            log.warning(f'... indexing folder:\n "{self._dump_path}"')
            self._dump_file_list = list(self.dump_files)
            self._n_dump_files = len(self._dump_file_list)


        def load_cfg(self, cfg_name=None):
            """
            load the configfile stored in the "cfg" folder

            Parameters
            ----------
            cfg_name : str, optional
                The name of the config-file to laod in case more than 1
                ".ini"-files are found. The default is None.

            Returns
            -------
            cfg : rt1.rtparse.RT1_configparser
                a configparser instance of the selected configuration.

            """
            cfgfiles = list(self._cfg_path.glob("*.ini"))
            assert len(cfgfiles) > 0, 'NO ".ini"-file found!'

            if cfg_name is None:
                assert len(cfgfiles) == 1, (
                    "there is more than 1 .ini file in the cfg folder..."
                    + 'provide a "cfg_name":\n'
                    + "    - "
                    + "\n    - ".join([i.name for i in cfgfiles])
                )

                cfg_name = cfgfiles[0].name

            cfg = RT1_configparser(self._cfg_path / cfg_name)
            return cfg

        @property
        def dump_files(self):
            """
            a generator returning the paths to the available dump-files

            NOTICE: only files that do NOT contain "error" in the filename and
            whose file-ending is ".dump" are returned!
            """
            for entry in os.scandir(self._dump_path):
                if (entry.name.endswith('.dump')
                    and entry.is_file()
                    and 'error' not in entry.name):

                    yield entry.path

        @property
        def NetCDF_variables(self):
            """
            print all available NetCDF-files and their variables
            """
            results = self._get_results(".nc")
            assert len(results) > 0, "no NetCDF file in the results folder!"
            for r in results:
                print(f"\n################ result:  {r}.nc")
                with self.load_nc(r) as ncfile:
                    print(ncfile)

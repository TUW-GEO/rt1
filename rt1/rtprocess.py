import multiprocessing as mp
from timeit import default_timer
from datetime import datetime, timedelta
from itertools import repeat, islice
import ctypes
import sys
import traceback
from textwrap import dedent

from functools import partial
from pathlib import Path
import shutil

import pandas as pd
import numpy as np

from .general_functions import dt_to_hms, update_progress, groupby_unsorted
from .rtparse import RT1_configparser
from .rtresults import RTresults
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
        fitlist=None
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
        fitlist : list, optional
            optional way to provide a list of paths to dump-files directly
            (useful if RTprocess is used without a config attached)
            The default is None, in which case the dump-files will be
            identified according to the definitions in the config-file.
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

            if not fitlist:
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
            if use_config is None and isinstance(fit, MultiFits):
                use_config = fit.config_names


            # assign pre-evaluated fn-coefficients
            if hasattr(self, "parent_fit"):
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
            if hasattr(self, "proc_cls") and callable(self.proc_cls.exceptfunc):
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
        if hasattr(self, "parent_fit"):
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

        # in case no config is provided return the output to the console
        if not hasattr(self, "proc_cls"):
            if isinstance(res[0], dict):
                out = dict()
                if use_config is None:
                    use_config = res[0].keys()
                for name in use_config:
                    if callable(finaloutput):
                        out[name] = finaloutput((i.get(name) for i in res if i is not None))
            else:
                if callable(finaloutput):
                    out = finaloutput(res)
                else:
                    out = res

            return out

        # in case of a multi-config, results will be dicts!
        # if isinstance(self.parent_fit, MultiFits):
        if isinstance(res[0], dict):
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

    def _postprocess_xarray(
        self,
        fit,
        saveparams=None,
        xindex=("x", -9999),
        yindex=None,
        staticlayers=None,
        auxdata=None,
        sig_to_dB=False,
        inc_to_degree=False
    ):
        """
        the identification of parameters is as follows:

            1) 'sig' (conv. to dB) and 'inc' (conv. to degrees) from dataset
            2) any parameter present in defdict is handled accordingly
            3) auxdata (a pandas-dataframe) is appended
            4) static layers are added according to the provided dict

        Parameters
        ----------
        fit : rt1.rtfits.Fits object
            the fit-object to use
        saveparams : list, optional
            a list of strings that correspond to parameter-names that should
            be included.
            can be any parameter present in "fit.dataset", "fit.res_df",
            The default is None.
        xindex : tuple, optional
            a tuple (name, value) that will be used as the x-index.
            The default is ('x', -9999).
        yindex : tuple, optional
            a tuple (name, value) that will be used as the y-index.
            if provided, a multiindex (x, y) will be used!
            Be warned... when combining xarrays the x- and y- coordinates will
            be assumed as a rectangular grid!
            The default is None.
        staticlayers : dict, optional
            a dict with parameter-names and values that will be added das
            static layers. The default is None.
        auxdata : pandas.DataFrame, optional
            a pandas DataFrame that will be concatenated to the DataFrame obtained
            from combining all 'saveparams'.
            NOTICE: if the index does not align well with the fit-index, the
            generated output can increase a lot in size due to missing values!
            The default is None.
        sig_to_dB : bool
            indicator if sigma0 values (e.g. "sig", "tot", "surf", "vol", "inter")
            should be converted to dB
        inc_to_degree : bool
            indicator if incidence-angle values (e.g. "inc")
            should be converted to degrees

        Returns
        -------
        dfxar : xarray.Dataset
            a xarray-dataset with all layers defined according to the specs.

        """
        if saveparams is None:
            saveparams = []

        if staticlayers is None:
            staticlayers = dict()

        defs = self._defdict_parser(fit.defdict)

        usedfs = []
        for key in saveparams:

            if key == "sig":
                if fit.dB is False and sig_to_dB:
                    usedfs.append(10.0 * np.log10(fit.dataset.sig))
                else:
                    usedfs.append(fit.dataset.sig)
            elif key == "inc" and inc_to_degree:
                usedfs.append(np.rad2deg(fit.dataset.inc))

            elif key in fit.defdict:
                if key in defs["fitted_dynamic"]:
                    usedfs.append(fit.res_df[key])
                elif key in defs["fitted_const"]:
                    staticlayers[key] = fit.res_dict[key][0]
                elif key in defs["constant"]:
                    staticlayers[key] = fit.defdict[key][1]
                elif key in defs["auxiliary"]:
                    usedfs.append(fit.dataset[key])
            elif key in fit.dataset:
                if (key in ["tot", "surf", "vol", "inter"] and
                    fit.dB is False and
                    sig_to_dB):
                    usedfs.append(10.0 * np.log10(fit.dataset[key]))
                else:
                    usedfs.append(fit.dataset[key])
            elif key in fit.reader_arg:
                staticlayers[key] = fit.reader_arg[key]
            else:
                log.warning(
                    f"the parameter {key} could not be processed"
                    + "during xarray postprocessing"
                )

        if auxdata is not None and len(auxdata) > 0:
            usedfs.append(auxdata)

        # combine all timeseries and set the proper index
        df = pd.concat(usedfs, axis=1)
        df.columns.names = ["param"]
        df.index.names = ["date"]

        if yindex is not None:
            df = pd.concat([df], keys=[yindex[1]], names=[yindex[0]])
            df = pd.concat([df], keys=[xindex[1]], names=[xindex[0]])

            # set static layers
            statics = pd.DataFrame(
                staticlayers,
                index=pd.MultiIndex.from_product(
                    iterables=[[xindex[1]], [yindex[1]]], names=["x", "y"]
                ),
            )

        else:
            df = pd.concat([df], keys=[xindex[1]], names=[xindex[0]])

            # set static layers
            statics = pd.DataFrame(staticlayers, index=[xindex[1]])
            statics.index.name = xindex[0]

        dfxar = xar.merge([df.to_xarray(), statics.to_xarray()])

        return dfxar

    @staticmethod
    def _defdict_parser(defdict):
        """
        get parameter-dynamics specifications from a given defdict

        Parameters
        ----------
        defdict : dict
            a defdict (e.g. fit.defdict)

        Returns
        -------
        parameter_specs : dict
            a dict that contains lists of parameter-names according to their
            specifications. (keys should be self-explanatory)
        """
        parameter_specs = dict(
            fitted=[],
            fitted_const=[],
            fitted_dynamic=[],
            fitted_dynamic_manual=[],
            fitted_dynamic_index=[],
            fitted_dynamic_datetime=[],
            fitted_dynamic_integer=[],
            constant=[],
            auxiliary=[],
        )

        for key, val in defdict.items():
            if val[0] is True:
                parameter_specs["fitted"].append(key)
                if val[2] is None:
                    parameter_specs["fitted_const"].append(key)
                else:
                    parameter_specs["fitted_dynamic"].append(key)
                    if val[2] == "manual":
                        parameter_specs["fitted_dynamic_manual"].append(key)
                    elif val[2] == "index":
                        parameter_specs["fitted_dynamic_index"].append(key)
                    elif isinstance(val[2], str):
                        parameter_specs["fitted_dynamic_datetime"].append((key, val[2]))
                    elif isinstance(val[2], int):
                        parameter_specs["fitted_dynamic_integer"].append((key, val[2]))
            else:
                if val[1] == "auxiliary":
                    parameter_specs["auxiliary"].append(key)
                else:
                    parameter_specs["constant"].append(key)

        return parameter_specs

    def _export_postprocess(self,
                            fit,
                            reader_arg=None,
                            parameters=[],
                            metrics=None,
                            export_functions=None,
                            model_keys=[],
                            index_col='ID',
                            sig_to_dB=False,
                            inc_to_degree=False,
                            _fnevals_input=None
                            ):
        """
        Parameters
        ----------

        fit: rt1.rtfits.Fits object
            The fits object.
        reader_arg: dict
            the arguments passed to the reader function.
        parameters : list
            a list of parameter-names to attach.
        metrics : dict
            key: the name to use for the metric in the returned dataset
            value: a tuple of the form:
                  (metric, parameter 1, parameter 2)
        export_fuinctions : dict
            a dict of functions to use for exporting the parameter
        model_keys : list
            a list of keys that correspond to model-calculation results
            (e.g. any of ["tot", "surf", "vol", "inter"])
        index_col : str
            the name of the reader-arg value to use as index
        _fnevals_input : dict or callable
            pre-evaluated fnevals functions. in case fit is a MultiFits object:
            a dict with the config-names and pre-evaluated fnevals functions
        Returns
        -------
        df: pandas.DataFrame
            a xarray.Dataset containing the fitted parameterss.

        """
        if reader_arg is None:
            reader_arg = fit.reader_arg

        # assign pre-evaluated fn-coefficients
        if _fnevals_input:
            if hasattr(fit, "config_name"):
                if fit.int_Q is True and fit.config_name in _fnevals_input:
                    fit._fnevals_input = _fnevals_input[fit.config_name]
            else:
                fit._fnevals_input = _fnevals_input

        staticlayers = dict()
        if metrics:
            for name, p in metrics.items():
                staticlayers[name] = getattr(
                    getattr(getattr(fit.metric, p[1]), p[2]), p[0].lower())

        auxdata = dict()
        if model_keys:
            auxdata = fit.calc_model(return_components=True)[model_keys]

        if export_functions:
            for key, func in export_functions.items():
                auxdata[key] = func(fit)

        ret = self._postprocess_xarray(
            fit=fit,
            saveparams=set(parameters) ^ set(auxdata) ^ set(model_keys),
            xindex=(index_col, reader_arg[index_col]),
            staticlayers=staticlayers,
            auxdata=pd.DataFrame(auxdata),
            sig_to_dB=sig_to_dB,
            inc_to_degree=inc_to_degree
        )

        return ret

    @staticmethod
    def _export_finalout(res, savepath=None, attrs=None, descriptions=None,
                         encoding=None, concat_dim="ID"):

        resxar = xar.combine_nested([i for i in res if i is not None],
                                    concat_dim=concat_dim)
        if attrs:
            for key, val in attrs.items():
                resxar.attrs[key] = val

        if descriptions:
            for p, a in descriptions.items():
                for key, val in a.items():
                    getattr(resxar, p).attrs[key] = val

        if savepath:
            if not encoding:
                encoding = {key: {"zlib": True, "complevel": 1}
                            for key in resxar.data_vars}

            log.info(
                "export of NetCDF file at location:\n"
                + '"'
                + savepath
                + '"'
            )

            resxar.to_netcdf(savepath, encoding=encoding)
        else:
            return resxar

    def export_data(self, parameters=None, metrics=None,
                    attributes=None, export_functions=None,
                    index_col="ID", use_config=None, dumpfolder=None,
                    use_nfiles=None, ncpu=1, sig_to_dB=False, inc_to_degree=False,
                    pre_evaluate_fn_coefs=True, savepath=None):
        """
        a convenience-method to export parameters and performance-metrics
        from a collection of rtfits.Fits objects

        Parameters
        ----------
        parameters : list or dict
            a list (or dict) of parameter-names to attach.
            can be any parameter available in "fit.dataset", "fit.res_df", any of
            the model-contributions, e.g.: ["tot", "surf", "vol", "inter"] or any
            parameter whose export-function has been provided via "export_functions"

            To attach descriptions, use a dict of dicts of the form:
            >>> dict(sig = dict(long_name = "sigma0 data",
            >>>                 units = "dB"),
            >>>      inc = dict(long_name = "incidence angle",
            >>>                 units = "degrees"))

        metrics : dict
            a dict of metrics to calculate between model-parameters and dataset-keys

            - key: the name to use for the metric in the returned dataset
            - value: a tuple of the form (description is optional):
                     (metric, parameter 1, parameter 2, [description-dict])

            >>> dict(R = ("pearson", "sig", "tot",
            >>>           dict(long_name="sig0 pearson correlation")))

        export_functions = dict, optional
            a dict with functions that will be used to export the parameter-values
            (Note that this will override the default extraction-procedures!)

            >>> dict(sig=lambda fit: fit.dataset.sig)

        attributes : dict, optional
            additional attributes to attach to the returned xarray.Dataset
        index_col : str
            the name of the reader-arg value to use as index
        use_config : list, optional
            a list of configs to use in case a multi-config fit has been performed.
            The default is None.
        dumpfolder : str, optional
            the dumpfolder to use. (must be specified if more than 1 result is found)
            The default is None.
        use_nfiles : int, optional
            the number of files to process.
            The default is None in which case ALL files are processed!
        ncpu : int, optional
            the number of cpu's to use. The default is 1.
        sig_to_dB : bool, optional
            indicator if sigma0 datasets ("sig", "tot", "surf", "vol", "inter")
            should be converted to dB
        inc_to_degrees : str, optional
            indicator if incidence-angle datasets ("inc") should be converted to
            degrees
        pre_evaluate_fn_coefs : bool
            indicator if the first Fits (or MultiFits) object should be used to
            pre-evaluate fn-coefficients required to evaluate interaction-terms.
            Note: if this is set to False, the coefficients have to be evaluated
            for every single file which can cause a major reduction in speed!
            The default is True.
        savepath : str
            the path to store the exported NetCDF files
        Returns
        -------
        out : xarray.Dataset
            a xarray Dataset of the exported data.

        """

        if not parameters:
            parameters = []

        if not metrics:
            metrics = dict()

        # ----- set descriptions (and separate model-keys)
        descriptions = dict()
        if isinstance(parameters, dict):
            model_keys = dict()
            descriptions.update(parameters)

            for key in ["tot", "surf", "vol", "inter"]:
                if key in parameters:
                    model_keys[key] = parameters.pop(key)
        else:
            model_keys = set(parameters) & set(["tot", "surf", "vol", "inter"])
            parameters = set(parameters) ^ model_keys

        for key, val in metrics.items():
            if len(val) == 4:
                descriptions[key] = val[3]

        # ----- initialize a RTresults object for easy access to the list of dump-files
        res = RTresults(self.config_path)

        # ----- get dumpfolder to use
        if not dumpfolder:
            if len(res._paths) > 1:
                log.error("there is more than 1 possible dumpfolder!\n" +
                          f"please explicitly specify one of:\n{list(res._paths)}")
                return
            else:
                dumpfolder = list(res._paths)[0]
                log.progress(f"exporting parameters from '{dumpfolder}' dumpfolder")

        # ----- get list of paths to fit-objects
        useres = getattr(res, dumpfolder)
        useres.scan_folder()
        fitlist = list(islice(useres.dump_files, use_nfiles))

        # load the first fit-object to pre-load fn-coefficients
        fit0 = useres.load_fit(0)

        fn_evals = None
        if pre_evaluate_fn_coefs:
            log.progress("... pre-evaluation of fn-coefficients")

            if isinstance(fit0, MultiFits):
                fn_evals = dict()
                for name, fit_cfg in fit0.accessor.config_fits.items():
                    if fit_cfg.int_Q is True:
                        fn_evals[name] = fit_cfg.R._fnevals
            elif fit0.int_Q is True:
                fn_evals = fit0.R._fnevals

        # ----- set postprocess and finalout functions
        func = partial(self._export_postprocess,
                       parameters=parameters,
                       metrics=metrics,
                       export_functions=export_functions,
                       model_keys=model_keys,
                       index_col=index_col,
                       _fnevals_input=fn_evals)

        finalout = partial(self._export_finalout,
                           descriptions=descriptions,
                           attrs=attributes,
                           concat_dim=index_col)

        # ----- run finalout generation
        out = self.run_finaloutput(
            ncpu=ncpu,
            use_N_files=use_nfiles,
            use_config=use_config,
            finalout_name=None,
            finaloutput=finalout,
            postprocess=func,
            print_progress=True,
            logfile_level=1,
            fitlist=fitlist
        )

        # ----- attach model definition strings as attributes
        if isinstance(fit0, MultiFits):
            # remove dataset since we are not interested in site-specific infos
            fit0.set_dataset(None)
            for key, val in out.items():
                val.attrs['model_definition'] = (
                    fit0.accessor.config_fits[key]._model_definition)
        else:
            fit0.dataset = None
            out.attrs['model_definition'] = fit0._model_definition

        # if savepath is provided, save NetCDF files to disc
        if savepath:
            savepath = Path(savepath)
            if isinstance(out, dict):
                parent = savepath.parent
                name = savepath.stem
                suffix = savepath.suffix
                for key, val in out.items():
                    val.to_netcdf(parent / (name + "__" + key + suffix))
            else:
                out.to_netcdf(savepath)

        return out



import multiprocessing as mp
from timeit import default_timer
from datetime import datetime, timedelta
from itertools import repeat, islice
from functools import partial
import ctypes
import sys
import traceback
from textwrap import dedent

from pathlib import Path
import shutil

import pandas as pd
from numpy.random import randint as np_randint

from .general_functions import dt_to_hms, update_progress, groupby_unsorted
from .rtparse import RT1_configparser
from .rtfits import load
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
            log.progress(msg.strip())
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
            log.progress(msg.strip())

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
        setup=True,
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
        setup : bool, optional
            indicator if `RTprocess.setup()` should be called during
            initialization or not. This is required for multiprocessing
            on Windows since `setup()` needs to be called outside of the
            `if __name__ == '__main__'` protector to ensure correct import
            of the used processing_config class.
            Notice that if setup is set to True (the default) the
            folder-structure etc. will be initialized as soon as the
            class object is created!
            The default is True.
        """

        self._config_path = config_path
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

        if setup:
            self.setup()

    def _listener_process(self, queue):

        # adapted from https://docs.python.org/3.7/howto/logging-cookbook.html
        # logging-to-a-single-file-from-multiple-processes
        if not hasattr(self, "dumppath"):
            log.error("listener process called without dumppath specified!")
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

    def setup(self):
        """
        perform necessary tasks to run a processing-routine
          - initialize the folderstructure (only from MainProcess!)
          - copy modules and .ini files (if copy=True) (only from MainProcess!)
          - load modules and set parent-fit-object
        """

        if self._config_path is not None and self._proc_cls is None:

            self.config_path = Path(self._config_path)
            assert self.config_path.exists(), (
                f"the file {self.config_path} " + "does not exist!"
            )

            self.cfg = RT1_configparser(self.config_path)

            # update specs with init_kwargs
            # remember warnings and log them later in case dumppath is changed
            for key, val in self.init_kwargs.items():
                if key in self.cfg.config["PROCESS_SPECS"]:
                    log.warning(
                        f'"{key} = {self.cfg.config["PROCESS_SPECS"][key]}" '
                        + "will be overwritten by the definition provided via "
                        + f'"init_kwargs": "{key} = {val}" '
                    )
                    # update the parsed config (for import of modules etc.)
                    self.cfg.config["PROCESS_SPECS"][key] = val

            specs = self.cfg.get_process_specs()

            self.dumppath = specs["save_path"] / specs["dumpfolder"]

            if mp.current_process().name == "MainProcess":
                # init folderstructure and copy files only from main process
                if self.autocontinue is False:

                    if self.dumppath.exists():

                        def remove_folder():

                            # try to close existing filehandlers to avoid
                            # errors if the files are located in the folder
                            # if they are actually in the cfg folder, delete
                            # the corresponding handler as well!
                            for h in log.handlers:
                                try:
                                    h.close()
                                    if Path(h.baseFilename).parent.samefile(
                                        self.dumppath / "cfg"
                                    ):
                                        log.removeHandler(h)
                                except Exception:
                                    pass

                            # remove folderstructure
                            shutil.rmtree(self.dumppath)

                            log.info(
                                f'"{self.dumppath}"'
                                + "\nhas successfully been removed.\n"
                            )

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
                                    remove_folder,
                                ]
                            },
                        )

                # initialize the folderstructure
                _make_folderstructure(
                    specs["save_path"] / specs["dumpfolder"],
                    ["results", "cfg", "dumps"],
                )

                if self.copy is True:
                    self._copy_cfg_and_modules()

            # load the processing-class
            if "processing_cfg_module" in specs:
                proc_module_name = specs["processing_cfg_module"]
            else:
                proc_module_name = "processing_cfg"

            if "processing_cfg_class" in specs:
                proc_class_name = specs["processing_cfg_class"]
            else:
                proc_class_name = "processing_cfg"

            # load ALL modules to ensure that the importer finds them
            procmodule = self.cfg.get_all_modules(load_copy=self.copy)[proc_module_name]

            log.debug(
                f'processing config class "{proc_class_name}"'
                + f'  will be imported from \n"{procmodule}"'
            )

            self.proc_cls = getattr(procmodule, proc_class_name)(**specs)

            # get the parent fit-object
            if self._parent_fit is None:
                self.parent_fit = self.cfg.get_fitobject()
            else:
                self.parent_fit = self._parent_fit
        else:
            self.dumppath = None

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
        if (copypath).exists():
            log.warning(
                f'the file \n"{copypath / self.cfg.configpath.name}"\n'
                + "already exists... NO copying is performed and the "
                + "existing one is used!\n"
            )
        else:
            if len(self.init_kwargs) == 0:
                # if no init_kwargs have been provided, copy the
                # original file
                shutil.copy(self.cfg.configpath, copypath.parent)
                log.info(
                    f'"{self.cfg.configpath.name}" copied to\n' + f'"{copypath.parent}"'
                )
            else:
                # if init_kwargs have been provided, write the updated
                # config to the folder
                with open(copypath.parent / self.cfg.configpath.name, "w") as file:
                    self.cfg.config.write(file)

                log.info(
                    f'the config-file "{self.cfg.configpath}" has been'
                    + " updated with the init_kwargs and saved to"
                    + f'"{copypath.parent / self.cfg.configpath.name}"'
                )

            # remove the config and re-read the config from the copied path
            del self.cfg
            self.cfg = RT1_configparser(copypath)

        # copy modules
        for key, val in self.cfg.config["CONFIGFILES"].items():
            if key.startswith("module__"):
                modulename = key[8:]

                module_path = self.cfg.config["CONFIGFILES"][f"module__{modulename}"]

                location = Path(module_path.strip())

                copypath = self.dumppath / "cfg" / location.name

                if copypath.exists():
                    log.warning(
                        f'the file \n"{copypath}" \nalready '
                        + "exists ... NO copying is performed "
                        + "and the existing one is used!\n"
                    )
                else:
                    shutil.copy(location, copypath)
                    log.info(f'"{location.name}" copied to \n"{copypath}"')

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
            # initialize a new fits-object and perform the fit
            fit = self.parent_fit.reinit_object(dataset=dataset)
            fit.performfit()

            # append auxiliary data
            if aux_data is not None:
                fit.aux_data = aux_data

            # append reader_arg
            fit.reader_arg = reader_arg

            if process_cnt is not None:
                _increase_cnt(process_cnt, start, err=False)

            # dump a fit-file
            if self._dump_fit:
                self.proc_cls.dump_fit_to_file(fit, reader_arg, mini=True)

            # if a post-processing function is provided, return its output,
            # else return None
            if self._postprocess and callable(self.proc_cls.postprocess):
                ret = self.proc_cls.postprocess(fit, reader_arg)
            else:
                ret = None

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

        # preload ALL modules to ensure that the importer finds them at the
        # desired locations (e.g. the cfg-folders)
        _ = self.cfg.get_all_modules(load_copy=self.copy)

        if queue is not None:
            self._worker_configurer(queue)

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

        if self.parent_fit.int_Q is True:
            # pre-evaluate the fn-coefficients if interaction terms are used
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

            # call the initializer if it has been provided
            if "initializer" in pool_kwargs:
                if "initargs" in pool_kwargs:
                    pool_kwargs["initializer"](*pool_kwargs["initargs"])
                else:
                    pool_kwargs["initializer"]()
            res = []
            for reader_arg in reader_args:
                res.append(
                    self._evalfunc(reader_arg=reader_arg, process_cnt=process_cnt)
                )

        if self._postprocess and callable(self.proc_cls.finaloutput):
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
                with open(self.dumppath / "cfg" / "model_definition.txt", "w") as file:

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

            return self._run_finalout(
                ncpu,
                use_N_files=use_N_files,
                finalout_name=finalout_name,
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

    def _run_postprocess(self, fitpath, process_cnt=None):
        if process_cnt is not None:
            start = _start_cnt()

        # load fitobjects based on fit-paths
        try:
            fit = load(fitpath)
        except Exception:
            log.error(f"there was something wrong with {fitpath.name}")
            _increase_cnt(process_cnt, start, err=True)

        try:
            # run postprocessing
            ret = self.proc_cls.postprocess(fit, fit.reader_arg)

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

    def _run_finalout(
        self,
        ncpu,
        use_N_files=None,
        finalout_name=None,
        finaloutput=None,
        postprocess=None,
        queue=None,
        print_progress=True,
    ):

        # override finalout_name
        if finalout_name is not None:
            assert isinstance(finalout_name, str), "finalout_name must be a string"
            self.proc_cls.finalout_name = finalout_name

        # override finaloutput function
        if finaloutput is not None:
            assert callable(finaloutput), "finaloutput must be callable!"
            self.proc_cls.finaloutput = partial(finaloutput, self.proc_cls)
        # override postprocess function
        if postprocess is not None:
            assert callable(postprocess), "postprocess must be callable!"
            self.proc_cls.postprocess = partial(postprocess, self.proc_cls)

        # initialize RTresults class
        useres = getattr(RTresults(self.dumppath), self.proc_cls.dumpfolder)

        # get dump-files
        dumpiter = useres.dump_files

        if use_N_files is not None:
            fitlist = list(islice(dumpiter, use_N_files))
        else:
            fitlist = list(dumpiter)

        # add provided initializer and queue (used for subprocess-logging)
        # to the initargs and use "self._initializer" as initializer-function
        # Note: this is a pickleable way for decorating the initializer!
        pool_kwargs = dict(initializer=self._initializer, initargs=[None, queue])

        if print_progress is True:
            # initialize shared values that will be used to track the number
            # of completed processes and the mean time to complete a process
            process_cnt = _setup_cnt(N_items=len(fitlist), ncpu=ncpu)
        else:
            process_cnt = None

        if ncpu > 1:
            log.info(f"start of finalout generation on {ncpu} cores")
            with mp.Pool(ncpu, **pool_kwargs) as pool:
                # loop over the reader_args
                res_async = pool.starmap_async(
                    self._run_postprocess, zip(fitlist, repeat(process_cnt))
                )

                pool.close()  # Marks the pool as closed.
                pool.join()  # Waits for workers to exit.
                res = res_async.get()
        else:
            log.info("start of single-core finalout generation")

            # call the initializer if it has been provided
            if "initializer" in pool_kwargs:
                if "initargs" in pool_kwargs:
                    pool_kwargs["initializer"](*pool_kwargs["initargs"])
                else:
                    pool_kwargs["initializer"]()
            res = []
            for fitpath in fitlist:
                res.append(
                    self._run_postprocess(fitpath=fitpath, process_cnt=process_cnt)
                )

        if callable(self.proc_cls.finaloutput):
            return self.proc_cls.finaloutput(res)
            log.info("finished generation of finaloutput!")
        else:
            return res


class RTresults(object):
    """
    A class to provide easy access to processed results.
    On initialization the class will traverse the provided "parent_path"
    and recognize any sub-folder that matches the expected folder-structure
    as a sub-result.


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
        ...     sub_RESULT2
        ...         ....


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

        if all(
            i in [i.stem for i in self._parent_path.iterdir()]
            for i in ["cfg", "results", "dumps"]
        ):
            self._paths[self._parent_path.stem] = self._parent_path
            log.info(f"... adding result {self._parent_path.stem}")
            setattr(
                self,
                self._parent_path.stem,
                self._RT1_fitresult(self._parent_path.stem, self._parent_path),
            )

        for p in self._parent_path.iterdir():
            if p.is_dir():
                if all(
                    i in [i.stem for i in p.iterdir()]
                    for i in ["cfg", "results", "dumps"]
                ):
                    self._paths[p.stem] = p
                    log.info(f"... adding result {p.stem}")
                    setattr(self, p.stem, self._RT1_fitresult(p.stem, p))

    class _RT1_fitresult(object):
        def __init__(self, name, path):
            self.name = name
            self.path = Path(path)
            self._result_path = self.path / "results"
            self._dump_path = self.path / "dumps"
            self._cfg_path = self.path / "cfg"

            self._nc_paths = self._get_results(".nc")

        def _get_results(self, ending):
            assert self._result_path.exists(), (
                f"{self._result_path}" + " does not exist"
            )

            results = {
                i.stem: i for i in self._result_path.iterdir() if i.suffix == ending
            }

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
                         (might take some time for very large amounts of files)
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
                    if not hasattr(self, "_n_dump_files"):
                        self._n_dump_files = len(list(self.dump_files))
                    Nid = np_randint(0, self._n_dump_files)

                    filepath = next(islice(self.dump_files, Nid, None))

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
            a generator of the available dump-files
            """
            return (
                i
                for i in self._dump_path.iterdir()
                if i.suffix == ".dump" and "error" not in i.stem
            )

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

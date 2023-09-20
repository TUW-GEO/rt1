import multiprocessing as mp
import threading

import ctypes
from timeit import default_timer
from datetime import timedelta
import sys
import pandas as pd

from queue import Empty as QueueEmpty
import os
from pathlib import Path

from contextlib import contextmanager
from collections import defaultdict
from itertools import count
import signal

from . import log
from .rtresults import HDFaccessor
from .general_functions import (
    dt_to_hms,
    update_progress,
    groupby_unsorted,
    isidentifier,
    find_missing,
)


class RepeatTimer(threading.Thread):
    """
    A simple timer that executes a function after a given amount of time.
    (as an asynchronous task)

    to setup a scheduled execution of a given function use:

        >>> r = RepeatTimer(1, lambda: print("hello"))
        >>> r.start()

    to stop the execution use:

        >>> r.finished.set()
    """

    def __init__(self, interval, function, args=None, kwargs=None):
        threading.Thread.__init__(self, name="RTprocess_print_thread")
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = threading.Event()

    def cancel(self):
        """Stop the timer if it hasn't finished yet."""
        self.finished.set()

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class RT1_processor(object):
    def __init__(
        self,
        n_worker=1,
        n_combiner=2,
        n_writer=1,
        dst_path=None,
        HDF_kwargs=None,
        write_chunk_size=1000,
        out_cache_size=10,
        min_itemsize=None,
    ):
        """
        A class that takes care of combining results and writing them
        to disk as a HDF-container.

        Parameters
        ----------
        n_worker, n_combiner, n_writer: int
            the number of workers, combiners and writers to use
        dst_path : str
            the path where the HDF-store will be saved
        HDF_kwargs : dict, optional
            a dict with keyword-arguments passed to the HDF-store initialization
            - see `pandas.HDFstore` for details
        write_chunk_size : int, optional
            the chunk-size for writing results to dict.
            e.g.: a set of N results is combined and the combined output
             is written to disc to reduce io
            The default is 1000.
        out_cache_size : int, optional
            the max. number of results cached for writing.
            e.g.: the number of combined results scheduled to be written to
            disc. once this treshold is reached, processing is throttled
            to protect memory-overload...
            The default is 10.
        min_itemsize : dict
            a dict with the minimum-number of characters used to save string-columns
            within pytables.
            for details, check:

               - https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#storing-types
               - https://github.com/PyTables/PyTables/issues/48

            NOTE: variable-length strigns are not supported by pytables!
                  -> in case variable lenght strings are encountered, you MUST specify
                  min_itemsize to avoid truncating the strings!

            the default is set to {"index": 99, "values": 99} which results in a
            minimum length of 99 characters for any strings encountered in indexes
            or values of the saved DataFrames
        """
        if min_itemsize is None:
            self.min_itemsize = {"index": 99, "values": 99}
        else:
            self.min_itemsize = min_itemsize

        signal.signal(signal.SIGINT, self.manual_shutdown)
        signal.signal(signal.SIGTERM, self.manual_shutdown)

        self.n_worker = n_worker
        self.n_combiner = n_combiner
        self.n_writer = n_writer

        self.dst_path = dst_path

        self.HDF_kwargs = dict(complevel=1, complib="blosc")
        if HDF_kwargs is not None:
            self.HDF_kwargs.update(HDF_kwargs)

        self.write_chunk_size = write_chunk_size
        self.out_cache_size = out_cache_size

    def manual_shutdown(self, *args, **kwargs):
        log.error("shutting down due to SIGTERM event")

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self.should_stop.set()
        self._stop.set()
        raise (SystemExit)

    def _increase_cnt(self, err=False):
        self.lock.acquire()
        if err is False:
            # increase the total counter
            self.p_totcnt.value += 1
            # increase the counter for successful fits
            self.p_meancnt.value += 1
        else:
            # only increase the total counter
            self.p_totcnt.value += 1
        self.lock.release()

    def print_progress(self):
        # shut down the print-thread in case should_stop is set and the queue is empty
        if (self.should_stop.is_set() and self.queue.empty()) or self._stop.is_set():
            threading.current_thread().cancel()
            print()  # print a newline

        self.lock.acquire()
        qsize = self.queue.qsize()
        p_totcnt = self.p_totcnt.value + qsize
        p_meancnt = self.p_meancnt.value + qsize
        self.lock.release()

        if self.p_meancnt.value == 0:
            title = f"{'estimating time ...':<28}"
            p_max = self.p_max.value if self.p_max.value else p_totcnt + 1
            meantime = 0
        else:
            end = default_timer()
            meantime = (end - self.p_start) / (p_meancnt)

            if self.p_max.value is None:
                p_max = p_totcnt + 1
                remain = timedelta(seconds=meantime * 1000)
                timesuffix = "for 1000 fits"
            else:
                p_max = self.p_max.value
                remain = timedelta(seconds=meantime * (p_max - p_totcnt))
                timesuffix = "remaining"

            d, h, m, s = dt_to_hms(remain)
            title = f"~ {d} {h:02}:{m:02}:{s:02} {timesuffix}"

        msg = update_progress(
            p_totcnt,
            p_max,
            title=title,
            finalmsg=(
                "finished! "
                + f"({p_max} [{self.p_totcnt.value - self.p_meancnt.value}] fits)"
            ),
            progress2=p_totcnt - p_meancnt,
        )

        sys.stdout.write(
            msg + f" Q={self.queue.qsize():03}" + f" O={self.out_queue.qsize():02}"
        )
        sys.stdout.flush()

    def _combine_results(self, results):
        outdict = dict()
        for key, val in results.items():
            if len(val) == 0:
                continue
            if isinstance(val[0], (pd.Series, pd.DataFrame)):
                outdict[key] = pd.concat(val)
            else:
                outdict[key] = pd.Series(val)

        self.out_queue.put(outdict)

    def _combiner_process(self, init_func=None, init_args=None):
        """
        combine results before writing them to the HDF-store

        any key starting with "const__" will be treated as a constant
        property that is maintained for all configs of a MultiConfig object

        """
        if callable(init_func) and init_func is not None:
            if init_args is None:
                init_func()
            else:
                init_func(*init_args)

        c = count()  # a counter to name threads
        threads = []
        while True:
            if self.should_stop.is_set():
                if not self.queue.empty():
                    log.progress(
                        "... combining remaining "
                        + f"{self.queue.qsize()} processed results"
                    )
                else:
                    log.progress("shutting down " + mp.current_process().name)
                    break
            if self._stop.is_set():
                log.progress("shutting down " + mp.current_process().name)
                break

            try:
                # caches for results
                results = defaultdict(list)
                # results counter
                nres = 0

                # append more results until cache-size is reached
                while nres < self.write_chunk_size:
                    try:
                        # get values from the queue, timeout after 5 seconds
                        val = self.queue.get(timeout=5)
                        if val is None:
                            log.debug("ignored a None in the queue")
                            self._increase_cnt(err=True)
                            continue
                    except QueueEmpty:
                        break

                    for cfg, cfg_results in val.items():
                        if cfg.startswith("const__"):
                            results[cfg.lstrip("const__")].append(cfg_results)
                        elif isinstance(cfg_results, dict):
                            for cfg_key, cfg_val in cfg_results.items():
                                results[cfg + "/" + cfg_key].append(cfg_val)
                        else:
                            results[cfg].append(cfg_results)

                    nres += 1
                    self._increase_cnt(err=False)

                if len(results) > 0:
                    t = threading.Thread(
                        target=self._combine_results,
                        args=(results,),
                        name=f"{mp.current_process().name} - {next(c)}",
                    )
                    t.start()
                    threads.append(t)

            except Exception:
                log.error("There was a problem combining results:", exc_info=True)

        # wait for all threads to finish before exiting the combiner process
        for n, t in enumerate(threads):
            if t.is_alive():
                log.progress(f"joining remaining thread... {t.name}")
                t.join()

    def _check_IDs(self, key="reader_arg", get_ID=None):
        """
        return a list of IDs that are not yet present in the hdf-container
        located at "dst_path"

        Parameters
        ----------
        key : str, optional
            the key to use in the HDF-container. The default is "reader_arg".
        get_ID : callable, optional
            a custom callable that extracts the ID from the provided
            IDs list.
            The default is None in which case the following function is used:

                >>> # extract filename from given path
                >>> lambda i: i.split(os.sep)[-1].split(".")[0]

        Returns
        -------
        new_IDs : set
            a set containing the unique elements of "inp" whose IDs are not
            already present in the output HDF container.

        """

        if Path(self.dst_path).exists():
            log.progress("evaluating list of IDs to process...")
            if get_ID is None:

                def get_ID(i):
                    ID = str(i["ID"]).split(os.sep)[-1].split(".")[0]
                    if not isidentifier(str(ID)):
                        return f"RT1_{ID}"
                    else:
                        return str(ID)

            with HDFaccessor(self.dst_path) as fit_db:
                # get all IDs that are already present in the HDF store
                # found_IDs = store.select_column("reader_arg", "ID").values
                # get a list of integers that have already been assigned to IDs
                # found_ID_nums = store.select_column("reader_arg", "index").values

                found_ID_nums = fit_db.IDs.index
                found_IDs = fit_db.IDs.ID.values

                # a list of bool's that indicate if the ID is already processed
                process_Q = pd.Index(map(get_ID, self.arg_list)).isin(found_IDs)

                # a counter that yields unique IDs that are not yet assigned
                # in the HDF store
                id_counter = find_missing(found_ID_nums)

                # set the processing-args
                # update ID to be a valid python-identifier
                self.args_to_process = [
                    {**arg, "_RT1_ID_num": next(id_counter)}
                    for arg, q in zip(self.arg_list, process_Q)
                    if not q
                ]
        else:
            self.args_to_process = [
                {**i, "_RT1_ID_num": n}
                for i, n in zip(self.arg_list, range(len(self.arg_list)))
            ]

            log.info("no existing output-HDF file found...")
            log.progress(f"processing all {len(self.args_to_process)} IDs!")
            return
        log.progress(
            f"found {len(self.args_to_process)} missing and "
            + f"{len(self.arg_list) - len(self.args_to_process)}"
            + " existing IDs!"
        )

    def _save_file_process(self, init_func=None, init_args=None):
        log_first_expected_rows = defaultdict(lambda: True)

        if callable(init_func) and init_func is not None:
            if init_args is None:
                init_func()
            else:
                init_func(*init_args)

        while True:
            # continue until the out_queue is emptied, then break
            if self.combiner_ready.is_set():
                if not self.out_queue.empty():
                    log.progress(
                        f"writing remaining {self.out_queue.qsize()}"
                        + " cached results to disk"
                    )
                else:
                    log.progress("shutting down " + mp.current_process().name)
                    break

            if self._stop.is_set():
                log.progress("shutting down " + mp.current_process().name)
                break

            # pause processing in case out_cache_size is reached
            if self.out_queue.qsize() < self.out_cache_size:
                self.should_wait.set()
            elif self.should_wait.is_set():
                self.should_wait.clear()
                log.warning("\nqueue full... waiting")

            # wait for results to appear in the out_queue
            # timeout after 2 sec to check for break-condition
            try:
                out = self.out_queue.get(timeout=2)
            except QueueEmpty:
                continue

            try:
                self.write_lock.acquire()
                with pd.HDFStore(self.dst_path, "a", **self.HDF_kwargs) as store:
                    for key, val in out.items():
                        # make sure the ID column is indexed for the reader_args
                        if key == "reader_arg":
                            data_columns = ["ID"]
                        else:
                            data_columns = []
                        # don't index data-columns
                        # don't index right away (will be done once all processes are
                        # finished to maintain write-speed during processing)

                        # evaluate the number of expected rows since it is
                        # "highly recommended" to do this for very large datasets
                        if self.p_max.value is None:
                            expectedrows = None
                        else:
                            expectedrows = self.p_max.value * len(val)

                        store.append(
                            key,
                            val,
                            format="t",
                            data_columns=data_columns,
                            index=False,
                            min_itemsize=self.min_itemsize,
                            expectedrows=expectedrows,
                        )

                        if log_first_expected_rows[key]:
                            log.debug(
                                "expecting to get "
                                + f"{self.p_max.value * len(val)} rows "
                                + "in the HDF5-container for "
                                + f"'{key}' based on {self.p_max.value}"
                                + " results to process"
                            )
                            # don't log this multiple times
                            # (even though it might change depending on
                            #  the length of the dataframes to append)!
                            log_first_expected_rows[key] = False

                self.write_lock.release()
            except Exception:
                log.error("problem while writing data... exiting", exc_info=True)
                self._stop.set()
                self.should_wait.clear()
                self.write_lock.release()

    def _worker_process(self, *args):
        self.should_wait.wait()

        try:
            if self.reader_func is not None:
                # use provided reader-func...
                data = self.reader_func(args[0])
            else:
                data = args[0]

            if data is not None:
                res = self.process_func(data, *args[1:])
                self.queue.put(res)
            else:
                log.warning(f"loading {args} resulted in None... skipping")
        except Exception:
            # put None in the queue to indicate an erroneous fit
            self.queue.put(None)

            print()
            log.error(f"problem while processing \n{args}", exc_info=True)

    def _start_writer_process(self, init_func=None, init_args=None):
        # define a process that writes the results to disc
        procs = []
        for i in range(self.n_writer):
            ctx = mp.get_context("spawn")
            writer = ctx.Process(
                target=self._save_file_process,
                args=(init_func, init_args),
                name=f"RTprocess writer {i}",
            )
            procs.append(writer)
            log.progress(f"starting {writer.name}")
            writer.start()
        return procs

    def _start_combiner_process(self, init_func=None, init_args=None):
        procs = []
        for i in range(self.n_combiner):
            # start thread to combine the results
            ctx = mp.get_context("spawn")
            t = ctx.Process(
                target=self._combiner_process,
                args=(init_func, init_args),
                name=f"RTprocess combiner {i}",
            )
            procs.append(t)
            # combiner_thread.setDaemon(True)
            log.progress(f"starting {t.name}")
            t.start()
        return procs

    def _setup_manager(self):
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()

        self.queue = manager.Queue()
        self.out_queue = manager.Queue()

        self.should_wait = manager.Event()
        self.should_stop = manager.Event()
        self._stop = manager.Event()

        self.lock = manager.Lock()
        self.write_lock = manager.Lock()

        self.p_totcnt = manager.Value(ctypes.c_ulonglong, 0)
        self.p_meancnt = manager.Value(ctypes.c_ulonglong, 0)

        self.combiner_ready = manager.Event()

        self.p_max = manager.Value(ctypes.c_ulonglong, None)

        return manager

    @classmethod
    @contextmanager
    def writer_pool(cls, init_func=None, init_args=None, *args, **kwargs):
        try:
            w = cls(*args, **kwargs)
            manager = w._setup_manager()
            writer = w._start_writer_process(init_func=init_func, init_args=init_args)
            combiner = w._start_combiner_process(
                init_func=init_func, init_args=init_args
            )

            yield w
        finally:
            w.stop(writer, combiner, manager)

    def start_print_thread(self):
        # start the timer for estimating the processing-time
        self.p_start = default_timer()

        print_thread = RepeatTimer(0.25, self.print_progress)
        print_thread.start()
        return print_thread

    def run_starmap(
        self,
        arg_list=None,
        process_func=None,
        reader_func=None,
        pool_kwargs=None,
        starmap_args=None,
        ID_getter=None,
    ):
        """


        Parameters
        ----------
        arg_list : iterable, optional
            list of arguments to be processed. if the HDF store already exists,
            a check for existing IDs will be performed prior to processing and only
            the non-existing IDs will be used!
            The default is None.
        process_func : callable, optional
            the function to use for obtaining the results.
            -> must return a dict of pandas.DataFrames !
            The default is None.
        reader_func : callable, optional
            optional explicit specification of a reader-function.
            If None, the arguments are directly forwarded to process_func
            The default is None.
        pool_kwargs : dict, optional
            a dict of kwargs passed to the initialization of the multiprocessing.Pool.
            The default is None.
        starmap_args : iterable, optional
            a list of additional arguments passed to the evaluation of
            "process_func". The default is None.

        Returns
        -------
        res : iterable
            the final output of the process.

        """

        if pool_kwargs is None:
            pool_kwargs = dict()
        if starmap_args is None:
            starmap_args = []

        self.arg_list = arg_list
        self.reader_func = reader_func
        self.process_func = process_func

        assert self.arg_list is not None, 'you must provide "arg_list"'

        self._check_IDs(get_ID=ID_getter)
        self.p_max.value = len(self.args_to_process)

        if self.p_max.value == 0:
            log.error("ALL IDs are already present in the HDF file!")
            return

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(self.n_worker, **pool_kwargs)
        # with ctx.Pool(self.n_worker, **pool_kwargs) as pool:
        try:
            worker = pool.starmap_async(
                self._worker_process,
                zip(self.args_to_process, *starmap_args),
            )

            # do this after calling starmap_async to wait for the initializers
            # to finish!
            _ = self.start_print_thread()

            res = worker.get()

            print()  # print a newline
            d, h, m, s = dt_to_hms(timedelta(seconds=default_timer() - self.p_start))
            log.progress(
                "finished processing "
                + f"{len(self.args_to_process)} [{self.p_totcnt.value}] IDs!"
                + f" ... it took {d} {h:02}:{m:02}:{s:02}"
            )
        finally:
            pool.close()
            pool.join()

        return res

    def stop(self, writer, combiner, manager):
        # initialize shutdown
        self.should_stop.set()
        print()  # newline

        # wait for cached results to be combined
        for c in combiner:
            c.join()
        self.combiner_ready.set()

        # wait for output-cache to be written to disc
        for w in writer:
            w.join()

        # tell all processes to stop
        self._stop.set()

        manager.shutdown()

        if hasattr(self, "p_start"):
            d, h, m, s = dt_to_hms(timedelta(seconds=default_timer() - self.p_start))
            log.progress(f"finished! ... it took {d} {h:02}:{m:02}:{s:02}")
        else:
            log.progress("exiting...")

    @staticmethod
    def create_index(dst_path, idx_levels=None, keys=None, kind="full", optlevel=6):
        """
        create an index for the given HDF-store to speed up querying.
        (the index is stored to disk, so this only needs to be executed once!)

        NOTICE: this can take quite some time for large datasets and
                complex indexes (e.g. MultiIndex etc.) !

        Parameters
        ----------
        dst_path : str or Path
            the path to the HDF-store.
        idx_levels : iterable, "all" or None, optional
            if iterable: a list of index-levels to use when creating the index.
                         only levels actually present in a dataset are
                         considered! multiindexes are also possible,
                         e.g.: (["ID", "date"])
            if "all": all available index-levels found in the dataset are used
            if None: only the first index-level found in the dataset is used
            The default is None.
        keys : iterable, optional
            the keys to create an index for.
            The default is None, in which case all keys are used!.
        optlevel : int or None, default None
            passed to "pandas.io.pytables.---FrameTable.create_index()"
            Optimization level [1-9], if None, pytables defaults to 6.

        kind : str or None, default None
            passed to "pandas.io.pytables.---FrameTable.create_index()"
            Kind of index, if None, pytables defaults to "medium".
            can be one of ["ultralight", "light", "medium", "full"]

        """
        with pd.HDFStore(dst_path, "a") as store:
            use_keys = keys if keys else store.keys()
            use_keys = [key for key in use_keys if not key.endswith("/meta")]
            log.progress(
                f"attempting to create '{kind}' indexes with "
                + f"optlevel {optlevel} for the columns:\n"
                + "    \n".join(use_keys)
            )
            for key in use_keys:
                s = store.get_storer(key)
                s_levels = s.levels

                if idx_levels is None:
                    try:
                        # if s_levels is iterable and idx_cols is None,
                        # use the first level
                        iter(s_levels)
                        use_levels = [s.levels[0]]
                    except TypeError:
                        # else use whatever the level-identification available
                        use_levels = [s.levels]

                elif idx_levels == "all":
                    try:
                        # if s_levels is iterable and idx_cols is None,
                        # use all available levels
                        iter(s_levels)
                        use_levels = s.levels
                    except TypeError:
                        # else use whatever the level-identification available
                        use_levels = s.levels

                else:
                    use_levels = list(set(idx_levels) & set(s_levels))

                if len(use_levels) > 0:
                    log.progress(f'creating index-levels "{use_levels}" for "{key}"')
                    s.create_index(
                        list(map(str, use_levels)),
                        kind=kind,
                        optlevel=optlevel,
                    )
                else:
                    log.warning(f'no index-level found for "{key}"... skipping')
        log.progress("finished index-creation")

    @staticmethod
    def get_data(dst_path, key, config=None, **kwargs):
        """
        read data from the associated HDF file

        Parameters
        ----------
        key : str
            the key to use.
        config : str, optional
            the config to use.
            The default is None, in which case "constant" keys are accessed
        **kwargs :
            kwargs passed to `pd.HDFStore.select()`.
            (e.g. where, start, stop, columns, iterator, chunksize)
        Returns
        -------
        res : pd.DataFrame
            the extracted dataset.

        """
        with pd.HDFStore(dst_path, "r") as store:
            keysplits = [key.lstrip("/").split("/") for key in store]
            cfg_keys = (i for i in keysplits if len(i) > 1)
            cfg_keys = groupby_unsorted(
                cfg_keys, key=lambda x: x[0], get=lambda x: x[1], sort=True
            )
            const_keys = [i[0] for i in keysplits if len(i) == 1]

            if config is None:
                assert key in const_keys, (
                    f"'{key}' not found in HDF-file\n"
                    + "... available constant layers are"
                    + f"\n    {const_keys}\n"
                    + "... available config layers are \n    "
                    + "\n    ".join(f"{key}: {val}" for key, val in cfg_keys.items())
                )

            usekey = f"{config}/{key}"
            res = store.select(usekey, **kwargs)
        return res

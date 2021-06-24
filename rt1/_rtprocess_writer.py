import multiprocessing as mp
from threading import Thread, current_thread, Timer

import ctypes
from timeit import default_timer
from datetime import timedelta
import sys
import pandas as pd
import traceback

from queue import Empty as QueueEmpty
import os
from pathlib import Path

from contextlib import contextmanager
from collections import defaultdict
from itertools import count
import signal

from . import log
from .general_functions import dt_to_hms, update_progress, groupby_unsorted


class RepeatTimer(Timer):
    """
    A simple timer that executes a function after a given amount of time.
    (as an asynchronous task)

    to setup a scheduled execution of a given function use:

        >>> r = RepeatTimer(1, lambda: print("hello"))
        >>> r.start()

    to stop the execution use:

        >>> r.finished.set()
    """
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class RT1_processor(object):
    def __init__(self,
                 n_worker=1, n_combiner=1, n_writer=1,
                 dst_path=None, HDF_kwargs=None,
                 write_chunk_size=1000, out_cache_size=10):
        """
        A class that takes care of combining results and writing them
        to disk as a HDF-container.

        Parameters
        ----------
        n_worker, n_combiner, n_writer: int
            the number of workers, combiners and writers to use
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
        """
        signal.signal(signal.SIGINT, self.manual_shutdown)
        signal.signal(signal.SIGTERM, self.manual_shutdown)


        self.n_worker = n_worker
        self.n_combiner = n_combiner
        self.n_writer = n_writer

        self.dst_path = dst_path

        self.HDF_kwargs = dict(complevel=5, complib="blosc:zstd")
        if HDF_kwargs is not None:
            self.HDF_kwargs.update(HDF_kwargs)

        self.write_chunk_size = write_chunk_size
        self.out_cache_size = out_cache_size

    def manual_shutdown(self, *args, **kwargs):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("I'm shutting down")
        self.should_stop.set()
        self._stop.set()
        raise(SystemExit)

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
        self.lock.acquire()
        p_totcnt = self.p_totcnt.value
        p_meancnt = self.p_meancnt.value
        self.lock.release()

        if self.p_meancnt.value == 0:
            title = f"{'estimating time ...':<28}"
            p_max = 1
            meantime = 0
        else:
            end = default_timer()
            meantime = (end - self.p_start) / (p_meancnt)

            if self.p_max is None:
                p_max = p_totcnt + 1
                remain = timedelta(seconds=meantime * 1000)
                timesuffix = "for 1000 fits"
            else:
                p_max = self.p_max
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
            msg
            + f" Q={self.queue.qsize():03}"
            + f" O={self.out_queue.qsize():02}"
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

        #outdict = {key: pd.concat(val) for key, val in results.items()}
        # return outdict
        self.out_queue.put(outdict)
        #self.threads.remove(current_thread())

    def _combiner_process(self):
        c = count() # a counter to name threads
        threads = []
        while True:
            if self.should_stop.is_set():
                if not self.queue.empty():
                    print("... combining remaining processed files")
                else:
                    print("shutting down", mp.current_process().name)
                    break
            if self._stop.is_set():
                print("shutting down", mp.current_process().name)
                break

            try:
                # caches for results
                results = defaultdict(list)
                # results counter
                nres = 0

                # append more results until cache-size is reached
                while nres < self.write_chunk_size:
                    try:
                        val = self.queue.get(timeout=2)
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
                    t = Thread(target=self._combine_results, args=(results,),
                               name=f"{current_thread().name} - {next(c)}")
                    t.start()
                    threads.append(t)

            except Exception:
                print("There was a problem combining results:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # wait for all threads to finish before exiting the combiner process
        for n, t in enumerate(threads):
            if t.is_alive():
                print(f"joining remaining thread... {t.name}")
                t.join()

    def _check_IDs(self, key="reader_arg", get_ID = None):
        """
        return a list of IDs that are not yet present in the hdf-container
        located at "dst_path"

        Parameters
        ----------
        inp : iterable
            the list of IDs that are intended to be processed.
        key : str, optional
            the key to use in the HDF-container. The default is "static".
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
            print("evaluating list of IDs to process...")
            if get_ID is None:
                get_ID = lambda i: i.split(os.sep)[-1].split(".")[0]

            with pd.HDFStore(self.dst_path, "r") as store:
                keys = [i.lstrip("/") for i in store.keys()]
                if key not in keys:
                    try:
                        msg = f"key {key} not found in HDF-file..."
                        key = next(key for key in keys if key.endswith("static"))
                        print(msg, f"using key {key}")
                    except StopIteration:
                        self.args_to_process = self.arg_list
                        print("unable to determine existing IDs...",
                              f"processing all {len(self.args_to_process)} IDs!")
                        return

                found_IDs = pd.Index(map(get_ID, self.arg_list)).isin(store[key].index)
                self.args_to_process = [i for i, q in zip(self.arg_list, found_IDs) if not q]
        else:
            self.args_to_process = self.arg_list

        if len(self.args_to_process) == 0:
            print("All IDs are already present in the HDF file")
        else:
            print(f"found {len(self.args_to_process)} missing and",
                  f"{len(self.arg_list) - len(self.args_to_process)} existing IDs!")

    def _save_file_process(self):
        while True:
            # continue until the out_queue is emptied, then break
            if self.should_stop.is_set():
                if not self.out_queue.empty():
                    print(f"writing remaining {self.out_queue.qsize()}",
                          "cached results to disk")
                else:
                    print("shutting down", mp.current_process().name)
                    break

            if self._stop.is_set():
                print("shutting down", mp.current_process().name)
                break

            # pause processing in case out_cache_size is reached
            if self.out_queue.qsize() < self.out_cache_size:
                self.should_wait.set()
            elif self.should_wait.is_set():
                self.should_wait.clear()
                print("\nqueue full... waiting")


            # wait for results to appear in the out_queue
            # timeout after 2 sec to check for break-condition
            try:
                out = self.out_queue.get(timeout=2)
            except QueueEmpty:
                continue

            # if out is None:
            #     break
            try:
                self.write_lock.acquire()
                with pd.HDFStore(self.dst_path, "a", **self.HDF_kwargs) as store:
                    for key, val in out.items():
                        store.append(key,
                                     val,
                                     format="t",
                                     data_columns=[],   # don't index data-columns for disc-queries
                                     index=False,)      # don't index already (will be done once all processes are finished)
                self.write_lock.release()
            except Exception:
                print("problem while writing data... exiting")
                self._stop.set()
                self.should_wait.clear()
                self.write_lock.release()

    def _worker_process(self, arg):
            self.should_wait.wait()

            try:
                fit = self.reader_func(arg)

                if fit is not None:
                    res = self.process_func(fit)
                    self.queue.put(res)
                else:
                    print("loading", arg, "resulted in None... skipping")
            except Exception:
                print()
                log.error(f"problem while processing \n{arg}")

    def _start_worker_process(self):

        pool = mp.Pool(self.n_worker)
        # worker = pool.map_async(self._worker_process,
        #                         self.args_to_process,
        #                         chunksize=self.write_chunk_size)
        # worker.get()
        self.worker_pool = pool
        return pool

    def _start_writer_process(self):
        # define a process that writes the results to disc
        procs = []
        for i in range(self.n_writer):
            writer = mp.Process(target=self._save_file_process,
                                    name=f"RTprocess writer {i}")
            procs.append(writer)
            print(f"starting {writer.name}")
            writer.start()
        return procs

    def _start_combiner_process(self):
        procs = []
        for i in range(self.n_combiner):
            # start thread to combine the results
            t = mp.Process(target=self._combiner_process,
                           name=f"RTprocess combiner {i}")
            procs.append(t)
            # combiner_thread.setDaemon(True)
            print(f"starting {t.name}")
            t.start()
        return procs


    def _setup_manager(self):
        manager = mp.Manager()

        self.queue = manager.Queue()
        self.out_queue = manager.Queue()

        self.should_wait = manager.Event()
        self.should_stop = manager.Event()
        self._stop = manager.Event()

        self.lock = manager.Lock()
        self.write_lock = manager.Lock()

        self.p_totcnt = manager.Value(ctypes.c_ulonglong, 0)
        self.p_meancnt = manager.Value(ctypes.c_ulonglong, 0)

        return manager

    @classmethod
    @contextmanager
    def writer_pool(cls, *args, **kwargs):
        try:
            w = cls(*args, **kwargs)

            manager = w._setup_manager()
            writer = w._start_writer_process()
            combiner = w._start_combiner_process()

            yield w
        finally:
            w.stop(writer, combiner, manager)

    def run(self,
            arg_list=None,
            process_func=None,
            reader_func=None,
            pool_kwargs=None):

        if pool_kwargs is None:
            pool_kwargs = dict()

        self.arg_list = arg_list
        self.reader_func = reader_func
        self.process_func = process_func

        if self.arg_list is None:
            self.p_max = None
        else:
            self._check_IDs()
            self.p_max = len(self.args_to_process)


        # start the timer for estimating the processing-time
        self.p_start = default_timer()
        # print the progress every 0.25 seconds
        print_thread = RepeatTimer(.25, self.print_progress)
        print_thread.start()

        try:
            with mp.Pool(self.n_worker) as pool:
                worker = pool.map_async(self._worker_process,
                                        self.args_to_process,
                                        chunksize=self.write_chunk_size,
                                        **pool_kwargs)
                res = worker.get()

        finally:
            # stop the printer thread
            print_thread.finished.set()
            print() # print a newline after stopping the printer-thread

            d, h, m, s = dt_to_hms(timedelta(
                seconds=default_timer() - self.p_start))
            print(f"finished processing! ... it took {d} {h:02}:{m:02}:{s:02}")

        return res

    def stop(self, writer, combiner, manager):
        # initialize shutdown
        self.should_stop.set()
        print() # newline

        # wait for cached results to be combined
        for c in combiner:
            c.join()
        # wait for output-cache to be written to disc
        for w in writer:
            w.join()

        # tell all processes to stop
        self._stop.set()

        manager.shutdown()

        d, h, m, s = dt_to_hms(timedelta(
            seconds=default_timer() - self.p_start))
        print(f"finished! ... it took {d} {h:02}:{m:02}:{s:02}")

    @staticmethod
    def create_index(dst_path, id_col="ID", index_dates=False):
        with pd.HDFStore(dst_path, "a") as store:

            for key in ["init_dict", "reader_arg", "res_dict"]:
                if key in store:
                    print(f"creating index for {key}")
                    s = store.get_storer(key)
                    s.create_index([id_col])

            for key in ["dataset", "aux_data"]:
                if key in store:
                    print(f"creating index for {key}")
                    s = store.get_storer(key)
                    s.create_index([id_col, "date"] if index_dates else
                                   [id_col])

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
            cfg_keys = groupby_unsorted(cfg_keys,
                                    key=lambda x: x[0],
                                    get = lambda x: x[1],
                                    sort=True)
            const_keys = [i[0] for i in keysplits if len(i) == 1]

            if config is None:
                assert key in const_keys, (f"'{key}' not found in HDF-file\n" +
                                           "... available constant layers are" +
                                           f"\n    {const_keys}\n" +
                                           "... available config layers are \n    " +
                                           "\n    ".join(
                                               f"{key}: {val}"
                                               for key, val in cfg_keys.items()))

            usekey = f"{config}/{key}"
            res = store.select(usekey, **kwargs)
        return res























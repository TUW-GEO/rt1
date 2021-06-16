import multiprocessing as mp
from threading import Thread, current_thread, Timer

import ctypes
from timeit import default_timer
from datetime import timedelta
from rt1.general_functions import dt_to_hms, update_progress
import sys
import pandas as pd
import traceback

from queue import Empty as QueueEmpty
import os
from pathlib import Path

from collections import defaultdict
import signal
import h5py


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class _RT1_writer_class(object):
    def __init__(self, dst_path, arg_list=None, write_chunk_size=1000, out_cache_size=10):
        """

        Parameters
        ----------
        dst_path : str
            The path to store the output-file.
        arg_list : iterable, optional
            the list of arguments that are intended to be processed
            ... use `_RT1_writer_class.args_to_process` to get a list of
            unique args that are not yet present in the HDF-container
        write_chunk_size : int, optional
            the chunk-size for writing results to dict. The default is 1000.
        out_cache_size : int, optional
            the max. number of results cached for writing.

            (implemented to protect memory-overload... processing will wait
             for the writer to empty the queue if the cache-size is exceeded)
            The default is 10.
        """
        signal.signal(signal.SIGINT, self.manual_shutdown)
        signal.signal(signal.SIGTERM, self.manual_shutdown)

        self.dst_path = dst_path
        self.arg_list = arg_list

        if self.arg_list is None:
            self.p_max = None
        else:
            self._check_IDs()
            self.p_max = len(self.args_to_process)

        self.write_chunk_size = write_chunk_size
        self.out_cache_size = out_cache_size

        self._stop = False

        manager = mp.Manager()

        self.queue = manager.Queue()
        self.out_queue = manager.Queue()
        self.event = manager.Event()

        self.lock = manager.Lock()
        self.p_totcnt = manager.Value(ctypes.c_ulonglong, 0)
        self.p_meancnt = manager.Value(ctypes.c_ulonglong, 0)

        self.threads = []
        self.exit = False

    def manual_shutdown(self, *args, **kwargs):
        print("I'm shutting down")
        self.queue.put(None)
        self.out_queue.put(None)
        raise(SystemExit)

    @staticmethod
    def _start_cnt():
        return default_timer()

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
        try:
            self.p_start
        except AttributeError:
            self.p_start = self._start_cnt()

        #self.lock.acquire()
        p_totcnt = self.p_totcnt.value
        p_meancnt = self.p_meancnt.value

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
            title = f"approx. {d} {h:02}:{m:02}:{s:02} {timesuffix}"
        #self.lock.release()

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
            msg + f" (~{meantime:.5f} s/fit)"
            #+ f" Q={self.queue.qsize():03}, outQ={self.out_queue.qsize():02}, T={len(self.threads):02}, time={meantime:.5f}"
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
        self.threads.remove(current_thread())

    def _combiner_process(self):
        print_thread = RepeatTimer(.25, self.print_progress)
        print_thread.start()
        while True:
            try:
                # caches for results
                results = defaultdict(list)

                nres = 0
                # wait for the first result
                val = self.queue.get()
                if val is None:
                    self._stop = True
                else:
                    for cfg, cfg_results in val.items():
                        if cfg.startswith("const__"):
                            results[cfg.lstrip("const__")].append(cfg_results)
                        elif isinstance(cfg_results, dict):
                            for cfg_key, cfg_val in cfg_results.items():
                                results[cfg + "/" + cfg_key].append(cfg_val)
                        else:
                            results[cfg].append(cfg_results)

                    # # append the first result
                    # results["static"].append(val[0])
                    # results["dynamic"].append(val[1])

                    # append more results until cache-size is reached
                    while nres < self.write_chunk_size:
                        try:
                            val = self.queue.get(timeout=2)
                            if val is None:
                                self._stop = True
                                break
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
                        #self.print_progress()

                    t = Thread(target=self._combine_results, args=(results,))
                    t.start()
                    self.threads.append(t)

                if self._stop and self.queue.empty():
                    # make sure all threads have finished before stopping the
                    # writer process
                    print()
                    for n, t in enumerate(self.threads):
                        print(f"joining remaining threads... {n + 1}", end="\r")
                        t.join()
                    # also tell the writer to stop
                    self.out_queue.put(None)
                    print_thread.cancel()
                    break

            except Exception:
                print("Whoops! Problem:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

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

            with pd.HDFStore(self.dst_path, "a", complevel=4, complib="zlib") as store:
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
        print(f"found {len(self.args_to_process)} missing and",
              f"{len(self.arg_list) - len(self.args_to_process)} existing IDs!")

    def _save_file_process(self):
        with pd.HDFStore(self.dst_path, "a", complevel=4, complib="zlib") as store:
            while True:
                # pause processing in case out_cache_size is reached
                if self.out_queue.qsize() < self.out_cache_size:
                    self.event.set()
                elif self.event.is_set():
                    self.event.clear()
                    print("queue full... waiting")

                # wait for results to appear in the out_queue
                out = self.out_queue.get()
                if out is None:
                    break

                for key, val in out.items():
                    store.append(key, val, format="t", data_columns=True, index=False)

    def _start_writer_process(self):
        # define a process that writes the results to disc
        writer = mp.Process(target=self._save_file_process)
        writer.start()
        return writer

    def _start_combiner_process(self):
        # start thread to combine the results
        combiner_thread = mp.Process(target=self._combiner_process)
        # combiner_thread.setDaemon(True)
        combiner_thread.start()
        return combiner_thread

    @staticmethod
    def fit_to_hdf(fit, path):
        # with h5py.File(path, 'w') as f:
        #     for key, val in fit._get_init_dict().items():

        #         dset = f.create_dataset(key, data=val)




        #import numpy as np

        #Writing data
        #d1 = np.random.random(size = (1000,20))  #sample data
        hf = h5py.File(path, 'w')
        #dset1=hf.create_dataset('dataset_1', data=d1)
        #set some metadata directly
        #hf.attrs['metadata1']=5

        #sample dictionary object
        sample_dict={"metadata2":1, "metadata3":2, "metadata4":"blah_blah"}

        #Store this dictionary object as hdf5 metadata
        for k in fit._get_init_dict():
            hf.attrs[k]=sample_dict[k]

        hf.close()




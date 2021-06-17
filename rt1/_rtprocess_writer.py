import multiprocessing as mp
from threading import Thread, current_thread, Timer

import ctypes
from timeit import default_timer
from datetime import timedelta
from rt1.general_functions import dt_to_hms, update_progress, groupby_unsorted
import sys
import pandas as pd
import traceback

from queue import Empty as QueueEmpty
import os
from pathlib import Path

from collections import defaultdict
import signal
import h5py
from ast import literal_eval


from rt1.rtfits import Fits

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

        if len(self.args_to_process) == 0:
            print("All IDs are already present in the HDF file")
        else:
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

    @classmethod
    def connect(cls, self, path, arg_list, **kwargs):
        self._RT1_writer = cls(path, arg_list, **kwargs)

    def start(self):
        writer = self._start_writer_process()
        combiner = self._start_combiner_process()
        return writer, combiner

    def stop(self, writer, combiner):
        self.queue.put(None)
        print(f"waiting for {len(self.threads)} threads to join...")
        combiner.join()
        writer.join()

    def get_data(self, key, config=None, **kwargs):
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
        with pd.HDFStore(self.dst_path, "r") as store:
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
                                           "\n    ".join(f"{key}: {val}" for key, val in cfg_keys.items()))


            usekey = f"{config}/{key}"

            res = store.select(usekey, **kwargs)
        return res

    @staticmethod
    def fit_to_hdf(fit, path, overwrite=False, ID_key="ID",
                   save_results=True, save_data=True, save_auxdata=True):

        # defdicts are stored as categories to avoid saving the same a lot of times


        ID = fit.reader_arg[ID_key]


        with pd.HDFStore(path, "a", complevel=5, complib="zlib") as hf:

            if not overwrite and "init_dict" in hf:
                assert ID not in hf["init_dict"].index, "file already exists! (use overwrite=True)"

            # -------------- save INIT_DICT (always saved)
            hf.append("init_dict",
                      pd.DataFrame(pd.Series(fit._get_init_dict()),
                                   dtype=str,
                                   columns=[ID]).T.astype("category"),
                      format="t", data_columns=True)

            # -------------- save READER_ARG   (always saved)
            if hasattr(fit, "reader_arg"):
                df = pd.DataFrame(fit.reader_arg, [ID])
                hf.append("reader_arg", df)

            # -------------- save DATASET
            if save_data:
                df = getattr(fit, "dataset", None)
                if df is not None:
                    df = pd.concat([df], keys=[ID])
                    hf.append("dataset", df, format="t", data_columns=True)
            elif save_results:
                # store only data that is required to re-create the results
                dynkeys = list(key + "_dyn" for key, val in fit.defdict.items()
                               if val[0] and val[2] == "manual")
                df = getattr(fit, "dataset", None)[dynkeys]
                df = pd.concat([df], keys=[ID])
                print("saving", dynkeys, df)

                hf.append("dataset", df, format="t", data_columns=True)

            # -------------- save AUX_DATA
            if save_auxdata:
                df = getattr(fit, "aux_data", None)
                if df is not None:
                    df = pd.concat([df], keys=[ID])
                    hf.append("aux_data", df, format="t", data_columns=True)

            # -------------- save RES_DICT
            if save_results:
                if fit.res_dict is not None:
                    dfs = []
                    for key, val in fit.res_dict.items():
                        idx = pd.MultiIndex.from_product([[ID],range(len(val))])

                        # TODO implement this after the following pandas-bug is fixed:
                        # https://github.com/pandas-dev/pandas/issues/42070
                        #dfs.append(pd.DataFrame({key:pd.arrays.SparseArray(val)}, idx))

                        dfs.append(pd.DataFrame({key:val}, idx))

                    df = pd.concat(dfs, axis=1)

                    hf.append("res_dict", df , format="t", data_columns=True)


    @staticmethod
    def load_fit(path, ID):
        datakeys = ["dataset", "aux_data"]

        with pd.HDFStore(path, 'r') as f:
            attrs = {key: literal_eval(val) for key, val in
                     f['init_dict'].loc[ID].items()}
            fit = Fits(**attrs)

            for key in datakeys:
                if key in f:
                    if ID in f[key].index:
                        setattr(fit, key, f[key].loc[ID])

            if "res_dict" in f:
                if ID in f["res_dict"].index:

                    fit.res_dict = {key:val.dropna().to_list()
                                    for key, val in f["res_dict"].loc[ID].items()}

            if "reader_arg" in f:
                if ID in f["reader_arg"].index:
                    fit.reader_arg = f["reader_arg"].loc[ID].to_dict()

        return fit


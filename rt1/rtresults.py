# -*- coding: utf-8 -*-

from itertools import islice
from collections import defaultdict
import weakref
import contextlib
import os
from pathlib import Path
from ast import literal_eval
import gc

import pandas as pd
from numpy.random import choice

from .rtparse import RT1_configparser
from .rtfits import load, Fits, MultiFits
from . import log


try:
    import xarray as xar
except ModuleNotFoundError:
    log.info(
        "xarray could not be imported, "
        + "NetCDF-features of RT1_results will not work!"
    )


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

    def __iter__(self):
        return (getattr(self, i) for i in self._paths)

    def __getitem__(self, key):
        assert key in self._paths, (
            f"'{key}' not found... use one of {list(self._paths)}")

        return getattr(self, key)

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
            self._hdf_paths = self._get_results(".h5")


        def __iter__(self):
            return self.dump_files

        @property
        def fit_db(self):
            assert "fit_db" in self._hdf_paths, '"fit_db.h5" not found in results-folder'

            return self.load_hdf("fit_db")

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

        def load_hdf(self, result_name, **kwargs):
            """
            open a HDF5 file stored in the "results"-folder with
            the HDFaccessor class (the file is opened in read-only mode!)

            Parameters
            ----------
            result_name : str, optional
                The name of the HDF-file (without a .h5 extension).
                If None, and only 1 file is available, the available file
                will be laoded. The default is None.
            **kwargs :
                kwargs passed to the initialization of the HDFaccessor class

            Returns
            -------
            HDFaccessor : the HDFaccessor instance for the given file
            """

            results = self._get_results(".h5")
            assert len(results) > 0, "no HDF file in the results folder!"

            assert len(results) == 1 or result_name in results, (
                "there is more than 1 HDF file... "
                + 'provide "result_name":\n    - '
                + "\n    - ".join(results.keys())
            )

            if result_name is None:
                result_name = list(results.keys())[0]

            return HDFaccessor(results[result_name], **kwargs)

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
                    filepath = Path(next(islice(self.dump_files, ID, None)))
                elif isinstance(ID, str):
                    if not ID.endswith(".dump"):
                        ID += ".dump"
                    filepath = self._dump_path / (ID)

            fit = load(filepath)

            if return_ID is True:
                return (fit, filepath.stem)
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
        def dump_fits(self):
            """
            a generator returning the fit-objects to the available dump-files

            NOTICE: only files that do NOT contain "error" in the filename and
            whose file-ending is ".dump" are returned!
            """
            for entry in os.scandir(self._dump_path):
                if (entry.name.endswith('.dump')
                    and entry.is_file()
                    and 'error' not in entry.name):

                    yield self.load_fit(entry.name)

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


class HDFaccessor(object):
    def __init__(self, path, **kwargs):
        """
        accessor class to read HDF5 files stored during RT1 processing

        Parameters
        ----------
        path : str
            the path to the HDF file.
        **kwargs :
            kwargs passed to pd.HDFstore().

        Returns
        -------
        None.

        """
        self.store = pd.HDFStore(path=path, mode="r", **kwargs)

        self.datasets = _data_container()

        if "reader_arg" in self.store:
            self.IDs = self.store.select_column("reader_arg", "index")
        else:
            self.IDs = None

        cfgkeys = defaultdict(list)
        for key in self.store.keys():
            if key.endswith("/meta"):
                continue

            keysplit = key.lstrip("/").split("/")
            if len(keysplit) == 2:
                cfgkeys[keysplit[0]].append(keysplit[1])
            elif len(keysplit) == 1:
                setattr(self.datasets, keysplit[0], _subselector(self, key))
            else:
                setattr(self.datasets, key, _subselector(self, key))

        for cfg, keys in cfgkeys.items():
            setattr(self.datasets, cfg, _cfgselector(self, cfg, keys))

        # make load_fit function public if "init_dict" is found in the store
        if "init_dict" in self.store:
            self.load_fit = self._load_fit

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.store.close()
        gc.collect()

    def __del__(self):
        self.store.close()
        gc.collect()

    def close(self):
        self.store.close()
        gc.collect()

    def _load_fit(self, ID):
        if isinstance(ID, int):
            ID = self.IDs[ID]
        try:
            init_dict = self.datasets.init_dict.get_id(ID)
        except Exception:
            print("'init_dict' not present... fit can not be loaded")
            return

        try:
            dataset = self.datasets.dataset.get_id(ID)
        except Exception:
            dataset = None
            pass

        try:
            aux_data = self.datasets.aux_data.get_id(ID)
        except Exception:
            aux_data = None
            pass

        try:
            reader_arg = self.datasets.reader_arg.get_id(ID)
            reader_arg = reader_arg.loc[ID].to_dict()
            if "ID" not in reader_arg:
                reader_arg["ID"] = ID
        except Exception:
            reader_arg = None
            pass

        try:
            res_dict = self.datasets.res_dict.get_id(ID)
        except Exception:
            res_dict = None
            pass


        if "cfg" in init_dict.index.names:
            mf = MultiFits()
            for cfg, cfg_attrs in init_dict.loc[ID].iterrows():

                attrs = {key: literal_eval(val) for key, val in
                         cfg_attrs.items()}

                fit = Fits(**attrs)
                fit.ID = ID

                if res_dict is not None:
                    # use dropna(how="all", axis=1) to make sure that parameters that
                    # are fitted in one config but not in another are not added as
                    # empty lists!
                    fit.res_dict = {key:val.dropna().to_list()
                                    for key, val in
                                    res_dict.loc[ID].loc[cfg].dropna(how="all",
                                                                     axis=1).items()}

                mf.add_config(cfg, fit)

            # attach shared properties
            if dataset is not None:
                mf.set_dataset(dataset.loc[ID])
            if aux_data is not None:
                mf.set_aux_data(aux_data.loc[ID])
            if reader_arg is not None:
                mf.set_reader_arg(reader_arg)

            mf.set_ID(ID)
            return mf

        else:
            attrs = {key: literal_eval(val) for key, val in
                     init_dict.loc[ID].items()}
            fit = Fits(**attrs)
            fit.ID = ID

            if dataset is not None:
                fit.dataset = dataset.loc[ID]

            if aux_data is not None:
                fit.aux_data = aux_data.loc[ID]

            if reader_arg is not None:
                fit.reader_arg = reader_arg

            if res_dict is not None:
                fit.res_dict = {key:val.dropna().to_list()
                                for key, val in res_dict.loc[ID].items()}
            return fit


    @staticmethod
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _get_vals(self, key, chunksize, start=None, **kwargs):
        n = self.store.get_node(key)

        # TODO
        if "ID" in n.table.colpathnames:
            idxname = "ID"
        else:
            idxname = "index"

        assert self.IDs is not None, "no IDs found in HDF-store"
        if start:
            assert start < len(self.IDs), (f"start={start} is bigger than the number" +
                                           f" of IDs ({len(self.IDs)})")
        use_ids = self._chunks(self.IDs[slice(start, None)], chunksize)
        for i, ids in enumerate(use_ids):
            indexes = self.store.select_as_coordinates(key=key,
                                                       where=f"{idxname} in ids")
            data = self.store.get_storer(key).read(where=indexes,
                                                   **kwargs)
            yield data

    def _get_vals_with_IDs(self, key, use_ids, chunksize=1 , **kwargs):
        n = self.store.get_node(key)
        # TODO
        if "ID" in n.table.colpathnames:
            idxname = "ID"
        else:
            idxname = "index"

        assert self.IDs is not None, "no IDs found in HDF-store"
        use_ids = self._chunks(use_ids, chunksize)

        for i, ids in enumerate(use_ids):
            indexes = self.store.select_as_coordinates(key=key,
                                                  where=f"{idxname} in ids")
            data = self.store.get_storer(key).read(where=indexes,
                                                   **kwargs)
            yield data

    def _get_nids(self, key, nids, start=None,
                 read_chunksize=None, print_progress=False, **kwargs):
        return next(self._get_nids_iter(key=key, nids=nids, start=start,
                                       read_chunksize=read_chunksize,
                                       print_progress=print_progress,
                                       **kwargs))

    def _get_id(self, key, n, **kwargs):
        if isinstance(n, str):
            ret = next(self._get_nids_iter(key=key, nids=[n], **kwargs))

        elif isinstance(n, int):
            ret = next(self._get_nids_iter(key=key, nids=1, start=n, **kwargs))

        return ret

    def _get_nids_iter(self, key, nids, start=None,
                       read_chunksize=None, print_progress=False, **kwargs):
        """
        return an iterator that iteratively yields "nids" IDs from the given key
        until all data has been returned

        Parameters
        ----------
        key : str
            the key to use.
        nids : int or iterable
            if int: the number of IDs to return.
            if iterable: a list of IDs to use
        start : int, optional
            the index-number to start from.

            e.g. equivalent to `IDs[start : start + nids]`

            The default is None.
        read_chunksize : int, optional
            the chunksize used for retrieving the data. The default is None.
        print_progress : bool, optional
            indicator if a progress-message should be printed.
            The default is False.
        **kwargs :
            kwargs passed to `storer.read()`

        Yields
        ------
        data : pandas.DataFrame
            the retrieved DataFrame.

        """
        assert key in self.store, f"invalid key provided, use one of {self.store.keys()}"

        if read_chunksize is None:
            s = self.store.get_storer(key)

            if isinstance(nids, int):
                if s.is_multi_index:
                    read_chunksize = min(10, nids)
                else:
                    read_chunksize = nids
            elif isinstance(nids, str):
                read_chunksize = 1
            else:
                read_chunksize = len(nids)

        if isinstance(nids, int):
            data_gen = self._get_vals(key=key, chunksize=read_chunksize,
                                      start=start, **kwargs)
        elif isinstance(nids, str):
            nids = 1
            data_gen = self._get_vals_with_IDs(key=key, chunksize=read_chunksize,
                                               use_ids=[nids], **kwargs)
        else:
            use_ids = nids
            nids = len(use_ids)
            data_gen = self._get_vals_with_IDs(key=key, chunksize=read_chunksize,
                                               use_ids=use_ids, **kwargs)

        ret = []
        for i, d in enumerate(data_gen):
            ret.append(d)

            if (i+1)%(nids//read_chunksize)==0:
                if nids%read_chunksize != 0:
                    data = next(data_gen)
                    ret.append(iloc_level(data, slice(nids%10)))

                data = pd.concat(ret)
                yield data

                if nids%read_chunksize != 0:
                    ret = [iloc_level(data, slice(nids%10, None))]
                else:
                    ret = []

            if print_progress:
                print("reading", (i + 1)*read_chunksize)
        else:
            print("asdf")
            data = pd.concat(ret)
            yield data

class _data_container(object):
    def __init__(self):
        pass

class _cfgselector(object):
    def __init__(self, parent, cfg, keys):
        # use weak references to ensure proper closing of the file-handlers
        # if parent-class is deleted
        self._parent = weakref.ref(parent)
        self._cfg = cfg
        self._keys = keys

        for key in self._keys:
            setattr(self, key, _subselector(parent, f"{cfg}/{key}"))

class _subselector(object):
    def __init__(self, parent, key):
        # use weak references to ensure proper closing of the file-handlers
        # if parent-class is deleted
        self._parent = weakref.ref(parent)
        self._key = key

    def select(self, **kwargs):
        ret = self._parent().store.select(self._key, **kwargs)
        return ret

    def get_nids_iter(self, nids, start=None,
                      read_chunksize=None, print_progress=False, **kwargs):
        return self._parent()._get_nids_iter(key=self._key,
                                          nids=nids,
                                          start=start,
                                          read_chunksize=read_chunksize,
                                          print_progress=print_progress,
                                          **kwargs)

    def get_nids(self, nids, start=None,
                      read_chunksize=None, print_progress=False, **kwargs):
        return self._parent()._get_nids(key=self._key,
                                     nids=nids,
                                     start=start,
                                     read_chunksize=read_chunksize,
                                     print_progress=print_progress,
                                     **kwargs)

    def get_id(self, n, **kwargs):
        return self._parent()._get_id(key=self._key, n=n, **kwargs)

def _sanitize_index(df, idx):
    # convert inputs to desired indexers
    # (so that we can use integers and None directly)
    try:
        idx = list(idx)
    except TypeError:
        idx = [idx]

    for level, i in enumerate(idx):
        if i is None:
            yield df.index.levels[level][slice(None)]
        elif isinstance(i, int):
            yield df.index.levels[level][slice(i, i + 1)]
        else:
            yield df.index.levels[level][i]

def iloc_level(df, idx):
    """
    iloc a MultiIndexed pandas.DataFrame
    Parameters
    ----------
    df : pandas.DataFrame
        the MultiIndexed dataframe to use.
    idx : iterable
        the indexers to use for the MultiIndex levels.
        can be (None, int, list or slice)

        >>> idx = [level_1 indexer, level_2 indexer, ...]

        if None: all index-values will be used
        if int: the n^th index-value is selected

    Returns
    -------
    pandas.DataFrame
    """
    return df.iloc[df.index.get_locs(list(_sanitize_index(df, idx)))]

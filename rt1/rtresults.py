# -*- coding: utf-8 -*-

from itertools import islice
import os
from pathlib import Path
from numpy.random import choice

from .rtparse import RT1_configparser
from .rtfits import load
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

        def __iter__(self):
            return self.dump_files

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

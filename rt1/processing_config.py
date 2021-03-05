"""convenient default functions for use with rt1.rtfits.processfunc()"""

import sys
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from .rtfits import load
from . import log

try:
    import xarray as xar
except ModuleNotFoundError:
    log.debug("xarray could not be imported, " + "postprocess_xarray will not work!")


def defdict_parser(defdict):
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
                if val[1] == "manual":
                    parameter_specs["fitted_dynamic_manual"].append(key)
                elif val[1] == "index":
                    parameter_specs["fitted_dynamic_index"].append(key)
                elif isinstance(val[1], str):
                    parameter_specs["fitted_dynamic_datetime"].append(key)
                elif isinstance(val[1], int):
                    parameter_specs["fitted_dynamic_integer"].append(key)
        else:
            if val[1] == "auxiliary":
                parameter_specs["auxiliary"].append(key)
            else:
                parameter_specs["constant"].append(key)

    return parameter_specs


def postprocess_xarray(
    fit,
    saveparams=None,
    xindex=("x", -9999),
    yindex=None,
    staticlayers=None,
    auxdata=None,
):
    """
    the identification of parameters is as follows:

        1) 'sig' (conv. to dB) and 'inc' (conv. to degrees) from dataset
        2) any parameter present in defdict is handled accordingly
        3) auxdata (a pandas-dataframe) is appended

        4) static layers are generated and added according to the provided dict


    Parameters
    ----------
    fit : rt1.rtfits.Fits object
        the fit-object to use
    saveparams : list, optional
        a list of strings that correspond to parameter-names that should
        be included. The default is None.
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
        a dict with parameter-names and values that will be adde das
        static layers. The default is None.
    auxdata : pandas.DataFrame, optional
        a pandas DataFrame that will be concatenated to the DataFrame obtained
        from combining all 'saveparams'. The default is None.

    Returns
    -------
    dfxar : xarray.Dataset
        a xarray-dataset with all layers defined according to the specs.

    """

    if saveparams is None:
        saveparams = []

    if staticlayers is None:
        staticlayers = dict()

    defs = defdict_parser(fit.defdict)

    usedfs = []
    for key in saveparams:

        if key == "sig":
            if fit.dB is False:
                usedfs.append(10.0 * np.log10(fit.dataset.sig))
            else:
                usedfs.append(fit.dataset.sig)
        elif key == "inc":
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
            usedfs.append(fit.dataset[key])
        else:
            log.warning(
                f"the parameter {key} could not be processed"
                + "during xarray postprocessing"
            )

    if auxdata is not None:
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


class rt1_processing_config(object):
    """
    a class that provides convenient default functions that can be used
    with `rt1.rtprocess.RTprocess`.

    Parameters
    ----------
    save_path : pathlib.Path, optional
        the parent-path where the files should be stored.
        The default is None.
    dumpfolder : str, optional
        the sub-folder in which the dump-files will be stored.
        The default is None.
    error_dumpfolder : srt, optional
        the sub-folder in which the error-dump-files will be stored.
        if None and dumpfolder is provided, dumpfolder will be used!
        The default is None.
    finalout_name : srt, optional
        the name of the hdf-file generated by finaloutput().
        The default is None.

    Examples
    --------
    >>> from rt1.processing_config import rt1_processing_config
    ...
    ... class run_configuration(rt1_processing_config):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...
    ...     # customize the naming of the output-files
    ...     def get_names_ids(self, reader_arg):
    ...         # define the naming-convention of the files etc.
    ...         ...
    ...         names_ids = dict(filename='site_1.dump', feature_id = '1')
    ...         return names_ids
    ...
    ...     # customize the preprocess function
    ...     def preprocess(self, **kwargs):
    ...         # run code prior to the init. of a multiprocessing.Pool
    ...         ...
    ...         return dict(reader_args=[dict(...), dict(...)],
    ...                     pool_kwargs=dict(...),
    ...                     ...)
    ...
    ...     # add a reader-function
    ...     def reader(self, **reader_arg):
    ...         # read the data for each fit
    ...         ...
    ...         return df, aux_data
    ...
    ...     # customize the postprocess function
    ...     def postprocess(self, fit, reader_arg):
    ...         # do something with each fit and return the desired output
    ...         return ...
    ...
    ...     # customize the finaloutput function
    ...     def finaloutput(self, res):
    ...         # do something with res
    ...         # (res is a list of outputs from the postprocess() function)
    ...         ...
    ...
    ...     # customize the function that is used to catch exceptions
    ...     def exceptfunc(self, ex, reader_arg):
    ...         # do something with the catched exception ex.args:
    ...         if 'lets catch this exception' in ex:
    ...             ...

    """

    def __init__(self, **kwargs):

        if "save_path" in kwargs and "dumpfolder" in kwargs:
            parentpath = Path(kwargs["save_path"]) / kwargs["dumpfolder"]

            self.rt1_procsesing_dumppath = parentpath / "dumps"
            self.rt1_procsesing_respath = parentpath / "results"
            self.rt1_procsesing_cfgpath = parentpath / "cfg"
        else:
            self.rt1_procsesing_dumppath = None
            self.rt1_procsesing_respath = None
            self.rt1_procsesing_cfgpath = None

        # append all arguments passed as kwargs to the class
        # NOTICE: ALL definitions in the 'PROCESS_SPECS' section of the
        #         config-file will be passed as kwargs to the initialization
        #          of this class!)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_names_ids(self, reader_arg):
        """
        A function that returns the file-name based on the passed reader_args

        - the filenames are generated from the reader-argument `'gpi'`

        Parameters
        ----------
        reader_arg : dict
            the arguments passed to the reader function.

        Returns
        -------
        filename : str
            the file-name that will be used to save the fit dump-files in case
            the processing was successful
        error_filename : str
            the file-name that will be used to save the error files in case an
            error occured
        """

        # the ID used for indexing the processed sites
        feature_id = reader_arg["gpi"]

        # the filename of the dump-file
        filename = f"{feature_id}.dump"

        # the filename of the error-dumpfile
        error_filename = f"{feature_id}_error.txt"

        return dict(
            feature_id=feature_id,
            filename=filename,
            error_filename=error_filename,
        )

    def check_dump_exists(self, reader_arg):
        """
        check if a dump of the fit already exists
        (used to determine if the fit has already been evaluated to avoid
         performing the same fit twice)

        Parameters
        ----------
        reader_arg : dict
            the reader-arg dict.

        Raises
        ------
        rt1_file_already_exists
            If a dump-file with the specified filename already exists
            at "save_path / dumpfolder / dumps /"
        """

        # check if the file already exists, and if yes, raise a skip-error
        if self.rt1_procsesing_dumppath is not None:
            names_ids = self.get_names_ids(reader_arg)
            if (self.rt1_procsesing_dumppath / names_ids["filename"]).exists():
                raise Exception("rt1_file_already_exists")

    def dump_fit_to_file(self, fit, reader_arg, mini=True):
        """
        pickle to Fits-object to  "save_path / dumpfolder / filename.dump"

        Parameters
        ----------
        fit : rt1.rtfits.Fits
            the rtfits.Fits object.
        reader_arg : dict
            the reader-arg dict.
        mini : bool, optional
            indicator if a mini-dump should be performed or not.
            (see rt1.rtfits.Fits.dump() for details)
            The default is True.
        """
        if self.rt1_procsesing_dumppath is not None:
            names_ids = self.get_names_ids(reader_arg)

            if not (self.rt1_procsesing_dumppath / names_ids["filename"]).exists():
                fit.dump(
                    self.rt1_procsesing_dumppath / names_ids["filename"],
                    mini=mini,
                )

    def preprocess(self, **kwargs):
        """a function that is called PRIOR to processing"""
        return

    def reader(self, reader_arg):
        """a function that is called for each site to obtain the dataset"""
        # get the incidence-angles of the data
        inc = [0.1, 0.2, 0.3, 0.4, 0.5]
        # get the sig0-values of the data
        sig = [-10, -11.0, -11.45, -13, -15]
        # get the index-values of the data
        index = pd.date_range("1.1.2020", "1.5.2020", freq="D")

        data = pd.DataFrame(dict(inc=inc, sig=sig), index=index)

        aux_data = pd.DataFrame(dict(something=[1, 2, 3, 4, 5]))

        return data, aux_data

    def postprocess(self, fit, reader_arg):
        """
        A function that is called AFTER processing of each site:

        - a pandas.DataFrame with the obtained parameters is returned.
          The columns are multiindexes corresponding to::

              columns = [feature_id,
                         [param_1, param_2, ...]]

        - in case "save_path" is provided:
            - a dump of the fit object will be stored in the folder
            - if the file already exists, a 'rt1_file_already_exists' error
              will be raised


        Parameters
        ----------
        fit: rt1.rtfits.Fits object
            The fits objec.
        reader_arg: dict
            the arguments passed to the reader function.

        Returns
        -------
        df: pandas.DataFrame
            a pandas dataframe containing the fitted parameterss.

        """

        # get filenames
        names_ids = self.get_names_ids(reader_arg)
        # parse defdict to find static and dynamic parameter names
        params = defdict_parser(fit.defdict)

        # make a dump of the fit
        self.dump_fit_to_file(fit, reader_arg)

        # add all constant (fitted) parameters as static layers
        staticlayers = dict()
        for key in params["fitted_const"]:
            staticlayers[key] = fit.res_dict[key][0]

        ret = postprocess_xarray(
            fit=fit,
            saveparams=["inc", "sig", *params["fitted_dynamic"]],
            xindex=("ID", names_ids["feature_id"]),
            staticlayers=staticlayers,
        )

        return ret

    def finaloutput(self, res, format="table"):
        """
        A function that is called after ALL sites are processed:

        First, the obtained parameter-dataframes returned by the
        `postprocess()` function are concatenated, then:

            - if `"save_path"` is defined, the resulting dataframe will be
              saved (or appended) to a hdf-store.
              the used key will be either the value of `"hdf_key"` or
              `"dumpfolder`" or in case both are `None`, the key `"result`"
              will be used
            - if `"save_path"` is None, the DataFrame of the concatenated
              results will be returned

        Notice:
        Since hdf is a row-based format, the HDF-file will contain a transposed
        version of the multiindexed "res"-DataFrame.
        Furthermore all values will be converted to numerical values
        by using `pd.to_numeric()`   -> timestaps need to be re-converted using
        `pd.to_datetime()`!


        Parameters
        ----------
        res: list
            A list of return-values from the "postprocess()" function.
        save_path: str, optional
            The path where the finalout-file will be stored.
            The default is None.
        format: str
            the format used when exporting the hdf-file.

            Notice: if there are more than 2000 entries, the 'table' format
            will not work and the 'fixed' format must be used!
            The default is 'fixed'
        Returns
        -------
        res: pandas.DataFrame
             the concatenated results (ONLY if `"save_path"` is `None`)
        """

        # concatenate the results
        resxar = xar.combine_nested([i for i in res if i is not None], concat_dim="ID")

        if self.rt1_procsesing_respath is None or self.finalout_name is None:
            log.info(
                "both save_path and finalout_name must be specified... "
                + "otherwise the final results can NOT be saved!"
            )
            return resxar
        else:
            # export netcdf file
            resxar.to_netcdf(self.rt1_procsesing_respath / self.finalout_name)

    def exceptfunc(self, ex, reader_arg):
        """
        a error-catch function that handles the following errors:

        - 'rt1_skip'
            exceptions are ignored and the next site is processed WITHOUT
            writing the exception to a file
        - 'rt1_data_error'
            exceptions are ignored and the next site is processed
        - 'rt1_file_already_exists'
            the already existing dump-file is loaded, the postprocess()
            function is applied and the result is returned
        - for any other exceptions:
            if `save_path` and `dumpfolder`or `error_dumpfolder` are specified
            a dump of the exception-message is saved and the exception
            is ignored. otherwise the exception will be raised.

        Parameters
        ----------
        ex : Exception
            the catched exception.
        reader_arg : dict
            the arguments passed to the reader function.

        """

        log.debug(
            f"catched an {type(ex).__name__} {ex.args} for the "
            + f"following reader_args: \n{reader_arg}"
        )

        names_ids = self.get_names_ids(reader_arg)
        raise_exception = True

        if "rt1_skip" in ex.args:
            log.debug("SKIPPED the following error:")
            log.debug(traceback.format_exc())
            # ignore skip exceptions
            raise_exception = False

        elif "rt1_file_already_exists" in ex.args:
            # if the fit-dump file already exists, try loading the existing
            # file and apply post-processing (e.g. avoid re-processing results)
            raise_exception = False

            log.debug(
                f"the file '{names_ids['filename']}' already exists... "
                + "I'm 'using the existing one!"
            )

            try:
                fit = load(self.rt1_procsesing_dumppath / names_ids["filename"])
                return self.postprocess(fit, reader_arg)
            except Exception:
                log.debug(
                    "the has been a problem while loading the "
                    + f"already processed file '{names_ids['filename']}'"
                )
                pass

        elif "rt1_data_error" in ex.args:
            # raised if there was a problem with the data, ignore and continue
            raise_exception = False

        # in case `save_path` is specified, write ALL exceptions to a file
        # and continue processing WITHOUT raising the exception.
        # if `save_path`is NOT specified, raise ONLY exceptions that
        # have not been explicitly catched
        if self.rt1_procsesing_dumppath is None and raise_exception is True:
            log.info(
                "`save_path` must be specified otherwise exceptions "
                + "that are not explicitly catched by `exceptfunc()` "
                + " will be raised!"
            )
            raise ex
        else:
            if "error_filename" in names_ids:
                error_filename = names_ids["error_filename"]
            else:
                error_filename = names_ids["filename"].split(".")[0] + "_error.txt"

            # dump the encountered exception to a file
            with open(self.rt1_procsesing_dumppath / error_filename, "w") as file:
                file.write(traceback.format_exc())

        # flush stdout to see output of child-processes
        sys.stdout.flush()

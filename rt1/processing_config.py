"""convenient default functions for use with rt1.rtfits.processfunc()"""

import sys
import traceback
from pathlib import Path
from .rtfits import load
from . import log


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
        #         of this class!)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_names_ids(self, reader_arg):
        """
        A function that returns the file-name based on the passed reader_args

        - by default, the filenames are generated from the reader-argument "ID"

        Parameters
        ----------
        reader_arg : dict
            the arguments passed to the reader function.

        Returns
        -------
        feature_id : str
            the ID that will be assigned to the fits-object (e.g. `fit.ID`)
        filename : str
            the file-name that will be used to save the pickle dump-files
        """

        # the ID used for indexing the processed sites
        feature_id = str(reader_arg["ID"])

        # the filename of the dump-file
        filename = f"{feature_id}.dump"

        return dict(
            feature_id=feature_id,
            filename=filename,
        )

    def preprocess(self, **kwargs):
        """
        a (optional) function that is called PRIOR to processing

        Parameters
        ----------
        kwargs :
            kwargs obtained from

            >>> RTprocess(...).run_processing(preprocess_kwargs=dict(...))

        Returns:
        --------
        dict
            a dict with the following keys defined:

                - "reader_args" : a list of dicts that will be used as input
                  for the call to the reader function (e.g. `reader_arg`)
                - "pool_kwargs" : a dict of kwargs passed to the initialization
                  of the multiprocessing.Pool used for processing
                  (useful for providing initializers etc.)
        --------
        """

        return

    def reader(self, reader_arg):
        """
        a function that is called for each site to obtain the dataset

        it must return a pandas.DataFrame with the following keys defined:
            - "inc" : the incidence-angle in radians
            - "sig" : the backscattering coefficient values

        any additional return-values will be appended to the fit as `fit.aux_data`

        >>> def reader(self, reader_arg):
        >>>    data = pd.DataFrame(inc=[...]
        >>>                        sig=[...],
        >>>                        index = [a datetime-index])
        >>>
        >>>    aux_data = "any aux-data that should be appended"
        >>>
        >>>    return data, aux_data

        Parameters
        ----------
        reader_arg : dict
            the arguments passed to the reader function.

        Returns
        -------
        return_data : pandas.DataFrame
            a pandas.DataFrame that will be used as "dataset" for the fit
            (e.g. `fit.dataset`)
        *aux_data :
            any additional return-arguments will be attached to the fits-object
            as `fit.aux_data`
        """
        assert False, "you must define a proper reader-function first!"

    def exceptfunc(self, ex, reader_arg):
        """
        a error-catch function that handles the following errors:

        - 'rt1_skip'
            exceptions are ignored and the next site is processed WITHOUT
            writing the exception to a file
        - 'rt1_data_error'
            exceptions are ignored and the next site is processed
        - 'rt1_file_already_exists'
            the already existing dump-file is loaded, and the dict defining
            the Fits-object is returned
        - for any other exceptions:
            if `save_path` and `dumpfolder` are specified
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
            # if a fit-dump file already exists, try loading the existing
            # file and apply post-processing (e.g. avoid re-processing results)
            raise_exception = False

            log.debug(
                f"the file '{names_ids['filename']}' already exists... "
                + "I'm 'using the existing one!"
            )

            try:
                fit = load(self.rt1_procsesing_dumppath / names_ids["filename"])
                return fit._get_fit_to_hdf_dict()
            except Exception:
                log.error(
                    "there has been a problem while loading the "
                    + f"already processed file '{names_ids['filename']}'"
                )
                log.error(traceback.format_exc())

        elif "rt1_data_error" in ex.args:
            # raised if there was a problem with the data, ignore and continue
            raise_exception = False

            log.error(
                f"there was a DATA-problem for '{names_ids['filename']}'"
                + "\n"
                + str([i for i in ex.args if i != "rt1_data_error"])
            )
        else:
            log.error(
                f"something went wrong for: {reader_arg}\n" + traceback.format_exc()
            )

        # in case `save_path` is specified, write ALL exceptions to a file
        # and continue processing WITHOUT raising the exception.
        # if `save_path`is NOT specified, raise ONLY exceptions that
        # have not been explicitly catched
        if self.rt1_procsesing_dumppath is None and raise_exception is True:
            log.debug(
                "`save_path` must be specified otherwise exceptions "
                + "that are not explicitly catched by `exceptfunc()` "
                + " will be raised!"
            )
            raise ex

        # flush stdout to see output of child-processes
        sys.stdout.flush()

    def mask_existing_HDF_IDs(self, arg_list, key="reader_arg", get_ID=None):
        """
        check if the IDs provided in "arg_list" are already present in the "fit_db.h5"
        container and return a list that contains only the missing entries of "arg_list".

        Parameters
        ----------
        key : str, optional
            the key to use in the HDF-container. The default is "reader_arg".
        get_ID : callable, optional
            a custom callable that extracts the ID from the provided
            IDs list.
            The default is None in which case the following function is used:

                >>> # extract filename from given path
                >>> def get_ID(i):
                >>>     ID = str(i["ID"]).split(os.sep)[-1].split(".")[0]
                >>>     if not isidentifier(str(ID)):
                >>>         return f"RT1_{ID}"
                >>>     else:
                >>>         return str(ID)

        Returns
        -------
        new_IDs : set
            a set containing the unique elements of "inp" whose IDs are not
            already present in the output HDF container.

        """
        import os
        from .general_functions import isidentifier, find_missing
        from .rtresults import HDFaccessor
        import pandas as pd

        dst_path = self.rt1_procsesing_respath / "fit_db.h5"

        if Path(dst_path).exists():
            log.progress("Checking for already existing IDs in the `fit_db.h5`...")
            if get_ID is None:

                def get_ID(i):
                    ID = str(i["ID"]).split(os.sep)[-1].split(".")[0]
                    if not isidentifier(str(ID)):
                        return f"RT1_{ID}"
                    else:
                        return str(ID)

            with HDFaccessor(dst_path) as fit_db:
                # get all IDs that are already present in the HDF store
                # found_IDs = store.select_column("reader_arg", "ID").values
                # get a list of integers that have already been assigned to IDs
                # found_ID_nums = store.select_column("reader_arg", "index").values

                found_ID_nums = fit_db.IDs.index
                found_IDs = fit_db.IDs.ID.values

                # a list of bool's that indicate if the ID is already processed
                process_Q = pd.Index(map(get_ID, arg_list)).isin(found_IDs)

                # a counter that yields unique IDs that are not yet assigned
                # in the HDF store
                id_counter = find_missing(found_ID_nums)

                # set the processing-args
                # update ID to be a valid python-identifier
                args_to_process = [
                    {**arg, "_RT1_ID_num": next(id_counter)}
                    for arg, q in zip(arg_list, process_Q)
                    if not q
                ]
        else:
            args_to_process = [
                {**i, "_RT1_ID_num": n} for i, n in zip(arg_list, range(len(arg_list)))
            ]

            log.info("no existing output-HDF file found...")
            log.progress(f"processing all {len(args_to_process)} IDs!")
            return args_to_process

        log.progress(
            f"Found {len(args_to_process)} missing and "
            + f"{len(arg_list) - len(args_to_process)}"
            + " existing IDs!"
        )

        return args_to_process

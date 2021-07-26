"""convenient default functions for use with rt1.rtfits.processfunc()"""

import sys
import pandas as pd
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

        # add the default key that is used to define the ID's from the
        # reader_arg dict.
        if not hasattr(self, "ID_key"):
            self.ID_key = "ID"

    def get_names_ids(self, reader_arg):
        """
        A function that returns the file-name based on the passed reader_args

        - the filenames are generated from the reader-argument `self.ID_key`
          (which is set to "ID" by default)

        Parameters
        ----------
        reader_arg : dict
            the arguments passed to the reader function.

        Returns
        -------
        filename : str
            the file-name that will be used to save the fit dump-files in case
            the processing was successful
        """

        # the ID used for indexing the processed sites
        feature_id = str(reader_arg[self.ID_key])

        # the filename of the dump-file
        filename = f"{feature_id}.dump"

        return dict(
            feature_id=feature_id,
            filename=filename,
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
        pickle Fits-object to  "save_path / dumpfolder / filename.dump"

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

            log.error(f"there was a DATA-problem for '{names_ids['filename']}'")
        else:
            log.error(traceback.format_exc())

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

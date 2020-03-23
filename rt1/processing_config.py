'''
a class providing convenient default functions for processing
'''

import sys
import pandas as pd
from datetime import datetime
import traceback
import cloudpickle
from pathlib import Path

class rt1_processing_config(object):
    def __init__(self, save_path=None, dumpfolder=None,
                 error_dumpfolder=None, finalout_name=None):

        self.save_path = Path(save_path)
        self.dumpfolder = dumpfolder
        self.error_dumpfolder = error_dumpfolder
        self.finalout_name = finalout_name


    def get_names_ids(self, reader_arg):
        '''
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
        '''

        # the ID used for indexing the processed sites
        feature_id = reader_arg['gpi']

        # the filename of the dump-file
        filename = f"{feature_id}.dump"

        # the filename of the error-dumpfile
        error_filename = f"{feature_id}_error.dump"

        return feature_id, filename, error_filename


    def preprocess(self, reader_arg):
        '''
        a function that is called PRIOR to processing that does the following:
            - create the folder-structure if it does not yet exist
            - check if a dumpfile of the site already exists and if yes,
              raise a 'rt1_file_already_exists' error

        Parameters
        ----------
        reader_arg : dict
            the arguments passed to the reader.

        '''
        if self.save_path is not None and self.dumpfolder is not None:
            # generate "save_path" directory if it does not exist
            if not self.save_path.exists():
                print(self.save_path, 'does not exist... creating directory')
                self.save_path.mkdir()
            # generate "dumpfolder" directory if it does not exist
            dumppath = self.save_path / self.dumpfolder
            if not dumppath.exists():
                print(dumppath, 'does not exist... creating directory')
                dumppath.mkdir()

            # obtain the path where the dump-file would be stored
            _, filename, _ = self.get_names_ids(reader_arg)
            dumppath = self.save_path.joinpath(self.dumpfolder, filename)

            # raise a file already exists error
            if dumppath.exists():
                raise Exception('rt1_file_already_exists')


    def postprocess(self, fit, reader_arg):
        '''
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

        '''

        # set filenames
        feature_id, fname, error_fname = self.get_names_ids(reader_arg)

        if self.save_path is not None and self.dumpfolder is not None:
            dumppath = self.save_path.joinpath(self.dumpfolder, fname)

            # if no dump exists, dump it, else load the existing dump
            if not dumppath.exists():
                fit.dump(dumppath, mini=True)
                print(datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                      'finished', feature_id)

        # get resulting parameter DataFrame
        df = fit.res_df
        # add the feature_id as first column-level
        df.columns = pd.MultiIndex.from_product([[feature_id], df.columns])
        df.columns.names = ['feature_id', 'param']

        # flush stdout to see output of child-processes
        sys.stdout.flush()

        return df


    def finaloutput(self, res, hdf_key=None, format='table'):
        '''
        A function that is called after ALL sites are processed:

        - first, the obtained parameter-dataframes returned by the
          `postprocess()` function are concatenated
            - if `"save_path"` is defined, the resulting dataframe will be
              saved (or appended) to a hdf-store.
              the used key will be either the value of `"hdf_key"` or
              `"dumpfolder`" or in case both are `None`, the key `"result`"
              will be used
            - if `"save_path"` is None, the DataFrame of the concatenated
              results will be returned

        Parameters
        ----------
        res: list
            A list of return-values from the "postprocess()" function.
        save_path: str, optional
            The path where the finalout-file will be stored.
            The default is None.
        hdf_key: str, optional
            The key used in the hdf-file. If None, the dumpfolder name will
            be used. If dumpfolder is not provided, 'result' is used.
            The default is None.

        Returns
        -------
        res: pandas.DataFrame
             the concatenated results (ONLY if `"save_path"` is `None`)
        '''

        # concatenate the results
        res = pd.concat([i for i in res if i is not None], axis=1)

        if self.save_path is None or self.finalout_name is None:
            print('both save_path and finalout_name must be specified... ',
                  'results have NOT been saved!')
            return res
        else:

            if hdf_key is None:
                if self.dumpfolder is not None:
                    hdf_key = str(self.dumpfolder)
                else:
                    hdf_key = 'result'

            # create (or append) results to a HDF-store
            res.to_hdf(self.save_path / self.finalout_name,
                       key=hdf_key, format=format, complevel=5)

        # flush stdout to see output of child-processes
        sys.stdout.flush()


    def exceptfunc(self, ex, reader_arg):
        '''
        a error-catch function that handles the following errors:

        - 'rt1_skip'
            - exceptions are ignored and the next site is processed
        - 'rt1_data_error'
            - exceptions are ignored and the next site is processed
        - 'rt1_file_already_exists'
            - the already existing dump-file is loaded, the postprocess()
            function is applied and the result is returned
        - for any other exceptions:
            - if 'save_path' and 'error_dumpfolder' are specified
              a dump of the exception-message is saved and the exception
              is ignored. otherwise the exception will be raised.


        Parameters
        ----------
        ex : Exception
            the catched exception.
        reader_arg : dict
            the arguments passed to the reader function.

        '''

        feature_id, fname, error_fname = self.get_names_ids(reader_arg)

        if self.save_path is None and self.error_dumpfolder is None:
            print('both save_path and error_dumpfolder must be specified ',
                  'otherwise exceptions will be raised')
            raise ex
        else:
            dumppath = self.save_path.joinpath(self.dumpfolder, fname)
            error_dump = self.save_path.joinpath(self.error_dumpfolder,
                                                 error_fname)

            if 'rt1_skip' in ex.args:
                # ignore skip exceptions
                print(datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                      '... skipping', feature_id)

            elif 'rt1_file_already_exists' in ex.args:
                # if the fit-dump file already exists, try loading the existing
                # file and apply post-processing if possible
                print('file already exists')

                try:
                    with open(dumppath, 'rb') as file:
                        fit = cloudpickle.load(file)

                    return self.postprocess(fit, reader_arg)
                except Exception:
                    print(datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                          'the file', dumppath, 'seems to be corrupted')


            elif 'rt1_data_error' in ex.args:
                # if no data is found, ignore and continue
                print(datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                      'there was no data for ', feature_id, '   ...passing on')
            else:
                # dump the encountered exception to a file
                print(datetime.now().strftime('%d-%b-%Y %H:%M:%S'),
                      'there was an exception for ', feature_id)
                with open(error_dump, 'w') as file:
                    file.write(traceback.format_exc())

        # flush stdout to see output of child-processes
        sys.stdout.flush()


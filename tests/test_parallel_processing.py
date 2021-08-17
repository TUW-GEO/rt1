from pathlib import Path
import shutil
import unittest
import unittest.mock as mock
from rt1 import log, start_log_to_file, stop_log_to_file
from rt1.rtprocess import RTprocess
from rt1.rtresults import RTresults
from rt1.rtfits import MultiFits
import warnings
import pytest
import multiprocessing as mp

warnings.simplefilter("ignore")


# use "test_0_---"   "test_1_---"   to ensure test-order

# # set this to False to avoid deleting results & processing-folders!
cleanup_before = False
cleanup_after = False


# define an export_function (used in test_92_export_results)
# (must be defined outside of __main__ to be pickleable!)
def export_sig2(fit):
    return fit.dataset.sig**2


class TestRTfits(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        """ ... called before the testing starts """
        if not cleanup_before:
            return

        folders = ["proc_test", "proc_test2", "proc_multi"]

        for folder in folders:
            p = Path(f"tests/{folder}")
            if p.exists():
                print(f"removing existing folder '{folder}' before starting tests")
                shutil.rmtree(p)

    @classmethod
    def teardown_class(cls):
        """ cleanup after the test finished """
        if not cleanup_after:
            return

        folders = ["proc_test", "proc_test2", "proc_multi"]

        for folder in folders:
            p = Path(f"tests/{folder}")
            if p.exists():
                print(f"removing folder '{folder}' to cleanup after testing")
                shutil.rmtree(p)

    def test_0_parallel_processing(self):
        reader_args = [dict(ID=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / "test_config.ini"

        proc = RTprocess(config_path, autocontinue=True)
        proc.run_processing(ncpu=4, reader_args=reader_args, dump_fit=True)

        # run again to check what happens if files already exist
        proc.run_processing(ncpu=4, reader_args=reader_args)

        # ----------------------------------------- check if files have been copied
        assert Path(
            "tests/proc_test/dump01/cfg"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test/dump01/results"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test/dump01/dumps"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test/dump01/cfg/test_config.ini"
        ).exists(), "copying did not work"
        assert Path(
            "tests/proc_test/dump01/cfg/parallel_processing_config.py"
        ).exists(), "copying did not work"

    def test_1_rtresults(self):

        results = RTresults("tests/proc_test")
        dumpresults = RTresults("tests/proc_test", use_dumps=True)

        assert hasattr(results, "dump01"), "dumpfolder not found by RTresults"

        dumpfiles = [i for i in results.dump01.dump_files]
        print(dumpfiles)

        fit = results.dump01.load_fit()
        _ = results.dump01.load_cfg()

        # with results.dump01.load_nc() as ncfile:
        #     processed_ids = list(ncfile.ID.values)

        # processed_ids.sort()
        # assert processed_ids == [1, 2, 3], "NetCDF export does not include all IDs"

        # # check if NetCDF_variables works as expected
        # results.dump01.NetCDF_variables

        # remove the save_path directory
        # print('deleting save_path directory...')
        # shutil.rmtree(results._parent_path)

        fit_db = results.dump01.fit_db

        assert not any(fit_db.IDs.duplicated()), (
            "there are duplicated IDs in the HDF container!")

        # check if all ID's are present in the HDF-container
        assert all(
            i in fit_db.IDs.ID.values for i in ["RT1_1", "RT1_2", "RT1_3"]
            ), ("HDF-container does not contain all IDs")

        # check if dumped-properties are equal for pickles and HDF-containers
        for fit in dumpresults.dump01.dump_fits:
            fit_hdf = fit_db.load_fit(fit.ID)
            assert fit_hdf.dataset.equals(fit.dataset), (
                "datasets of HDF-container and pickle-dumps are not equal!")
            assert fit_hdf.res_df.equals(fit.res_df), (
                "res_df of HDF-container and pickle-dumps are not equal!")

        # check accessing fits via ID
        id_fit = fit_db.load_fit("RT1_1")
        assert id_fit.ID == "RT1_1", "the ID of the loaded fit is not OK"

        # try accessing the data directly
        alldata = fit_db.datasets.dataset.select()
        assert len(alldata.index.levels[0]) == 3, (
            "accessing full dataset was not OK")

        data0 = fit_db.datasets.dataset.get_id(0)
        assert len(data0.index.levels[0]) == 1, "dataset-selection was not OK"
        assert data0.index.levels[0][0] == 0, "dataset-selection was not OK"

        for i in ["RT1_1", "RT1_2", "RT1_3"]:
            dataID = fit_db.datasets.dataset.get_id(i)
            assert len(dataID.index.levels[0]) == 1, (
                "dataset-selection via ID was not OK")
            assert fit_db.IDs.loc[dataID.index.levels[0][0]].ID == i, (
                "dataset-selection via ID was not OK")

        data2 = fit_db.datasets.dataset.get_nids(2)
        assert len(data2.index.levels[0]) == 2, (
            "multiple dataset-selection was not OK")
        assert all(fit_db.IDs.index[:2].isin(data2.index.levels[0])), (
            "multiple dataset-selection was not OK")

        # check data-generator
        data_iter = fit_db.datasets.dataset.get_nids_iter(2)

        data_iter_1 = next(data_iter)
        assert len(data_iter_1.index.levels[0]) == 2, (
            "dataset generator was not OK")
        assert all(fit_db.IDs.index[:2].isin(data_iter_1.index.levels[0])), (
            "dataset generator was not OK")

        data_iter_2 = next(data_iter)
        assert len(data_iter_2.index.levels[0]) == 1, (
            "dataset generator was not OK")
        assert all(fit_db.IDs.index[2:].isin(data_iter_2.index.levels[0])), (
            "dataset generator was not OK")

        with self.assertRaises(StopIteration):
            _ = next(data_iter)

        # make sure to close the HDF-file so that the folders can be savely deleted
        fit_db.close()

    def test_2_single_core_processing(self):
        reader_args = [dict(ID=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / "test_config.ini"

        # mock inputs as shown here: https://stackoverflow.com/a/37467870/9703451
        with self.assertRaises(SystemExit):
            with mock.patch("builtins.input", side_effect=["N"]):
                proc = RTprocess(config_path, autocontinue=False)
                proc.setup()

        with self.assertRaises(SystemExit):
            with mock.patch("builtins.input", side_effect=["N"]):
                proc = RTprocess(config_path, autocontinue=False)
                proc.setup()

        with mock.patch("builtins.input", side_effect=["REMOVE", "Y"]):
            proc = RTprocess(config_path, autocontinue=False)
            proc.setup()
        assert (
            len(list(Path("tests/proc_test/dump01/dumps").iterdir())) == 0
        ), "user-input REMOVE did not work"

        with mock.patch("builtins.input", side_effect=["REMOVE", "Y"]):
            proc = RTprocess(config_path, autocontinue=False)
            proc.run_processing(ncpu=1, reader_args=reader_args)

        # ----------------------------------------- check if files have been copied
        assert Path(
            "tests/proc_test/dump01/cfg"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test/dump01/results"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test/dump01/dumps"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test/dump01/cfg/test_config.ini"
        ).exists(), "copying did not work"
        assert Path(
            "tests/proc_test/dump01/cfg/parallel_processing_config.py"
        ).exists(), "copying did not work"

        # remove the save_path directory
        # print('deleting save_path directory...')
        # shutil.rmtree(proc.dumppath.parent)

    def test_3_parallel_processing_init_kwargs(self):
        # test overwriting keyword-args from .ini file
        reader_args = [dict(ID=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / "test_config.ini"

        proc = RTprocess(config_path, autocontinue=True)

        proc.override_config(
            PROCESS_SPECS=dict(
                path__save_path="tests/proc_test2",
                dumpfolder="dump02",
            )
        )

        proc.run_processing(ncpu=4, reader_args=reader_args)

        # ----------------------------------------- check if files have been copied
        assert Path(
            "tests/proc_test2/dump02/cfg"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test2/dump02/results"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test2/dump02/dumps"
        ).exists(), "folder-generation did not work"
        assert Path(
            "tests/proc_test2/dump02/cfg/test_config.ini"
        ).exists(), "copying did not work"
        assert Path(
            "tests/proc_test2/dump02/cfg/parallel_processing_config.py"
        ).exists(), "copying did not work"

    def test_5_multiconfig(self):

        config_path = Path(__file__).parent.absolute() / "test_config_multi.ini"
        reader_args = [dict(ID=i) for i in [1, 2, 3, 4]]

        proc = RTprocess(config_path)
        proc.run_processing(ncpu=4, reader_args=reader_args)

        for cfg in ["cfg_0", "cfg_1"]:
            # check if model-definition files are written correctly
            assert Path(
                f"tests/proc_multi/dump01/cfg/model_definition__{cfg}.txt"
            ).exists(), "multiconfig model_definition.txt export did not work"

            # # # check if netcdf have been exported
            # assert Path(
            #     f"tests/proc_multi/dump01/results/results__{cfg}.nc"
            # ).exists(), "multiconfig NetCDF export did not work"

    def test_6_multiconfig_rtresults(self):
        results = RTresults("tests/proc_multi")
        assert hasattr(results, "dump01"), "dumpfolder not found by RTresults"

        _ = [i for i in results.dump01.dump_files]

        fit = results.dump01.load_fit()
        _ = results.dump01.load_cfg()

        fit_db = results.dump01.fit_db

        # check if all ID's are present in the HDF-container
        assert all(
            i in fit_db.IDs.ID.values for i in ["RT1_1", "RT1_2", "RT1_3"]
            ), ("HDF-container does not contain all IDs")

        # check if dumped-properties are equal for pickles and HDF-containers
        for fit in results.dump01.dump_fits:
            fit_hdf = fit_db.load_fit(fit.ID)
            _ = fit_hdf.dataset[list(fit.dataset)]
            assert fit_hdf.dataset.equals(fit.dataset), (
                "datasets of HDF-container and pickle-dumps are not equal!")

            for fit_cfg in fit.configs:
                hdf_res_df = fit_hdf.configs[fit_cfg.config_name].res_df
                # make sure column-order is the same
                hdf_res_df = hdf_res_df[list(fit_cfg.res_df)]

                assert hdf_res_df.equals(fit_cfg.res_df), (
                    "res_df of HDF-container and pickle-dumps are not equal!"
                    )

        # make sure to close the HDF-file so that the folders can be savely deleted
        fit_db.close()

    def test_8_multiconfig_rtresults(self):
        res = RTresults("tests/proc_multi")

        mfit = res.dump01.load_fit()
        assert isinstance(mfit, MultiFits), "the dumped fitobject is not a MultiFits!"

        for cfg in ["cfg_0", "cfg_1"]:
            # check if all configs are added
            assert hasattr(mfit.configs, cfg), "multi-results are not properly added"

    def test_9_multiconfig_props(self):
        # check if properties have been set correctly

        res = RTresults("tests/proc_multi")
        mfit = res.dump01.load_fit()

        fit = mfit.configs.cfg_0
        assert fit.defdict["omega"][0] is False, "multiconfig props not correct"
        assert fit.defdict["omega"][1] == 0.5, "multiconfig props not correct"
        assert fit.int_Q is False, "multiconfig props not correct"

        assert str(fit.V.t) == "t_v", "multiconfig props not correct"
        assert fit.SRF.ncoefs == 10, "multiconfig props not correct"

        assert fit.lsq_kwargs["ftol"] == 0.0001, "multiconfig props not correct"

        # -----------------------------

        fit = mfit.configs.cfg_1
        assert fit.defdict["omega"][0] is True, "multiconfig props not correct"
        assert fit.defdict["omega"][1] == 0.05, "multiconfig props not correct"
        assert fit.defdict["omega"][2] == "2M", "multiconfig props not correct"
        assert fit.int_Q is True, "multiconfig props not correct"

        assert fit.V.t == 0.25, "multiconfig props not correct"
        assert fit.SRF.ncoefs == 5, "multiconfig props not correct"
        assert fit.lsq_kwargs["ftol"] == 0.001, "multiconfig props not correct"

    def test_92_export_results(self):
        for folder in ["proc_test", "proc_test2", "proc_multi"]:
            # select the first subfolder and find the .ini file used
            res = list(RTresults(f"tests/{folder}"))
            assert len(res) == 1, "there's more than one result-folder???"
            res = res[0]
            configpath = res.load_cfg().configpath

            proc = RTprocess(configpath)

            parameters = ["t_s", "tau", "sig2"]
            export_functions = dict(sig2=export_sig2)

            metrics = dict(R=("pearson", "sig", "tot"),
                           RMSD=("rmsd", "sig", "tot")
                           )

            proc.export_data_to_HDF(parameters=parameters,
                                    metrics=metrics,
                                    export_functions=export_functions,
                                    ncpu=1,
                                    finalout_name="export_ncpu1.h5")

            proc.export_data_to_HDF(parameters=parameters,
                                    metrics=metrics,
                                    export_functions=export_functions,
                                    ncpu=3,
                                    finalout_name="export_ncpu3.h5")

            for export_name in ["export_ncpu1", "export_ncpu3"]:

                data = res.load_hdf(export_name)

                # check if all ID's are present in the HDF-container
                assert all(
                    i in data.IDs.ID.values for i in ["RT1_1", "RT1_2", "RT1_3"]
                    ), ("HDF-container does not contain all IDs")

                # some basic checks if Fits and MultiFits are correctly exported

                if folder in ["proc_test", "proc_test2"]:
                    configs = ["default"]  # single-config is called "default"
                else:
                    configs = ["cfg_0", "cfg_1"]

                for cfg in configs:
                    cols = list(getattr(
                        data.datasets, cfg).dynamic.get_id(0).columns)
                    assert cols == ["tau", "sig2"], "dynamic columns are not OK"

                    cols = list(getattr(
                        data.datasets, cfg).static.get_id(0).columns)
                    assert cols == ["t_s"], "static columns are not OK"

                    cols = list(getattr(
                        data.datasets, cfg).metrics.get_id(0).columns)
                    assert cols == ["R", "RMSD"], "metrics columns are not OK"

    # this is needed to disable log-capturing during testing
    @pytest.fixture(autouse=True)
    def caplog(self, caplog):
        self.caplog = caplog

    def test_log_to_file(self):
        logpath = Path("tests/proc_test/testlog.log")
        # temporarily set the log-capture level to 0 (e.g. allow all logs)
        with self.caplog.at_level(0):
            start_log_to_file(logpath, 0)
            log.error(str(log.handlers))
            log.error("error message")
            log.warning("warning message")
            log.debug("debug message")
            log.info("info message")
            log.progress("progress message")

            log.progress("a multiline\nmessage nice!")

        stop_log_to_file()

        assert logpath.exists(), "the logfile does not exist!"

        with open(logpath, "r") as file:
            msgs = [line.split(mp.current_process().name)[-1].strip()
                    for line in file.readlines()]

        expected_msgs = ["ERROR   error message",
                         "WARNING warning message",
                         "DEBUG   debug message",
                         "INFO    info message",
                         "PROG.   progress message",
                         "PROG.   a multiline",
                         "message nice!"]

        # skip the first message since it comes from starting the file-handler
        for i, msg in enumerate(msgs[1:]):
            assert msg == expected_msgs[i], (
                f"log message {i} not OK:\n {msgs}")


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import shutil
import unittest
import unittest.mock as mock
from rt1.rtprocess import RTprocess
from rt1.rtresults import RTresults
from rt1.rtfits import MultiFits
import warnings

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

        folders = ["proc_test", "proc_test2", "proc_multi" ]

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

        folders = ["proc_test", "proc_test2", "proc_multi" ]

        for folder in folders:
            p = Path(f"tests/{folder}")
            if p.exists():
                print(f"removing folder '{folder}' to cleanup after testing")
                shutil.rmtree(p)


    def test_0_parallel_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / "test_config.ini"

        proc = RTprocess(config_path, autocontinue=True)
        proc.run_processing(ncpu=4, reader_args=reader_args)

        #run again to check what happens if files already exist
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
        assert hasattr(results, "dump01"), "dumpfolder not found by RTresults"

        dumpfiles = [i for i in results.dump01.dump_files]
        print(dumpfiles)

        fit = results.dump01.load_fit()
        cfg = results.dump01.load_cfg()

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

        # check if all ID's are present in the HDF-container
        assert all(
            i in fit_db.IDs.values for i in ["RT1_1", "RT1_2", "RT1_3"]
            ), ("HDF-container does not contain all IDs")

        # check if dumped-properties are equal for pickles and HDF-containers
        for fit in results.dump01.dump_fits:
            fit_hdf = fit_db.load_fit(fit.ID)
            assert fit_hdf.dataset.equals(fit.dataset), (
                "datasets of HDF-container and pickle-dumps are not equal!")
            assert fit_hdf.res_df.equals(fit.res_df), (
                "res_df of HDF-container and pickle-dumps are not equal!")

        # make sure to close the HDF-file so that the folders can be savely deleted
        fit_db.close()

    def test_2_single_core_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
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
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
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

    # def test_4_postprocess_and_finalout(self):
    #     config_path = Path(__file__).parent.absolute() / "test_config.ini"
    #     reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

    #     proc = RTprocess(config_path)
    #     proc.override_config(
    #         PROCESS_SPECS=dict(path__save_path="tests/proc_test3", dumpfolder="dump03")
    #     )

    #     proc.run_processing(ncpu=4, reader_args=reader_args, postprocess=False)

    #     results = RTresults("tests/proc_test3")
    #     assert hasattr(results, "dump03"), "dumpfolder dump02 not found by RTresults"

    #     finalout_name = results.dump03.load_cfg().get_process_specs()["finalout_name"]

    #     assert not Path(
    #         f"tests/proc_test3/dump03/results/{finalout_name}.nc"
    #     ).exists(), "disabling postprocess did not work"

    #     proc.run_finaloutput(ncpu=1, finalout_name="ncpu_1.nc")
    #     assert Path(
    #         "tests/proc_test3/dump03/results/ncpu_1.nc"
    #     ).exists(), "run_finalout with ncpu=1 not work"

    #     proc.run_finaloutput(ncpu=4, finalout_name="ncpu_2.nc")
    #     assert Path(
    #         "tests/proc_test3/dump03/results/ncpu_2.nc"
    #     ).exists(), "run_finalout with ncpu=2 not work"

    def test_5_multiconfig(self):

        config_path = Path(__file__).parent.absolute() / "test_config_multi.ini"
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

        proc = RTprocess(config_path)
        proc.run_processing(ncpu=4, reader_args=reader_args, postprocess=True)

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

        dumpfiles = [i for i in results.dump01.dump_files]

        fit = results.dump01.load_fit()
        cfg = results.dump01.load_cfg()

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

        # check if all ID's are present in the HDF-container
        assert all(
            i in fit_db.IDs.values for i in ["RT1_1", "RT1_2", "RT1_3"]
            ), ("HDF-container does not contain all IDs")

        # check if dumped-properties are equal for pickles and HDF-containers
        for fit in results.dump01.dump_fits:
            fit_hdf = fit_db.load_fit(fit.ID)
            hdf_dataset = fit_hdf.dataset[list(fit.dataset)]
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

    # def test_7_multiconfig_finalout(self):
    #     config_path = Path(__file__).parent.absolute() / "test_config_multi.ini"

    #     proc = RTprocess(config_path)
    #     proc.run_finaloutput(
    #         ncpu=1,
    #         finalout_name="ncpu1.nc",
    #     )

    #     proc.run_finaloutput(
    #         ncpu=3,
    #         finalout_name="ncpu3.nc",
    #     )

    #     for cfg in ["cfg_0", "cfg_1"]:
    #         # check if all folders are properly initialized
    #         assert Path(
    #             f"tests/proc_multi/dump01/results/ncpu1__{cfg}.nc"
    #         ).exists(), "multiconfig finaloutput with ncpu=1 did not work"
    #         assert Path(
    #             f"tests/proc_multi/dump01/results/ncpu3__{cfg}.nc"
    #         ).exists(), "multiconfig finaloutput with ncpu=3 did not work"


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

    # def test_91_export_results(self):
    #     for folder in ["proc_test", "proc_test2", "proc_test3", "proc_multi"]:
    #         proc = RTprocess(f"tests/{folder}")

    #         parameters = dict(t_s=dict(long_name="bare soil directionality"),
    #                           tau=dict(long_name="optical depth"),
    #                           sig2=dict(long_name="sigma0 squared")
    #                           )

    #         metrics = dict(R=["pearson", "sig", "tot",
    #                           dict(long_name="sig0 pearson correlation")],
    #                        RMSD=["rmsd", "sig", "tot",
    #                           dict(long_name="sig0 RMSD")]
    #                        )

    #         export_functions = dict(sig2=lambda fit: fit.dataset.sig**2)

    #         attributes = dict(info="some info")

    #         res = proc.export_data(parameters=parameters,
    #                                metrics=metrics,
    #                                export_functions=export_functions,
    #                                attributes=attributes,
    #                                index_col='gpi')

    #         if folder != "proc_multi":
    #             # make single-fit results a dict as well so that they can be treated
    #             # in the same way as multi-fits
    #             res = dict(single_fit=res)

    #         for cfg_name, useres in res.items():
    #             assert useres.R.dims == ('gpi',), "metric dim is wrong"
    #             assert useres.RMSD.dims == ('gpi',), "metric dim is wrong"

    #             assert useres.t_s.dims == ('gpi',), "static parameter dim is wrong"
    #             assert useres.tau.dims == ('gpi', 'date'), "dynamic parameter dim is wrong"
    #             assert useres.sig2.dims == ('gpi', 'date'), "dynamic parameter dim is wrong"

    #             # check if attributes are correctly attached
    #             assert useres.attrs['info'] == attributes['info'], "attributes not correctly attached"
    #             for key, attrs in parameters.items():
    #                 for name, a in attrs.items():
    #                     assert useres[key].attrs[name] == a, "attributes not correctly attached"

    #             assert "model_definition" in useres.attrs, "model_definition not correctly attached"


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
                    i in data.IDs.values for i in ["RT1_1", "RT1_2", "RT1_3"]
                    ), ("HDF-container does not contain all IDs")


                    #TODO add some basic tests to check that the HDF file is OK


if __name__ == "__main__":
    unittest.main()

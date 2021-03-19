from pathlib import Path
import unittest
import unittest.mock as mock
from rt1.rtprocess import RTprocess, RTresults

import warnings

warnings.simplefilter("ignore")


# use "test_0_---"   "test_1_---"   to ensure test-order


class TestRTfits(unittest.TestCase):
    def test_0_parallel_processing(self):
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]
        config_path = Path(__file__).parent.absolute() / "test_config.ini"

        proc = RTprocess(config_path, autocontinue=True)

        proc.run_processing(ncpu=4, reader_args=reader_args)

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
        assert hasattr(results, "dump01"), "dumpfolder not found by RTresults"

        dumpfiles = [i for i in results.dump01.dump_files]
        print(dumpfiles)

        fit = results.dump01.load_fit()
        cfg = results.dump01.load_cfg()

        with results.dump01.load_nc() as ncfile:
            processed_ids = list(ncfile.ID.values)

        processed_ids.sort()
        assert processed_ids == [1, 2, 3], "NetCDF export does not include all IDs"

        # check if NetCDF_variables works as expected
        results.dump01.NetCDF_variables

        # remove the save_path directory
        # print('deleting save_path directory...')
        # shutil.rmtree(results._parent_path)

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
            proc = RTprocess(config_path, autocontinue=False, copy=False)
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
        assert not Path(
            "tests/proc_test/dump01/cfg/test_config.ini"
        ).exists(), "NOT copying did not work"
        assert not Path(
            "tests/proc_test/dump01/cfg/parallel_processing_config.py"
        ).exists(), "NOT copying did not work"

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

    def test_4_postprocess_and_finalout(self):
        config_path = Path(__file__).parent.absolute() / "test_config.ini"
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

        proc = RTprocess(config_path)
        proc.override_config(
            PROCESS_SPECS=dict(path__save_path="tests/proc_test3", dumpfolder="dump03")
        )

        proc.run_processing(ncpu=4, reader_args=reader_args, postprocess=False)

        results = RTresults("tests/proc_test3")
        assert hasattr(results, "dump03"), "dumpfolder dump02 not found by RTresults"

        finalout_name = results.dump03.load_cfg().get_process_specs()["finalout_name"]

        assert not Path(
            f"tests/proc_test3/dump03/results/{finalout_name}.nc"
        ).exists(), "disabling postprocess did not work"

        proc.run_finaloutput(ncpu=1, finalout_name="ncpu_1.nc")
        assert Path(
            "tests/proc_test3/dump03/results/ncpu_1.nc"
        ).exists(), "run_finalout with ncpu=1 not work"

        proc.run_finaloutput(ncpu=4, finalout_name="ncpu_2.nc")
        assert Path(
            "tests/proc_test3/dump03/results/ncpu_2.nc"
        ).exists(), "run_finalout with ncpu=2 not work"

    def test_5_multiconfig(self):

        config_path = Path(__file__).parent.absolute() / "test_config_multi.ini"
        reader_args = [dict(gpi=i) for i in [1, 2, 3, 4]]

        proc = RTprocess(config_path)
        proc.run_processing(ncpu=4, reader_args=reader_args, postprocess=True)

        for cfg in ["cfg_0", "cfg_1"]:
            # check if all folders are properly initialized
            assert Path(
                f"tests/proc_multi/dump01/dumps/{cfg}"
            ).exists(), "multiconfig folder generation did not work"
            assert Path(
                f"tests/proc_multi/dump01/dumps/{cfg}"
            ).exists(), "multiconfig folder generation did not work"

            # check if model-definition files are written correctly
            assert Path(
                f"tests/proc_multi/dump01/cfg/model_definition__{cfg}.txt"
            ).exists(), "multiconfig model_definition.txt export did not work"

            # check if netcdf have been exported
            assert Path(
                f"tests/proc_multi/dump01/results/{cfg}/results.nc"
            ).exists(), "multiconfig NetCDF export did not work"

    def test_7_multiconfig_finalout(self):
        config_path = Path(__file__).parent.absolute() / "test_config_multi.ini"

        proc = RTprocess(config_path)
        proc.run_finaloutput(
            ncpu=1,
            finalout_name="ncpu1.nc",
        )

        proc.run_finaloutput(
            ncpu=3,
            finalout_name="ncpu3.nc",
        )

        for cfg in ["cfg_0", "cfg_1"]:
            # check if all folders are properly initialized
            assert Path(
                f"tests/proc_multi/dump01/results/{cfg}/ncpu1.nc"
            ).exists(), "multiconfig finaloutput with ncpu=1 did not work"
            assert Path(
                f"tests/proc_multi/dump01/results/{cfg}/ncpu3.nc"
            ).exists(), "multiconfig finaloutput with ncpu=3 did not work"

    def test_8_multiconfig_rtresults(self):
        res = RTresults("tests/proc_multi")

        for cfg in ["cfg_0", "cfg_1"]:
            assert hasattr(
                res, f"dump01__{cfg}"
            ), "multi-results are not properly added"

            assert (
                getattr(res, f"dump01__{cfg}")._dump_path.stem == cfg
            ), "multi-result dump-path is not properly set"

    def test_9_multiconfig_props(self):
        # check if properties have been set correctly

        res = RTresults("tests/proc_multi")

        fit = getattr(res, "dump01__cfg_0").load_fit()
        assert fit.defdict["omega"][0] is False, "multiconfig props not correct"
        assert fit.defdict["omega"][1] == 0.5, "multiconfig props not correct"
        assert fit.int_Q is False, "multiconfig props not correct"

        assert str(fit.V.t) == "t_v", "multiconfig props not correct"
        assert fit.SRF.ncoefs == 10, "multiconfig props not correct"

        assert fit.lsq_kwargs["ftol"] == 0.0001, "multiconfig props not correct"

        # -----------------------------

        fit = getattr(res, "dump01__cfg_1").load_fit()
        assert fit.defdict["omega"][0] is True, "multiconfig props not correct"
        assert fit.defdict["omega"][1] == 0.05, "multiconfig props not correct"
        assert fit.defdict["omega"][2] == "2M", "multiconfig props not correct"
        assert fit.int_Q is True, "multiconfig props not correct"

        assert fit.V.t == 0.25, "multiconfig props not correct"
        assert fit.SRF.ncoefs == 5, "multiconfig props not correct"
        assert fit.lsq_kwargs["ftol"] == 0.001, "multiconfig props not correct"


if __name__ == "__main__":
    unittest.main()

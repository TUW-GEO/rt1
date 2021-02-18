"""a module to parse .ini files"""

from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
from functools import partial
import sys
import importlib.util
from pathlib import Path
from .rtfits import Fits
from . import log


class RT1_configparser(object):
    """
    A configparser that can be used to fully specify a RT1-processing routine.

    Extended interpolation is used -> variables can be addressed within the
    config-file by using:

        >>> [class1]
        >>> var1 = asdf
        >>> var2 = ${var1} bsdf
        >>>
        >>> [class2]
        >>> var3 = asdf
        >>> var4 = ${class1:var1} bsdf

    Methods
    -------
    get_config():
        get a dict of the following structure:

        >>> {fits_kwargs : parsed args from "[fits_kwargs]",
        >>>  lsq_kwargs  : parsed args from "[lsq_kwargs]",
        >>>  defdict     : parsed args from "[defdict]",
        >>>  set_V_SRF   : parsed args from "[set_V_SRF]"}

    get_fitobject():
        get the rt1.rtfits.Fits() object

    get_module(NAME):
        programmatically import the module with the name NAME from the path
        "module__NAME" defined in the "[CONFIGFILES]" section of the .ini file
        and return a reference to the module.

    get_all_modules():
        get a dict of ALL defined modules with the following structure:

        >>> {module_name1 : reference to imported module1 from "[CONFIGFILES]",
        >>>  module_name2 : reference to imported module2 from "[CONFIGFILES]"}

    get_process_specs():
        get a dict of the following structure:

        >>> {name1 : parsed value1 from "[PROCESS_SPECS]",
        >>>  name2 : parsed value2 from "[PROCESS_SPECS]",
             ...}

    Notes
    -----
    The sections are interpreted as follows:

    - [fits_kwargs]
        keyword-arguments passed to the `rt1.rtfits.Fits()` object
    - [defdict]
        the defdict passed to the `rt1.rtfits.Fits()` object
    - [RT1_V]
        the specifications of the `rt1.volume` object passed to the
        `rt1.rtfits.Fits()` object
    - [RT1_SRF]
        the specifications of the `rt1.surface` object passed to the
        `rt1.rtfits.Fits()` object
    - [least_squares_kwargs]
        keyword-arguments passed to the call of `scipy.stats.least_squares()`
    - [CONFIGFILES]

        - any passed argument starting with `module__NAME` will be interpreted
          as the location of a python-module that is intended to be imported.

          use .get_module(NAME) to get a reference to the module imported
          from the specified loaction.

          the `.get_all_modules()` function returns a dict of ALL specified
          modules:     processfuncs = {NAME : imported module, ...}

        - if `copy = path to a folder` is provided, both the .ini file as well
          as any file imported via `module__` arguments will be copied to the
          specified folder. All modules will be imported from the files within
          the specified folder.

    - [PROCESS_SPECS]
        additional properties needed to specify the process

        - any argument starting with `int__NAME` will be parsed as
          `integer` with the name `NAME`
        - any argument starting with `float__NAME` will be parsed as
          `float` with the name `NAME`
        - any argument starting with `bool__NAME` will be parsed as
          `bool` with the name `NAME`
        - any argument starting with `path__NAME` will be parsed as
          `pathlib.Path` with the name `NAME`
        - any argument starting with `datetime__NAME` will be parsed as
          `datetime.datetime` with the name `NAME` as follows:

              >>> datetime__NAME = datetime-string  fmt= datetime-format

              for example:

              >>> datetime_d1 = 1.1.2018 fmt=%d%m%Y

              if fmt is not provided, `"%Y-%m-%d %H:%M:%S.%f"` is used.
              Note that the exact string "fmt=" must be used to separate
              the datetime-string from the format-string!
        - any argument starting with list__NAME will be parsed as a `list`
            - list__NAME will be parsed as a list of strings with the
              name NAME
            - list__float__NAME will be parsed as a list of floats with the
              name NAME
            - list__int__NAME will be parsed as a list of integers with the
              name NAME
            - list__bool__NAME will be parsed as a list of bools with the
              name NAME
            - list__datetime__NAME will be parsed as a list of datetimes with
              the name NAME as follows:

              >>> list__datetime__NAME = [df1, dt2, ...]  fmt= datetime-format

              for example:

              >>> list__datetime__dts = [1.1.2018, 1.10.2018] fmt= %d%m%Y
    """

    def __init__(self, configpath, interpolation=ExtendedInterpolation()):

        self.configpath = Path(configpath)
        self.interpolation = interpolation
        # setup config (allow empty values -> will result in None)
        self.config = ConfigParser(
            allow_no_value=True, interpolation=self.interpolation
        )
        # avoid converting uppercase characters
        # (see https://stackoverflow.com/a/19359720/9703451)
        self.config.optionxform = str

        # read config file
        self.cfg = self.config.read(self.configpath)

        # keys that will be converted to int, float or bool
        self.lsq_parse_props = dict(
            section="least_squares_kwargs",
            int_keys=["verbose", "max_nfev"],
            float_keys=["ftol", "gtol", "xtol"],
        )

        self.fitargs_parse_props = dict(
            section="fits_kwargs",
            bool_keys=["sig0", "dB", "int_Q"],
            int_keys=["verbose"],
            list_keys=[],
        )

        self._check_integrity()

    def _check_integrity(self):

        # check if required sections are defined
        required_sections = [
            "PROCESS_SPECS",
            "CONFIGFILES",
            "fits_kwargs",
            "least_squares_kwargs",
            "defdict",
            "RT1_V",
            "RT1_SRF",
        ]
        for key in required_sections:
            assert (
                key in self.config
            ), f'a section "{key}" MUST be present in the .ini file!'

        # check if save_path etc. has been defined
        required_process_spec_keys = ["path__save_path", "dumpfolder"]
        for key in required_process_spec_keys:
            assert (
                key in self.config["PROCESS_SPECS"]
            ), f'a key "{key}" MUST be provided in the section PROCESS_SPECS'

        # check if processing-modue is specified
        if "processing_cfg_module" in self.config["CONFIGFILES"]:
            proc_module_name = self.config["CONFIGFILES"]["processing_cfg_module"]
        else:
            proc_module_name = "processing_cfg"

        assert ("module__" + proc_module_name) in self.config["CONFIGFILES"], (
            '"module__'
            + proc_module_name
            + '" must be defined in the '
            + 'CONFIGFILES section... \n(specify "processing_cfg_module" if '
            + "you want to use a different name for the processing-module key!)"
        )

    def _parse_dict(
        self, section, int_keys=[], float_keys=[], bool_keys=[], list_keys=[]
    ):
        """
        a function to convert the parsed string values to int, float or bool
        (any additional values will be left unchanged)

        Parameters
        ----------
        section : str
            the name of the section in the config-file.
        int_keys : list
            keys that should be converted to int.
        float_keys : list
            keys that should be converted to float.
        bool_keys : list
            keys that should be converted to bool.

        Returns
        -------
        parsed_dict : dict
            a dict with the converted values.
        """

        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            if key in float_keys:
                val = inp.getfloat(key)
            elif key in int_keys:
                val = inp.getint(key)
            elif key in bool_keys:
                val = inp.getboolean(key)
            elif key in list_keys:
                if inp[key] == "None":
                    val = []
                else:
                    val = inp[key].replace(" ", "").split(",")
                    val = [i for i in val if len(i) > 0]
            else:
                # parse None as "real" None
                if inp[key] == "None":
                    val = None
                else:
                    val = inp[key]

            parsed_dict[key] = val
        return parsed_dict

    def _parse_V_SRF(self, section):

        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            val = None
            if key == "ncoefs":
                val = inp.getint(key)
            else:
                # try to convert floats, if it fails return the string
                try:
                    val = inp.getfloat(key)
                except Exception:
                    val = inp[key]

            parsed_dict[key] = val

        return parsed_dict

    def _parse_defdict(self, section):
        inp = self.config[section]

        parsed_dict = dict()
        for key in inp:
            val = inp[key]

            if val.startswith("[") and val.endswith("]"):
                val = val[1:-1].replace(" ", "").split(",")
            else:
                assert False, f"the passed defdict for {key} is not a list"

            # convert values
            parsed_val = []
            if val[0] == "True":
                parsed_val += [True]
            elif val[0] == "False":
                parsed_val += [False]
            else:
                assert False, (
                    f"the passed first defdict-argument of {key} "
                    + f'({val[0]}) must be either "True" or "False"'
                )

            if val[1] == "auxiliary":
                parsed_val += [val[1]]
            else:
                parsed_val += [float(val[1])]

            if len(val) > 2:
                if val[2] == "None":
                    parsed_val += [None]
                else:
                    parsed_val += [val[2]]

                parsed_val += [([float(val[3])], [float(val[4])])]

                try:
                    interp_Q = val[5]
                    if interp_Q == "True":
                        parsed_val += [True]
                    else:
                        parsed_val += [False]
                except Exception:
                    parsed_val += [False]

            parsed_dict[key] = parsed_val

        return parsed_dict

    def _to_dt(self, s, fmt=None):
        if fmt is None:
            fmt = "%Y-%m-%d %H:%M:%S.%f"
        return datetime.strptime(s, fmt)

    def get_config(self):
        """
        get configuration dictionary

        Returns
        -------
        dict
            a dict with the model-configuration.

        """
        lsq_kwargs = self._parse_dict(**self.lsq_parse_props)

        fits_kwargs = self._parse_dict(**self.fitargs_parse_props)
        defdict = self._parse_defdict("defdict")
        set_V_SRF = dict(
            V_props=self._parse_V_SRF("RT1_V"),
            SRF_props=self._parse_V_SRF("RT1_SRF"),
        )

        return dict(
            lsq_kwargs=lsq_kwargs,
            defdict=defdict,
            set_V_SRF=set_V_SRF,
            fits_kwargs=fits_kwargs,
        )

    def get_fitobject(self):
        """
        get the rt1.rtfits.Fits object

        Returns
        -------
        rt1_fits : rt1.rtfits.Fits
            the used Fits object.

        """
        cfg = self.get_config()

        rt1_fits = Fits(
            dataset=None,
            defdict=cfg["defdict"],
            set_V_SRF=cfg["set_V_SRF"],
            lsq_kwargs=cfg["lsq_kwargs"],
            **cfg["fits_kwargs"],
        )

        return rt1_fits

    def get_all_modules(self, load_copy=False):
        """
        programmatically import all defined modules from the paths provided
        in the [CONFIGFILES] section

        Parameters
        ----------
        load_copy : bool, optional
            indicator if the modules should be loaded from the specified path
            or if it should be loaded from "save_path/dumpfolder/cfg".
            The default is False.

        Returns
        -------
        processmodules : dict
            a dict with the imported modules
        """

        processmodules = dict()
        for key, val in self.config["CONFIGFILES"].items():
            if not key.startswith("module__"):
                continue

            modulename = key[8:]
            processmodules[modulename] = self.get_module(
                modulename, load_copy=load_copy
            )

        return processmodules

    def get_module(self, modulename, load_copy=False):
        """
        programmatically import the module 'modulename' as described here:
        https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly


        Parameters
        ----------
        modulename : str
            the module-name (e.g. the key specified in the ".ini"-file).
        load_copy : bool, optional
            indicator if the module should be loaded from the specified path
            or if it should be loaded from "save_path/dumpfolder/cfg".
            The default is False.

        Returns
        -------
        module : the imported module

        """

        assert f"module__{modulename}" in self.config["CONFIGFILES"], (
            f'the module "{modulename}" is not defined in the [CONFIGFILES]'
            + " section of the .ini file     ...available modules are:  "
            + "  ,  ".join(
                [
                    '"' + i.replace("module__", "") + '"'
                    for i in self.config["CONFIGFILES"]
                    if i.startswith("module__")
                ]
            )
        )

        if load_copy is True:
            save_path = Path(self.config["PROCESS_SPECS"]["path__save_path"].strip())

            dumpfolder = self.config["PROCESS_SPECS"]["dumpfolder"].strip()

            modulename = Path(
                self.config["CONFIGFILES"][f"module__{modulename}"].strip()
            ).name

            modulepath = save_path / dumpfolder / "cfg" / modulename
            assert modulepath.exists(), (
                f'the module "{modulename}" has not yet been copied to '
                + f'"{save_path / dumpfolder / "cfg"}" and can not be '
                + 'imported using "load_copy=True"'
            )
        else:
            modulepath = Path(
                self.config["CONFIGFILES"][f"module__{modulename}"].strip()
            )

        spec = importlib.util.spec_from_file_location(
            name=modulepath.stem, location=modulepath
        )

        foo = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = foo
        spec.loader.exec_module(foo)

        return foo

    def get_process_specs(self):
        """
        get the process_specs dict

        Returns
        -------
        dict
            the process specs dict.

        """
        inp = self.config["PROCESS_SPECS"]

        process_specs = dict()
        for key, val in inp.items():
            # parse None as a 'real' None and not as the string 'None'
            if val == "None":
                val = None

            if key.startswith("datetime__"):
                date = dict(zip(["s", "fmt"], [i.strip() for i in val.split("fmt=")]))
                process_specs[key[10:]] = self._to_dt(**date)
            elif key.startswith("path__"):
                # resolve the path in case .. syntax is used to traverse dirs
                if ".." in val.strip():
                    try:
                        process_specs[key[6:]] = Path(val.strip()).resolve()
                    except PermissionError:
                        log.warning(
                            "some paths could not be resolved!"
                            + 'avoid using ".." syntax in path-specifications'
                            + " to circuumvent this issue"
                        )
                        process_specs[key[6:]] = Path(val.strip())
                else:
                    process_specs[key[6:]] = Path(val.strip())

            elif key.startswith("bool__"):
                process_specs[key[6:]] = inp.getboolean(key)
            elif key.startswith("float__"):
                process_specs[key[7:]] = inp.getfloat(key)
            elif key.startswith("int__"):
                process_specs[key[5:]] = inp.getint(key)
            elif key.startswith("list__"):
                listkey = key[6:]
                if val.startswith("[") and (val.endswith("]") or "fmt=" in val):
                    # allow direct conversion of values
                    if listkey.startswith("bool__"):
                        listkey = listkey[6:]
                        conffunc = ConfigParser()._convert_to_boolean
                    elif listkey.startswith("float__"):
                        listkey = listkey[7:]
                        conffunc = float
                    elif listkey.startswith("int__"):
                        listkey = listkey[5:]
                        conffunc = int
                    elif listkey.startswith("datetime__"):
                        listkey = listkey[10:]
                        spl = dict(
                            zip(
                                ["val", "fmt"],
                                [i.strip() for i in val.split("fmt=")],
                            )
                        )
                        if "fmt" in spl:
                            conffunc = partial(self._to_dt, fmt=spl["fmt"])
                        else:
                            conffunc = self._to_dt
                        val = spl["val"]
                    else:

                        def conffunc(x):
                            return x

                    process_specs[listkey] = [
                        conffunc(i.strip()) for i in val.strip()[1:-1].split(",")
                    ]

                else:
                    assert False, f"{key} is not a list! (use brackets [] !)"
            else:
                process_specs[key] = val

        return process_specs

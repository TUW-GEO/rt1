import copy


class _RT1_defdict:
    def __init__(self):
        self._variables = set()

    def __repr__(self):
        return "\n".join(
            [f"{name}: {repr(getattr(self, name))}" for name in self._variables]
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, key, val):
        if not key.startswith("_") and key not in self._variables:
            # set the value of the corresponding property on item assignment
            if isinstance(val, dict):
                self.add_variable(key, **val)
            elif isinstance(val, list):
                self.add_variable(key, *val)
            else:
                raise AttributeError(
                    "RT1 defdicts only support setting new variables"
                    + " via lists or dicts!"
                )
        else:
            object.__setattr__(self, key, val)

    def __iter__(self):
        return iter(self._variables)

    def add_variable(
        self, name, fitQ=True, val=0.5, freq=None, minval=0.0, maxval=1.0, interpQ=False
    ):
        """
        Add a new definition for a variable to the defdict.

        Parameters
        ----------
        name : str
            the name of the variable (as specified in set_V_SRF).
        fitQ : bool, optional
            indicator if the quantity should be fitted (True) or used as a constant
            during the fit (False). The default is True.
        val : float, optional
            - If figQ is True:
              - if float: The (constant) start-value for the fit.
              - if "auxiliary": The start-values for the fit are evaluated via the
              mean-values of each "dyn-group" of the dataset-column '<name>_start'.

            - if fitQ is False:
              - if float: The (constant) value that will be assigned to the variable.
              - if "auxiliary": The corresponding column "<name>" of the provided
              dataset will be used as values for the parameter.

            The default is 0.5.
        freq : str or None, optional
            Only relevant if fitQ is True.

            - if None, a constant value will be fitted
            - if "manual", the DataFrame column "<name>_dyn" provided
              in the dataset will be used to assign the temporal
              variability within the fit
            - if "index", a unique value will be fitted to each
              unique index of the provided dataset
            - if freq corresponds to a pandas offset-alias, it
              will be used together with the dataset-index to
              assign the temporal variability within the fit
              (see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)
            - if freq is an integer (N), the dataset will be grouped
              such that each group contains N unique dataset-indexes
              (in case an exact split is not possible, the split is
               performed such that the groups are as similar as
               possible)
            - if "freq + manual" AND a dataset-column "key_dyn" is
              provided, the the provided variability will be
              superimposed onto the variability resulting
              form the chosen offset-alias

            The default is None.
        minval, maxval : float, optional
            Only relevant if fitQ is True.

            The boundary-values used within the fit. The default is (0, 1).
        interpQ : bool, optional
            Indicator if the obtained values should be interpolated to the
            dataset-timestamps via a quadratic interpolation function or if a
            simple step-function should be used. (only affects parameters
            where freq is not None). The default is False.
        """

        if name not in self._variables:
            self._variables.add(name)
        obj = _RT1_variable(
            name=name,
            fitQ=fitQ,
            val=val,
            freq=freq,
            minval=minval,
            maxval=maxval,
            interpQ=interpQ,
        )
        object.__setattr__(
            self,
            name,
            obj,
        )

    def remove_variable(self, name):
        """
        Remove a variable from the defdict.

        Parameters
        ----------
        name : str
            The name of the variable to remove.
        """
        if name in self._variables:
            self._variables.remove(name)
        delattr(self, name)

    @classmethod
    def from_dict(cls, d, copy_vals=True):
        """
        Initialize a RT1-defdict object from a given dict.

        Parameters
        ----------
        d : dict
            a dict of the following shape:
                {key : [fitQ, val, freq, ([minval], [maxval]), interpQ]}
            or
                {key : {fitQ=.., val=.., freq=.., minval=.., maxval=.., interpQ=..}}
        copy_vals : bool
            Indicator if a (deep)-copy of the values should be performed or not
            The default is True

        Returns
        -------
        defdict : _RT1_defdict
            The RT1-defdict with all variables assigned according to the provided dict.

        """

        defdict = cls()
        for key, val in d.items():
            defdict.add_variable(key)
            if isinstance(val, list):
                getattr(defdict, key).from_list(
                    copy.deepcopy(val) if copy_vals else val
                )
            elif isinstance(val, dict):
                getattr(defdict, key).from_dict(
                    copy.deepcopy(val) if copy_vals else val
                )
        return defdict

    def to_dict(self, props="list"):
        """
        Convert the RT1-defdict to an ordinary dict

        Parameters
        ----------
        props : str, optional
            Indicator if lists or dicts should be used for the properties.
            The default is "list".

        Returns
        -------
        d : dict
            A dict-representation of the definitions.

        """
        d = dict()
        if props == "list":
            for name in self._variables:
                d[name] = getattr(self, name).to_list()
        elif props == "dict":
            for name in self._variables:
                d[name] = getattr(self, name).to_dict()

        return d

    def items(self):
        return self.to_dict().items()

    def keys(self):
        return self._variables


class _var:
    def __init__(self, name, val):
        self._name = name
        self._val = val
        pass

    def __repr__(self):
        return f"{self._name}: {self._val}"

    def __setitem__(self, val):
        self._val = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, val):
        self._val = val


class _RT1_variable:
    def __init__(
        self,
        name=None,
        fitQ=None,
        val=None,
        freq=None,
        minval=None,
        maxval=None,
        interpQ=None,
    ):

        self._name = _var("name", name)
        self._fitQ = _var("fitQ", fitQ)
        self._val = _var("val", val)
        self._freq = _var("freq", freq)
        self._minval = _var("minval", minval)
        self._maxval = _var("maxval", maxval)
        self._interpQ = _var("interpQ", interpQ)

    def __repr__(self):
        return str(self.props)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.props[key]
        else:
            return getattr(self, key)

    def __setitem__(self, key, val):
        # set the value of the corresponding property on item assignment
        getattr(self, key).value = val

    def __setattr__(self, key, val):
        if not key.startswith("_"):
            # set the value of the corresponding property on item assignment
            getattr(self, key).value = val
        else:
            object.__setattr__(self, key, val)

    @property
    def name(self):
        return self._name

    @property
    def fitQ(self):
        return self._fitQ

    @property
    def val(self):
        return self._val

    @property
    def freq(self):
        return self._freq

    @property
    def minval(self):
        return self._minval

    @property
    def maxval(self):
        return self._maxval

    @property
    def interpQ(self):
        return self._interpQ

    @property
    def props(self):
        return [
            self.fitQ,
            self.val,
            self.freq,
            ([self.minval], [self.maxval]),
            self.interpQ,
        ]

    def from_list(self, l):
        """
        Parse properties of a variable from a list

        Parameters
        ----------
        l : list
            A list of the specifications in the following shape:

                [fitQ, val, freq, ([minval], [maxval]), interpQ]
        """
        assert isinstance(l, list), "The provided value must be a list!"
        [
            self.fitQ,
            self.val,
            self.freq,
            [[self.minval], [self.maxval]],
            self.interpQ,
        ] = (
            l + [None, None, None, [[None], [None]], None][len(l) :]
        )

    def to_list(self):
        """
        Get a list of the properties

        Returns
        -------
        list
            [fitQ, val, freq, ([minval], [maxval]), interpQ]

        """
        return [
            self.fitQ.value,
            self.val.value,
            self.freq.value,
            ([self.minval.value], [self.maxval.value]),
            self.interpQ.value,
        ]

    def from_dict(self, d):
        """
        Parse properties of a variable from a dict

        Parameters
        ----------
        d : dict
            A dict of the specifications with the following keys:

            - "fitQ", "val", "freq", "minval", "maxval", "interpQ"
        """

        for key, val in d.items():
            setattr(self, key, val)

    def to_dict(self):
        """
        Get a dict of the properties

        Returns
        -------
        list
            A dict of the properties
        """

        return dict(
            fitQ=self.fitQ.value,
            val=self.val.value,
            freq=self.freq.value,
            minval=self.minval.value,
            maxval=self.maxval.value,
            interpQ=self.interpQ.value,
        )

"""
Class for quick visualization of results and used phasefunctions
"""

from functools import partial
import copy
import datetime

import numpy as np
import sympy as sp
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.ticker as ticker

try:
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass

from .general_functions import (
    rectangularize,
    dBsig0convert,
    meandatetime,
    pairwise,
    split_into,
)


def polarplot(
    X=None,
    inc=[15.0, 35.0, 55.0, 75.0],
    multip=2.0,
    label=None,
    aprox=True,
    legend=True,
    legpos=(0.75, 0.5),
    groundcolor="none",
    param_dict=[{}],
    polarax=None,
):
    """
    Generation of polar-plots of the volume- and the surface scattering
    phase function as well as the used approximations in terms of
    legendre-polynomials.


    Parameters
    -----------
    SRF : RT1.surface class object
          Alternative direct specification of the surface BRDF,
          e.g. SRF = CosineLobe(i=3, ncoefs=5)
    V : RT1.volume class object
        Alternative direct specification of the volume-scattering
        phase-function  e.g. V = Rayleigh()

    Other Parameters
    -----------------
    inp : list of floats (default = [15.,35.,55.,75.])
           Incidence-angles in degree at which the volume-scattering
           phase-function will be plotted
    multip : float (default = 2.)
              Multiplicator to scale the plotrange for the plot of the
              volume-scattering phase-function
              (the max-plotrange is given by the max. value of V in
              forward-direction (for the chosen incp) )
    label : string
             Manual label for the volume-scattering phase-function plot
    aprox : boolean (default = True)
             Indicator if the approximation of the phase-function in terms
             of Legendre-polynomials will be plotted.
    legend : boolean (default = True)
             Indicator if a legend should be shown that indicates the
             meaning of the different colors for the phase-function
    legpos : (float,float) (default = (0.75,0.5))
             Positioning of the legend for the V-plot (controlled via
             the matplotlib.legend keyword  bbox_to_anchor = plegpos )
    groundcolor : string (default = "none")
             Matplotlib color-indicator to change the color of the lower
             hemisphere in the BRDF-plot possible values are:
             ('r', 'g' , 'b' , 'c' , 'm' , 'y' , 'k' , 'w' , 'none')
    polarax: matplotlib.axes
             the axes to use... it must be a polar-axes, e.g.:

                 >>> polarax = fig.add_subplot(111, projection='polar')

    Returns
    ---------
    polarfig : figure
               a matplotlib figure showing a polar-plot of the functions
               specified by V or SRF
    """

    assert isinstance(inc, list), (
        "Error: incidence-angles for " + "polarplot must be a list"
    )
    assert isinstance(multip, float), (
        "Error: plotrange-multiplier " + "for polarplot must be a floating-point number"
    )

    if X is None:
        assert False, "Error: You must provide a volume- or surface object!"

    if polarax is None:
        fig = plt.figure(figsize=(7, 7))
        polarax = fig.add_subplot(111, projection="polar")
    else:
        assert polarax.name == "polar", "you must provide a polar-axes!"

    if ".surface." in str(X.__class__):
        if label is None:
            label = "Surface-Scattering Phase Function"
        funcname = "brdf"
        angs = ["theta_ex", "theta_s", "phi_ex", "phi_s"]

        def angsub(ti):
            return ti

        thetass = np.arange(-np.pi / 2.0, np.pi / 2.0, 0.01)

        polarax.fill(
            np.arange(np.pi / 2.0, 3.0 * np.pi / 2.0, 0.01),
            np.ones_like(np.arange(np.pi / 2.0, 3.0 * np.pi / 2.0, 0.01)) * 1 * 1.2,
            color=groundcolor,
        )

    if ".volume." in str(X.__class__):
        if label is None:
            label = "Volume-Scattering Phase Function"

        funcname = "p"
        angs = ["theta_0", "theta_s", "phi_0", "phi_s"]

        def angsub(ti):
            return np.pi - ti

        thetass = np.arange(0.0, 2.0 * np.pi, 0.01)

    # plot of volume-scattering phase-function's
    pmax = 0
    for n_X, X in enumerate(np.atleast_1d(X)):
        # define a plotfunction of the legendre-approximation of p
        if aprox is True:
            phasefunktapprox = sp.lambdify(
                (*angs, *param_dict[n_X].keys()),
                X.legexpansion(*angs, geometry="vvvv").doit(),
                modules=["numpy", "sympy"],
            )

        # set incidence-angles for which p is calculated
        plottis = np.deg2rad(inc)
        colors = ["k", "r", "g", "b", "c", "m", "y"] * int(
            round((len(plottis) / 7.0 + 1))
        )

        for i in plottis:
            ts = np.arange(0.0, 2.0 * np.pi, 0.01)
            pmax_i = multip * np.max(
                getattr(X, funcname)(
                    np.full_like(ts, i),
                    ts,
                    0.0,
                    0.0,
                    param_dict=param_dict[n_X],
                )
            )
            if pmax_i > pmax:
                pmax = pmax_i

        if legend is True:
            legend_lines = []

        # set color-counter to 0
        i = 0
        for ti in plottis:
            color = colors[i]
            i = i + 1
            rad = getattr(X, funcname)(
                ti, thetass, 0.0, 0.0, param_dict=param_dict[n_X]
            )
            if aprox is True:
                # the use of np.pi-ti stems from the definition
                # of legexpansion() in volume.py
                radapprox = phasefunktapprox(
                    angsub(ti), thetass, 0.0, 0.0, **param_dict[n_X]
                )
            # set theta direction to clockwise
            polarax.set_theta_direction(-1)
            # set theta to start at z-axis
            polarax.set_theta_offset(np.pi / 2.0)

            polarax.plot(thetass, rad, color)
            if aprox is True:
                polarax.plot(thetass, radapprox, color + "--")
            polarax.arrow(
                -ti,
                pmax * 1.2,
                0.0,
                -pmax * 0.8,
                head_width=0.0,
                head_length=0.0,
                fc=color,
                ec=color,
                lw=1,
                alpha=0.3,
            )

            polarax.fill_between(thetass, rad, alpha=0.2, color=color)
            polarax.set_xticks(np.deg2rad([0, 45, 90, 125, 180]))
            polarax.set_xticklabels(
                [
                    r"$0^\circ$",
                    r"$45^\circ$",
                    r"$90^\circ$",
                    r"$135^\circ$",
                    r"$180^\circ$",
                ]
            )
            polarax.set_yticklabels([])
            polarax.set_rmax(pmax * 1.2)
            polarax.set_title(label + "\n")
            polarax.set_rmin(0.0)
    # add legend for covering layer phase-functions
    if legend is True:
        i = 0
        for ti in plottis:
            color = colors[i]
            legend_lines += [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    label=r"$\theta_0$ = "
                    + str(np.round_(np.rad2deg(ti), decimals=1))
                    + r"${}^\circ$",
                )
            ]
            i = i + 1

        if aprox is True:
            legend_lines += [
                mlines.Line2D([], [], color="k", linestyle="--", label="approx.")
            ]

        legend = polarax.legend(bbox_to_anchor=legpos, loc=2, handles=legend_lines)
        legend.get_frame().set_facecolor("w")
        legend.get_frame().set_alpha(0.5)

    return fig


def hemreflect(
    R=None,
    SRF=None,
    phi_0=0.0,
    t_0_step=5.0,
    t_0_min=0.0,
    t_0_max=90.0,
    simps_N=1000,
    showpoints=True,
    returnarray=False,
    param_dict={},
):
    """
    Numerical evaluation of the hemispherical reflectance of the given
    BRDF-function using scipy's implementation of the Simpson-rule
    integration scheme.

    Parameters
    ------------
    R : RT1-class object
        definition of the brdf-function to be evaluated
        (either R or SRF  must be provided) The BRDf is defined via:

            BRDF = R.SRF.NormBRDF * R.SRF.brdf()
    SRF : Surface-class object
          definition of the brdf-function to be evaluated
          (either R or SRF must be provided) The BRDf is defined via:

              BRDF = SRF.NormBRDF * SRF.brdf()

    Other Parameters
    -----------------
    phi_0 : float
            incident azimuth-angle
            (for spherically symmetric phase-functions the result is
            independent of the choice of phi_0)
    t_0_step : float
               separation of the incidence-angle grid-spacing in DEGREE
               for which the hemispherical reflectance will be calculated
    t_0_min : float
              minimum incidence-angle
    t_0_max : float
              maximum incidence-angle
    simps_N : integer
              number of points used in the discretization of the brdf
              within the Simpson-rule
    showpoints : boolean
                 show or hide integration-points in the plot
    param_dict : dict
                 a dictionary containing the names and values of the symbolic
                 parameters required to define the SRF function

    Returns
    --------
    fig : figure
        a matplotlib figure showing the incidence-angle dependent
        hemispherical reflectance
    """

    from scipy.integrate import simps

    # choose BRDF function to be evaluated
    if R is not None:
        BRDF = R.SRF.brdf

        try:
            Nsymb = R.SRF.NormBRDF.free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF, modules=["numpy"])
            NormBRDF = Nfunc(*[param_dict[str(i)] for i in Nsymb])
        except Exception:
            NormBRDF = R.SRF.NormBRDF
    elif SRF is not None:
        BRDF = SRF.brdf
        try:
            Nsymb = SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, SRF.NormBRDF, modules=["numpy"])
            NormBRDF = Nfunc(*[param_dict[str(i)] for i in Nsymb])
        except Exception:
            NormBRDF = SRF.NormBRDF
    else:
        assert False, "Error: You must provide either R or SRF"

    # set incident (zenith-angle) directions for which the integral
    # should be evaluated!
    incnum = np.arange(t_0_min, t_0_max, t_0_step)

    # define grid for integration
    x = np.linspace(0.0, np.pi / 2.0, simps_N)
    y = np.linspace(0.0, 2 * np.pi, simps_N)

    # initialize array for solutions

    sol = []

    # ---- evaluation of Integral
    # adapted from
    # (http://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy)

    for theta_0 in np.deg2rad(incnum):
        # define the function that has to be integrated
        # (i.e. Eq.20 in the paper)
        # notice the additional  np.sin(thetas)  which oritinates from
        # integrating over theta_s instead of mu_s
        def integfunkt(theta_s, phi_s):
            return (
                np.sin(theta_s)
                * np.cos(theta_s)
                * BRDF(theta_0, theta_s, phi_0, phi_s, param_dict=param_dict)
            )

        # evaluate the integral using Simpson's Rule twice
        z = integfunkt(x[:, None], y)
        sol = sol + [simps(simps(z, y), x)]

    sol = np.array(sol) * NormBRDF

    # print warning if the hemispherical reflectance exceeds 1
    if np.any(sol > 1.0):
        print("ATTENTION, Hemispherical Reflectance > 1 !")

    if returnarray is True:
        return sol
    else:
        # generation of plot
        fig = plt.figure()
        axnum = fig.add_subplot(1, 1, 1)

        if len(sol.shape) > 1:
            for i, sol in enumerate(sol):
                axnum.plot(incnum, sol, label="NormBRDF = " + str(NormBRDF[i]))
                if showpoints is True:
                    axnum.plot(incnum, sol, "r.")
        else:
            axnum.plot(incnum, sol, "k", label="NormBRDF = " + str(NormBRDF))
            if showpoints is True:
                axnum.plot(incnum, sol, "r.")

        axnum.set_xlabel("$\\theta_0$ [deg]")
        axnum.set_ylabel("$R(\\theta_0)$")
        axnum.set_title("Hemispherical reflectance ")
        axnum.set_ylim(0.0, np.max(sol) * 1.1)

        axnum.legend()

        axnum.grid()
        return fig


class plot:
    """
    Generation of plots to visualize rtfits results

    - scatter
        generate a scatterplot of measured vs. modelled data

    - fit_timeseries
        generate a plot showing the temporal and incidence-angle dependency
        of measured vs. modelled data.

    - fit_errors
        generate a plot showing the temporal and incidence-angle dependency
        of the residuals of measured vs. modelled data.

    - results
        generate a plot showing the fitted curves and the obtained parameter
        timeseries

    - single_results
        plot the data and the fitted curves for individual measurement-groups
        as defined by the frequencies of the fitted parameters

    - intermediate_results
        only available if *performfit* has been called with
        *intermediate_results = True* !

        generate a plot showing the development of the fitted parameters
        and the residuals for each fit-iteration

    - printsig0analysis
        a widget to analyze the fit-results of individual timestamps within
        the considered timeseries

    """

    def __init__(self, fit=None, **kwargs):
        self.fit = fit

    def scatter(self, fit=None, mima=None, pointsize=0.5, regression=True, **kwargs):
        """
        geerate a scatterplot of modelled vs. original backscatter data

        Parameters
        -----------
        fit : list
              output of performfit()-function
        Other Parameters
        ------------------
        mima : list
               manual definition plot-boundaries via mima = [min, max]
        pointsize : float
                    manual specification of pointsize
        regression : bool (default = True)
                     indicator if the scipy.stats.linregress should be called
                     to get the regression-line and the r^2 value
        kwargs : -
                 kwargs passed to matplotlib.pyplot.scatter()

        Returns
        -------
        fig : matplotlib.figure
            the used matplotlib figure instance
        """
        plot("asdf")
        if fit is None:
            fit = self.fit

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # prepare measurements
        # measures = fit.dataset.sig.values
        # calculate estimates
        # estimates = fit.calc_model().tot.values
        measures, estimates = pd.concat(
            [fit.dataset.sig, fit.calc_model().tot], axis=1
        ).values.T
        if mima is None:
            mi = np.min((measures, estimates))
            ma = np.max((measures, estimates))
        else:
            mi, ma = mima

        ax.scatter(estimates, measures, s=pointsize, alpha=0.7, **kwargs)

        # plot 45degree-line
        ax.plot([mi, ma], [mi, ma], "k--")

        if fit.sig0 is True:
            quantity = r"$\sigma_0$"
        else:
            quantity = "Intensity"

        if fit.dB is True:
            scale = "[dB]"
        else:
            scale = ""

        ax.set_xlabel("modelled " + quantity + scale)
        ax.set_ylabel("measured " + quantity + scale)

        if regression is True:
            from scipy.stats import linregress

            # evaluate linear regression to get r-value etc.
            slope, intercept, r_value, p_value, std_err = linregress(
                estimates, measures
            )

            ax.plot(
                np.sort(measures),
                intercept + slope * np.sort(measures),
                "r--",
                alpha=0.4,
            )

            ax.text(
                0.8,
                0.1,
                "$R^2$ = " + str(np.round(r_value ** 2, 2)),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        return fig

    def fit_timeseries(
        self,
        fit=None,
        dB=True,
        sig0=True,
        params=None,
        printtot=True,
        printsurf=True,
        printvol=True,
        printint=True,
        printorig=True,
        months=None,
        years=None,
        ylim=None,
        printinc=True,
    ):
        """
        Print individual contributions, resulting parameters and the
        reference dataset of an rt1.rtfits object as timeseries.

        Parameters
        -------------
        fit : rtfits object
              the rtfits-object containing the fit-results
        dB : bool (default = True)
             indicator if the plot is intended to be in dB or linear units
        sig0 : bool (default = True)
             indicator if sigma_0 (sig0) or intensity (I) is displayed
             The applied relation is: sig0 = 4.*pi*cos(theta) * I
        params: list
                a list of parameter-names that should be overprinted
                (the names must coincide with the arguments of set_V_SRF())
        printtot, printsurf, printvol, printint, printorig : bool
                indicators if the corresponding components should be plotted
        months : list of int (default = None)
                 a list of months to plot (if None, all will be plotted)
        years : list of int (default = None)
                a list of years to select (if None, all will be plotted)
        ylim : tuple
               a tuple of (ymin, ymax) that will be used as boundaries for the
               y-axis
        printinc : bool (default = True)
                   indicator if the incidence-angle dependency should be
                   plotted (in a separate plot alongside the timeseries)

        Returns
        --------------
        f : matplotlib.figure
            the used matplotlib figure instance
        """

        if fit is None:
            fit = self.fit

        # get incidence-angles
        inc_array = np.ma.masked_array(fit.inc, fit.mask)
        inc = inc_array.compressed()
        # get input dataset
        data = np.ma.masked_array(fit.data, fit.mask)

        # calculate individual contributions
        contrib = fit.calc_model(return_components=True)
        contrib["$\\sigma_0$ dataset"] = fit.dataset.sig
        contrib["inc"] = fit.dataset.inc

        # convert units
        complist = [i for i in contrib.keys() if i not in ["inc"]]
        contrib[complist] = contrib[complist].apply(
            dBsig0convert,
            inc=inc,
            dB=dB,
            sig0=sig0,
            fitdB=fit.dB,
            fitsig0=fit.sig0,
        )

        # drop unneeded columns
        if printint is False and "inter" in contrib:
            contrib = contrib.drop("inter", axis=1)
        if printtot is False:
            contrib = contrib.drop("tot", axis=1)
        if printsurf is False:
            contrib = contrib.drop("surf", axis=1)
        if printvol is False:
            contrib = contrib.drop("vol", axis=1)
        if printorig is False:
            contrib = contrib.drop("$\\sigma_0$ dataset", axis=1)

        # select years and months
        if years is not None:
            contrib = contrib.loc[contrib.index.year.isin(years)]
        if months is not None:
            contrib = contrib.loc[contrib.index.month.isin(months)]

        # print incidence-angle dependency
        if printinc is True:
            f, [ax, ax_inc] = plt.subplots(
                ncols=2,
                figsize=(15, 5),
                gridspec_kw={"width_ratios": [3, 1]},
                sharey=True,
            )
            f.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.1, wspace=0.1)

            # -------------------
            color = {
                "tot": "r",
                "surf": "b",
                "vol": "g",
                "inter": "y",
                "$\\sigma_0$ dataset": "k",
            }

            groupedcontrib = contrib.groupby(contrib.index)

            # return contrib, groupedcontrib
            for label in contrib.keys():
                if label in ["inc"]:
                    continue
                a = np.rad2deg(
                    rectangularize(
                        [x.values for _, x in groupedcontrib["inc"]],
                        return_masked=True,
                    )
                ).T
                b = np.array(
                    rectangularize(
                        [x.values for _, x in groupedcontrib[label]],
                        return_masked=True,
                    )
                ).T
                x = np.array([a, b]).T

                l_col = mpl.collections.LineCollection(
                    x, linewidth=0.25, label="x", color=color[label], alpha=0.5
                )
                ax_inc.add_collection(l_col)
                ax_inc.scatter(a, b, color=color[label], s=1)
                ax_inc.set_xlim(a.min(), a.max())
                ax_inc.set_xlabel("$\\theta_0$")
            ax_inc.set_xlabel("$\\theta_0$")

        else:
            f, ax = plt.subplots(figsize=(12, 5))
            f.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.1, wspace=0.05)

        for label, val in contrib.items():
            if label in ["inc"]:
                continue
            color = {"tot": "r", "surf": "b", "vol": "g", "inter": "y"}
            if printorig is True:
                color["$\\sigma_0$ dataset"] = "k"
            ax.plot(
                val.sort_index(),
                linewidth=0.25,
                marker=".",
                ms=2,
                label=label,
                color=color[label],
                alpha=0.5,
            )
        # overprint parameters
        if params != None:
            paramdf = fit.res_df

            if years is not None:
                paramdf = paramdf.loc[paramdf.index.year.isin(years)]
            if months is not None:
                paramdf = paramdf.loc[paramdf.index.month.isin(months)]

            pax = ax.twinx()
            for k in params:
                pax.plot(paramdf[k], lw=1, marker=".", ms=2, label=k)
            pax.legend(loc="upper right", ncol=5)
            pax.set_ylabel("parameter-values")

        # format datetime index
        ax.xaxis.set_minor_locator(mpl.dates.MonthLocator())
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%m"))
        ax.xaxis.set_major_locator(mpl.dates.YearLocator())
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("\n%Y"))

        # set ylabels
        if sig0 is True:
            label = "$\\sigma_0$"
        else:
            label = "Intensity"
        if dB is True:
            label += " [dB]"
        ax.set_ylabel(label)

        # generate legend
        hand, lab = ax.get_legend_handles_labels()
        lab, unique_ind = np.unique(lab, return_index=True)
        ax.legend(
            handles=list(np.array(hand)[unique_ind]),
            labels=list(lab),
            loc="upper left",
            ncol=5,
        )

        if ylim is not None:
            ax.set_ylim(ylim)

        return f

    def fit_errors(self, fit=None, relative=False, result_selection="all"):
        """
        a function to quickly print residuals for each measurement
        and for each incidence-angle value

        Parameters
        ------------
        fit : list
            output of performfit()-function
        relative : bool (default = False)
                   indicator if relative (True) or absolute (False) residuals
                   shall be plotted

        Returns
        --------------
        fig : matplotlib.figure
            the used matplotlib figure instance
        """

        if fit is None:
            fit = self.fit

        if result_selection == "all":
            result_selection = range(len(fit.data))

        # Calculate the residuals (based on R, inc and res_dict)

        fit.R.t_0 = fit.inc
        fit.R.p_0 = np.zeros_like(fit.inc)

        estimates = fit._calc_model()
        # calculate the residuals based on masked arrays
        masked_estimates = np.ma.masked_array(estimates, mask=fit.mask)
        masked_data = np.ma.masked_array(fit.data, mask=fit.mask)

        res = np.ma.sqrt((masked_estimates - masked_data) ** 2)

        if relative is True:
            res = res / masked_estimates

        # apply mask to data and incidence-angles (and convert to degree)
        inc = np.ma.masked_array(np.rad2deg(fit.inc), mask=fit.mask)

        # make new figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(212)
        if relative is True:
            ax.set_title("Mean relative residual per measurement")
        else:
            ax.set_title("Mean absolute residual per measurement")

        ax2 = fig.add_subplot(211)
        if relative is True:
            ax2.set_title("Relative residuals per incidence-angle")
        else:
            ax2.set_title("Residuals per incidence-angle")

        # the use of masked arrays might cause python 2 compatibility issues!
        ax.plot(fit.index[result_selection], res[result_selection], ".", alpha=0.5)

        # plot mean residual for each measurement
        ax.plot(
            fit.index[result_selection],
            np.ma.mean(res[result_selection], axis=1),
            "k",
            linewidth=3,
            marker="o",
            fillstyle="none",
        )

        # plot total mean of mean residuals per measurement
        ax.plot(
            fit.index[result_selection],
            [np.ma.mean(np.ma.mean(res[result_selection], axis=1))]
            * len(result_selection),
            "k--",
        )

        # add some legends
        res_h = mlines.Line2D(
            [],
            [],
            color="black",
            label="Mean res.  per measurement",
            linestyle="-",
            linewidth=3,
            marker="o",
            fillstyle="none",
        )
        res_h_dash = mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            label="Average mean res.",
            linewidth=1,
            fillstyle="none",
        )

        res_h_dots = mlines.Line2D(
            [],
            [],
            color="black",
            label="Residuals",
            linestyle="-",
            linewidth=0,
            marker=".",
            alpha=0.5,
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [res_h_dots] + [res_h] + [res_h_dash], loc=1)

        ax.set_ylabel("Residual")

        #        # evaluate mean residuals per incidence-angle
        meanincs = np.ma.unique(np.concatenate(inc[result_selection]))
        mean = np.full_like(meanincs, 0.0)

        for a, incval in enumerate(meanincs):
            select = np.where(inc[result_selection] == incval)
            res_selected = res[result_selection][
                select[0][:, np.newaxis], select[1][:, np.newaxis]
            ]
            mean[a] = np.ma.mean(res_selected)

        sortpattern = np.argsort(meanincs)
        meanincs = meanincs[sortpattern]
        mean = mean[sortpattern]

        # plot residuals per incidence-angle for each measurement
        for i, resval in enumerate(res[result_selection]):
            sortpattern = np.argsort(inc[result_selection[i]])
            ax2.plot(
                inc[result_selection[i]][sortpattern],
                resval[sortpattern],
                ":",
                alpha=0.5,
                marker=".",
            )

        # plot mean residual per incidence-angle
        ax2.plot(meanincs, mean, "k", linewidth=3, marker="o", fillstyle="none")

        # add some legends
        res_h2 = mlines.Line2D(
            [],
            [],
            color="black",
            label="Mean res.  per inc-angle",
            linestyle="-",
            linewidth=3,
            marker="o",
            fillstyle="none",
        )
        res_h_lines = mlines.Line2D(
            [], [], color="black", label="Residuals", linestyle=":", alpha=0.5
        )

        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles2 + [res_h_lines] + [res_h2], loc=1)

        ax2.set_xlabel("$\\theta_0$ [deg]")
        ax2.set_ylabel("Residual")

        # find minimum and maximum incidence angle
        maxinc = np.max(inc)
        mininc = np.min(inc)

        ax2.set_xlim(np.floor(mininc) - 1, np.ceil(maxinc) + 1)

        # set major and minor ticks
        ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

        # set ticks
        if isinstance(fit.index[0], datetime.datetime):
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.tight_layout()

        return fig

    def results(
        self,
        fit=None,
        startvals=False,
        legend=False,
        result_selection="all",
        legend_fmt="%d.%m.%Y %H:%M:%S",
    ):
        """
        a function to quickly print the fit-results and the gained parameters

        Parameters
        ------------
        fit : list
              output of performfit()-function
        truevals : dict (default = None)
                   dictionary of the expected parameter-values (must be of the
                   same shape as the parameter-values gained from the fit).
                   if provided, the difference between the expected- and
                   fitted values is plotted
        startvals : bool (default = False)
                    if True, the model-results using the start-values are
                    plotted as black lines
        legend : bool (default = True)
                  indicator if legends should be plotted
        legend_fmt : str (default = '%d.%m.%Y %H:%M:%S')
                     the datetime-format to use when printing a legend
        result_selection : list-like or 'all'
                           a list of the measurement-numbers that should be
                           plotted (indexed starting from 0) or 'all' in case
                           all measurements should be plotted
        Returns
        --------------
        fig : matplotlib.figure
              the used matplotlib figure instance
        """

        if fit is None:
            fit = self.fit

        if result_selection == "all":
            result_selection = range(len(fit.data))

        # generate figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(211)
        ax.set_title("Fit-results")

        # calculate results
        fitplot = fit._calc_model()
        # get masked data
        masked_data = np.ma.masked_array(fit.data, fit.mask)
        # get labels
        try:
            labels = pd.to_datetime(fit.fit_index).strftime(legend_fmt)[
                result_selection
            ]
        except:
            labels = result_selection

        for i in result_selection:
            (l,) = ax.plot(fit.inc[i], fitplot[i], alpha=0.4, label=labels[i])
            ax.plot(fit.inc[i], masked_data[i], ".", c=l.get_color())

        if legend is True:
            if len(result_selection) > 20:
                print("more than 20 lines are plotted... legend is disabled")
            else:
                ax.legend(loc=1)

        # ----------- plot start-values ------------
        if startvals is True:
            startplot = fit._calc_model(res_dict=fit.start_dict)
            for i, val in enumerate(startplot[result_selection]):
                if i == 0:
                    label = "fitstart"
                else:
                    label = ""
                ax.plot(
                    fit.inc[result_selection[i]],
                    val,
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    label=label,
                )

        if fit.sig0 is False:
            label = "$I^{tot}$"
        elif fit.sig0 is True:
            label = r"$\sigma_0^{tot}$"

        if fit.dB is True:
            label += " [dB]"

        ax.set_ylabel(label)
        ax.set_xlabel("$\\theta_0$")

        ax2 = fig.add_subplot(212)
        ax2.set_title("Estimated parameters")
        ax2.set_ylabel("Parameters")

        # plot fitted values
        # assign colors
        colordict = {key: f"C{i%10}" for i, key in enumerate(fit.res_dict.keys())}

        for key, val in fit.res_df.items():
            ax2.plot(val, alpha=1.0, label=key, color=colordict[key])

        if len(result_selection) < len(fit.data):
            for i, resid in enumerate(result_selection):
                ax2.text(
                    fit.index[resid],
                    ax2.get_ylim()[1] * 0.9,
                    resid,
                    bbox=dict(facecolor=f"C{i}", alpha=0.5),
                )
        ax2.legend(loc=1)

        fig.tight_layout()
        return fig

    def single_results(
        self,
        fit=None,
        fit_numbers=None,
        fit_indexes=None,
        hexbinQ=True,
        hexbinargs={},
        convertTodB=False,
        datetime_unit="h",
    ):
        """
        a function to investigate the quality of the individual fits


        Parameters
        ------------
        fit : rt1.rtfits.Fits object
              the fit-object to use
        fit_numbers : list
                      a list containing the position of the measurements
                      that should be plotted (starting from 0)
        fit_indexes : list
                      a list containing the index-values of the measurements
                      that should be plotted

        Other Parameters
        ------------------
        hexbinQ : bool (default = False)
                  indicator if a hexbin-plot should be underlayed
                  that shows the distribution of the datapoints
        hexbinargs : dict
                     a dict containing arguments to customize the hexbin-plot
        convertTodB : bool (default=False)
                      if set to true, the datasets will be converted to dB

        Returns
        --------------
        fig : matplotlib.figure
              the used matplotlib figure instance
        """
        if fit is None:
            fit = self.fit

        if fit_numbers is not None and fit_indexes is not None:
            assert False, "please provide EITHER fit_numbers OR fit_indexes!"
        elif fit_indexes is not None:
            # fit_numbers = np.where(fit.index.isin(fit_indexes))[0]
            fit_numbers = np.argmin(
                np.abs(
                    fit.fit_index
                    - np.expand_dims(
                        np.atleast_1d(np.array(fit_indexes, dtype=fit.index.dtype)),
                        -1,
                    )
                ),
                axis=1,
            )

        elif fit_numbers is None and fit_indexes is None:
            fit_numbers = [1]

        # function to generate colormap that fades between colors
        def CustomCmap(from_rgb, to_rgb):

            # from color r,g,b
            r1, g1, b1 = from_rgb

            # to color r,g,b
            r2, g2, b2 = to_rgb

            cdict = {
                "red": ((0, r1, r1), (1, r2, r2)),
                "green": ((0, g1, g1), (1, g2, g2)),
                "blue": ((0, b1, b1), (1, b2, b2)),
            }

            cmap = LinearSegmentedColormap("custom_cmap", cdict)
            return cmap

        estimates = fit._calc_model()
        indexsplits = fit._orig_index

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for m_i, m in enumerate(fit_numbers):

            if convertTodB is True:
                y = 10.0 * np.log10(estimates[m][~fit.mask[m]])
            else:
                y = estimates[m][~fit.mask[m]]

            # plot data
            nindexes = len(indexsplits[m])
            if nindexes > 1:
                mindate = np.datetime_as_string(indexsplits[m][0], unit=datetime_unit)
                maxdate = np.datetime_as_string(indexsplits[m][-1], unit=datetime_unit)
                if mindate == maxdate:
                    label = f"{mindate} [{nindexes}]"
                else:
                    label = f"({mindate} - {maxdate}) [{nindexes}]"
            else:
                label = np.datetime_as_string(indexsplits[m][0], unit=datetime_unit)

            xdata = np.rad2deg(fit.inc[m][~fit.mask[m]])

            if convertTodB is True:
                ydata = 10.0 * np.log10(fit.data[m][~fit.mask[m]])
            else:
                ydata = fit.data[m][~fit.mask[m]]

            # get color that will be applied to the next line drawn
            (dummy,) = ax.plot(xdata[0], ydata[0], ".", alpha=0.0)
            color = dummy.get_color()

            if hexbinQ is True:
                args = dict(gridsize=15, mincnt=1, linewidths=0.0, alpha=0.7)
                args.update(hexbinargs)

                # evaluate the hexbinplot once to get the maximum number of
                # datapoints within a single hexagonal (used for normalization)
                dummyargs = args.copy()
                dummyargs.update({"alpha": 0.0})
                hb = ax.hexbin(xdata, ydata, **dummyargs)

                # generate colormap that fades from white to the color
                # of the plotted data  (asdf.get_color())
                cmap = CustomCmap([1.00, 1.00, 1.00], plt.cm.colors.hex2color(color))
                # setup correct normalizing instance
                norm = Normalize(vmin=0, vmax=hb.get_array().max())

                ax.hexbin(xdata, ydata, cmap=cmap, norm=norm, **args)

            # plot datapoints
            (asdf,) = ax.plot(
                xdata,
                ydata,
                ".",
                color=color,
                alpha=1.0,
                label=label,
                markersize=10,
            )

            # plot results
            iii = fit.inc[m][~fit.mask[m]]
            ax.plot(
                np.rad2deg(iii[np.argsort(iii)]),
                y[np.argsort(iii)],
                "-",
                color="w",
                linewidth=3,
            )

            ax.plot(
                np.rad2deg(iii[np.argsort(iii)]),
                y[np.argsort(iii)],
                "-",
                color=asdf.get_color(),
                linewidth=2,
                marker="x",
            )

        ax.set_xlabel("$\\theta_0$ [deg]")
        ax.set_ylabel("$\\sigma_0$ [dB]")

        ax.legend(title="# Measurement")

        return fig

    def intermediate_results(self, fit=None, params=None, cmaps=None):
        """
        a function to plot the intermediate-results
        (the data is only available if rtfits.performfit has been called with
        the argument intermediate_results=True!)

        Parameters
        -----------
        fit : rtfits object
              the rtfits-object containing the fit-results
        params : list
            a list of parameter-names that are intended to be plotted
            as timeseries.
        cmaps : list
            a list of the colormaps used to plot the parameter variations

        Returns
        -------
        f : matplotlib.figure
            the used matplotlib figure instance
        """

        if fit is None:
            fit = self.fit

        try:
            fit.intermediate_results
        except AttributeError:
            assert False, (
                "No intermediate results are found, you must run"
                + " performfit() with intermediate_results=True!"
            )

        if params is None:
            params = fit.res_dict.keys()

        # constant parameters
        constparams = [key for key in params if len(fit.res_dict[key]) == 1]
        # timeseries-parameters
        tsparams = [i for i in params if i not in constparams]

        if cmaps is None:
            cmaps = [
                "Reds",
                "Greens",
                "Blues",
                "Purples",
                "Oranges",
                "Greys",
                "YlOrBr",
                "YlOrRd",
                "OrRd",
                "PuRd",
                "RdPu",
                "BuPu",
                "GnBu",
                "PuBu",
                "YlGnBu",
                "PuBuGn",
                "BuGn",
                "YlGn",
            ]

        interparams = {}
        for i, valdict in enumerate(fit.intermediate_results["parameters"]):
            for key, val in valdict.items():
                if key in interparams:
                    interparams[key] += [[i, np.mean(val[0])]]
                else:
                    interparams[key] = [[i, np.mean(val[0])]]
        intererrs = {}
        for i, valdict in enumerate(fit.intermediate_results["residuals"]):
            for key, val in valdict.items():
                if key in intererrs:
                    intererrs[key] += [[i, np.mean(val)]]
                else:
                    intererrs[key] = [[i, np.mean(val)]]
        interjacs = {}
        for i, valdict in enumerate(fit.intermediate_results["jacobian"]):
            for key, val in valdict.items():
                if key in interjacs:
                    interjacs[key] += [[i, np.mean(val)]]
                else:
                    interjacs[key] = [[i, np.mean(val)]]

        for i in [interparams, intererrs, interjacs]:
            for key, val in i.items():
                i[key] = np.array(val).T

        interres_params = {}
        for key in params:
            interres_p = pd.concat(
                [
                    pd.DataFrame(valdict[key], fit.meandatetimes[key], columns=[i])
                    for i, valdict in enumerate(fit.intermediate_results["parameters"])
                ],
                axis=1,
            )
            interres_params[key] = interres_p

        f = plt.figure(figsize=(15, 10))
        f.subplots_adjust(top=0.98, left=0.05, right=0.95)
        gs = GridSpec(
            4,
            len(interjacs),
            # width_ratios=[1, 2],
            # height_ratios=[1, 2, 1]
        )
        axsm = plt.subplot(gs[0, :])
        axerr = plt.subplot(gs[1, :])
        paramaxes, jacaxes = [], []
        for i in range(len(interjacs)):
            paramaxes += [plt.subplot(gs[2, i])]
            jacaxes += [plt.subplot(gs[3, i])]

        smhandles, smlabels = [], []
        nparam = 0
        for [parameter, paramdf] in interres_params.items():
            # plot only temporally varying parameters as timeseries
            if parameter not in tsparams:
                continue
            cmap = plt.get_cmap(cmaps[nparam])

            for key, val in paramdf.items():
                axsm.plot(
                    val,
                    c=cmap((float(key) / len(paramdf.keys()))),
                    lw=0.25,
                    marker=".",
                    ms=2,
                )

            # add colorbar
            axcb = f.add_axes(
                [
                    axsm.get_position().x1 - 0.01 * (nparam + 1),
                    axsm.get_position().y0,
                    0.01,
                    axsm.get_position().y1 - axsm.get_position().y0,
                ]
            )

            cbbounds = [1] + list(np.arange(2, len(paramdf.keys()) + 1, 1))

            cb = mpl.colorbar.ColorbarBase(
                axcb,
                cmap=cmap,
                orientation="vertical",
                boundaries=[0] + cbbounds + [cbbounds[-1] + 1],
                spacing="proportional",
                norm=mpl.colors.BoundaryNorm(cbbounds, cmap.N),
            )
            axcb.text(
                0.51,
                0.5,
                parameter,
                rotation=90,
                fontsize=8,
                horizontalalignment="center",
                verticalalignment="center",
            )
            if nparam > 0:
                cb.set_ticks([])

            smhandles += [mpl.lines.Line2D([], [], color=cmap(0.9))]
            smlabels += [parameter]
            nparam += 1

        axsm.legend(handles=smhandles, labels=smlabels, loc="upper left")

        axsmbounds = list(axsm.get_position().bounds)
        axsmbounds[2] = axsmbounds[2] - 0.015 * nparam
        axsm.set_position(axsmbounds)

        for [pax, jax, [key, val]] in zip(paramaxes, jacaxes, interparams.items()):
            if key not in interjacs:
                continue

            if key in tsparams:
                label = f"{key} (mean)"
            else:
                label = key

            pax.plot(*val, label=label, marker=".", ms=3, lw=0.5)
            pax.legend(loc="upper center")
            jax.plot(
                interjacs[key][0],
                interjacs[key][1],
                label=label,
                marker=".",
                ms=3,
                lw=0.5,
            )
            jax.legend(loc="upper center")

        paramaxes[-1].text(
            1.03,
            0.5,
            "Parameter estimates",
            rotation=90,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
            transform=paramaxes[-1].transAxes,
        )
        jacaxes[-1].text(
            1.03,
            0.5,
            "Jacobi determinant",
            rotation=90,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
            transform=jacaxes[-1].transAxes,
        )

        for key, val in intererrs.items():
            if key == "abserr":
                axerr.semilogy(
                    val[0],
                    np.abs(val[1]),
                    label="absolute error",
                    marker=".",
                    ms=3,
                    lw=0.5,
                    c="r",
                )
                axerr.legend(ncol=5, loc="upper left")
            if key == "relerr":
                axrelerr = axerr.twinx()
                axrelerr.semilogy(
                    val[0],
                    np.abs(val[1]),
                    label="relative error",
                    marker=".",
                    ms=3,
                    lw=0.5,
                    c="g",
                )
                axrelerr.legend(ncol=5, loc="upper right")

        return f

    def printsig0analysis(
        self,
        fit=None,
        range1=1,
        range2=None,
        use_index="groups",
        printfullt_0=True,
        printfulldata=True,
        dB=True,
        sig0=True,
        printcomponents1=True,
        printcomponents2=True,
        printparamnames=None,
        printparamstyles=dict(),
    ):
        """
        A widget to analyze the results of a rt1.rtfits.Fits object.
        (In order to keep the widget responsive, a reference to the returns
        must be kept!)


        Parameters
        ----------
        fit : rt1.rtfits.Fits object
            The used fit-object.
        range1, range2 : int, optional
            The number of consecutive measurements considered by the
            first/second slider. The default is (1 / None).
        use_index : str, optional
            Select the partition of the index accessible via the sliders.
                - if 'groups', the dataset will be grouped with respect to
                  the parameter-dynamics
                - if 'dataset', each unique dataset-index is used

        printcomponents1, printcomponents2 : bool, optional
            Indicator if individual backscatter contributions (surface, volume,
            interaction) should be plotted or not. The default is (True, True).
        printfullt_0 : bool, optional
            Indicator if backscatter-contributions should be evaluated over the
            full incidence-angle range or just over the range coverd by the
            dataset. The default is True.
        printfulldata : bool, optional
            Indicator if the range of the whole dataset should be indicated by
            small dots or not. The default is True.
        dB : bool, optional
            Indicator if the backscatter-plot should be in linear-units or dB.
            The default is True.
        sig0 : bool, optional
            Indicator if the backscatter-plot should represent sigma0 or
            intensity. The default is True.
        printparamnames : list of str, optional
            A list of strings corresponding to the parameter-names whose
            results should be added to the plot. The default is None.
        printparamstyles : dict
            A dict with style-options (updating the default ones) passed to the
            plot for each parameter selected in printparamnames via:

                >>> plt.plot(x, y, **printparamstyles['parameter-name'])

        Returns
        -------
        list
            a reference to the matplotlib-objects corresponding to:
            [figure, first_slider, (second_slider)].

        """

        if fit is None:
            fit = self.fit

        if printparamnames is None:
            printparamnames = fit.res_dict.keys()

        # gridspec for incidence-angle and parameter plots
        gs = GridSpec(3, 2, height_ratios=[0.65, 0.25, 0.1])
        gs.update(top=0.98, left=0.1, right=0.9, bottom=0.025)
        # sub-gridspec for sliders
        gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :])

        f = plt.figure(figsize=(10, 6))
        # add plot-axes
        ax = f.add_subplot(gs[0, 0])
        ax1 = f.add_subplot(gs[0, 1])
        ax2 = f.add_subplot(gs[1, :])

        # add slider axes
        slider_ax = f.add_subplot(gs2[0])
        if range2 is not None:
            slider_bx = f.add_subplot(gs2[1])

        ax.grid()
        ax1.grid()

        if use_index == "dataset":
            sig0_vals = fit.calc_model(return_components=True)
            sig0_vals["data"] = fit.dataset.sig
            sig0_vals["incs"] = fit.dataset.inc
            sig0_vals = sig0_vals.groupby(level=0).agg(list).to_dict(orient="list")
            sig0_vals = {
                key: rectangularize(val, return_masked=True)
                for key, val in sig0_vals.items()
            }
            sig0_vals["indexes"] = pd.to_datetime(fit.index)
        elif use_index == "groups":
            # calculate backscatter values and ensure correct index-order
            # ( in case a unordered dyn-dict is used)
            mask = fit.mask

            sig0_vals = fit._calc_model(return_components=True)
            # apply mask and convert to pandas dataframe
            sig0_vals = [np.ma.masked_array(con, mask) for con in sig0_vals]
            sig0_vals = dict(zip(["tot", "surf", "vol", "inter"], sig0_vals))

            sig0_vals["data"] = np.ma.masked_array(fit.data, mask)
            sig0_vals["incs"] = np.ma.masked_array(fit.inc, mask)
            sig0_vals["indexes"] = pd.to_datetime(fit.meandatetimes_group)
        # convert to sig0 and dB if necessary
        sig0_vals_I_linear = dict()
        for key in ["tot", "surf", "vol", "inter", "data"]:
            if key not in sig0_vals:
                continue
            sig0_vals[key] = dBsig0convert(
                sig0_vals[key],
                sig0_vals["incs"],
                dB=dB,
                sig0=sig0,
                fitdB=fit.dB,
                fitsig0=fit.sig0,
            )
            sig0_vals_I_linear[key] = dBsig0convert(
                sig0_vals[key],
                sig0_vals["incs"],
                dB=False,
                sig0=False,
                fitdB=dB,
                fitsig0=sig0,
            )
        if printfullt_0 is True:
            inc = np.deg2rad(np.arange(1, 89, 1))
            if use_index == "dataset":
                newsig0_vals = fit.calc(
                    param=fit.res_df,
                    inc=inc,
                    fixed_param=fit.dataset[fit.fixed_dict.keys()],
                    return_components=True,
                )
            elif use_index == "groups":
                if len(fit.fixed_dict) > 0:
                    # get the average value of the fixed-parameters
                    # for each group
                    usefixedparams = (
                        fit.dataset[fit.fixed_dict.keys()]
                        .groupby(fit._groupindex, sort=False)
                        .mean()
                        .set_index(pd.to_datetime(fit.meandatetimes_group))
                    )
                else:
                    usefixedparams = dict()
                newsig0_vals = fit.calc(
                    param=fit.res_df_group,
                    inc=inc,
                    fixed_param=usefixedparams,
                    return_components=True,
                )

            newsig0_vals = dict(zip(["tot", "surf", "vol", "inter"], newsig0_vals))

            newsig0_vals["incs"] = np.broadcast_to(inc, newsig0_vals["tot"].shape)

            newsig0_vals_I_linear = dict()
            for key in ["tot", "surf", "vol", "inter"]:
                if key not in newsig0_vals:
                    continue
                newsig0_vals[key] = dBsig0convert(
                    newsig0_vals[key],
                    newsig0_vals["incs"],
                    fitdB=fit.dB,
                    fitsig0=fit.sig0,
                    dB=dB,
                    sig0=sig0,
                )
                newsig0_vals_I_linear[key] = dBsig0convert(
                    newsig0_vals[key],
                    newsig0_vals["incs"],
                    dB=False,
                    sig0=False,
                    fitdB=dB,
                    fitsig0=sig0,
                )

            ax.set_xlim(
                [
                    -2 + np.rad2deg(np.ma.min(newsig0_vals["incs"])),
                    2 + np.rad2deg(np.ma.max(newsig0_vals["incs"])),
                ]
            )
            ax.set_ylim(
                [
                    np.min(
                        [
                            np.ma.min(newsig0_vals["tot"]),
                            np.ma.min(sig0_vals["data"]),
                        ]
                    ),
                    np.max(
                        [
                            np.ma.max(newsig0_vals["tot"]),
                            np.ma.max(sig0_vals["data"]),
                        ]
                    ),
                ]
            )
        else:
            ax.set_xlim(
                [
                    -2 + np.rad2deg(np.ma.min(sig0_vals["incs"][0])),
                    2 + np.rad2deg(np.ma.max(sig0_vals["incs"][0])),
                ]
            )
            ax.set_ylim(
                [
                    np.min(
                        [
                            np.ma.min(sig0_vals["tot"]),
                            np.ma.min(sig0_vals["data"]),
                        ]
                    ),
                    np.max(
                        [
                            np.ma.max(sig0_vals["tot"]),
                            np.ma.max(sig0_vals["data"]),
                        ]
                    ),
                ]
            )

        # ensure sort-order
        if use_index == "dataset":
            inc_sortp = np.argsort(fit.index)
        elif use_index == "groups":
            inc_sortp = np.argsort(fit.meandatetimes_group)

        for key, val in sig0_vals.items():
            sig0_vals[key] = val[inc_sortp]
        for key, val in newsig0_vals.items():
            newsig0_vals[key] = val[inc_sortp]
        for key, val in newsig0_vals_I_linear.items():
            newsig0_vals_I_linear[key] = val[inc_sortp]

        # print full data points in the background
        if printfulldata is True:
            ax.plot(
                np.rad2deg(sig0_vals["incs"]),
                sig0_vals["data"],
                lw=0.0,
                marker=".",
                ms=0.5,
                color="k",
                alpha=0.5,
            )

        # plot parameters as specified in printparamnames
        axparamplot = ax2
        handles, labels = [], []
        i = 0
        pos = 1 - len(printparamnames) // 2 * 0.035
        for key in printparamnames:
            try:
                style = dict(color="C" + str(i), marker=".", lw=0.75)
                style.update(printparamstyles.get(key, dict()))

                if i > 0:
                    axparamplot = ax2.twinx()
                    axparamplot.tick_params(
                        axis="y", which="both", labelsize=5, length=2
                    )
                    pos += 0.035
                    axparamplot.spines["right"].set_position(("axes", pos))
                    axparamplot.tick_params(
                        axis="y", which="both", labelsize=5, length=2
                    )
                if key in fit.res_df:
                    (l,) = axparamplot.plot(
                        fit.res_df.sort_index()[key], label=key, **style
                    )
                elif key in fit.dataset:
                    (l,) = axparamplot.plot(
                        fit.dataset.sort_index()[key], label=key, **style
                    )
                else:
                    print(
                        f'parameter "{key}" not found in "fit.res_df"',
                        'and "fit.dataset"',
                    )
                    continue

                # add handles and labels to legend
                handles += axparamplot.get_legend_handles_labels()[0]
                labels += axparamplot.get_legend_handles_labels()[1]

                # change color of axis to fit color of lines
                axparamplot.yaxis.label.set_color(l.get_color())
                axparamplot.tick_params(axis="y", colors=l.get_color())
                # shift twin-axes if necessary
                i += 1
            except:
                pass
        axparamplot.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            ncol=len(printparamnames),
        )

        ax2.xaxis.set_minor_locator(mpl.dates.MonthLocator())
        ax2.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%m"))
        ax2.xaxis.set_major_locator(mpl.dates.YearLocator())
        ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter("\n%Y"))

        indicator_bounds = [ax2.get_ylim()[0] * 1.05, ax2.get_ylim()[1] * 0.95]

        # -----------------------------------------------------------
        # plot lines
        def plotlines(
            dayrange,
            printcomponents,
            printfullt_0,
            styledict_dict,
            styledict_fullt0_dict,
        ):

            incs = np.rad2deg(sig0_vals["incs"])

            lines = []
            for day in np.arange(0, dayrange, 1):
                lines += ax.plot(
                    incs[day], sig0_vals["tot"][day], **styledict_dict["tot"]
                )
                if printcomponents:
                    lines += ax.plot(
                        incs[day],
                        sig0_vals["surf"][day],
                        **styledict_dict["surf"],
                    )
                    lines += ax.plot(
                        incs[day],
                        sig0_vals["vol"][day],
                        **styledict_dict["vol"],
                    )
                    if fit.int_Q is True:
                        lines += ax.plot(
                            incs[day],
                            sig0_vals["inter"][day],
                            **styledict_dict["inter"],
                        )

                lines += ax.plot(
                    incs[day], sig0_vals["data"][day], **styledict_dict["data"]
                )
                lines += ax2.plot(
                    [sig0_vals["indexes"][day]] * 2,
                    indicator_bounds,
                    **styledict_dict["indicator"],
                )

            lines_frac = []
            for day in np.arange(0, dayrange, 1):
                lintot = sig0_vals_I_linear["tot"][day]
                linsurf = sig0_vals_I_linear["surf"][day]
                linvol = sig0_vals_I_linear["vol"][day]

                if fit.int_Q is True:
                    lininter = sig0_vals_I_linear["inter"][day]

                lines_frac += ax1.plot(
                    incs[day], linsurf / lintot, **styledict_dict["surf"]
                )
                lines_frac += ax1.plot(
                    incs[day], linvol / lintot, **styledict_dict["vol"]
                )
                if fit.int_Q is True:
                    lines_frac += ax1.plot(
                        incs[day], lininter / lintot, **styledict_dict["inter"]
                    )

            # plot full-incidence-angle lines
            linesfull = []
            lines_frac_full = []
            if printfullt_0 is True:
                newincs = np.rad2deg(newsig0_vals["incs"])

                for day in np.arange(0, dayrange, 1):

                    sortp = np.argsort(newincs[day])

                    linesfull += ax.plot(
                        newincs[day][sortp],
                        newsig0_vals["tot"][day][sortp],
                        **styledict_fullt0_dict["tot"],
                    )
                    if printcomponents:
                        linesfull += ax.plot(
                            newincs[day][sortp],
                            newsig0_vals["surf"][day][sortp],
                            **styledict_fullt0_dict["surf"],
                        )
                        linesfull += ax.plot(
                            newincs[day][sortp],
                            newsig0_vals["vol"][day][sortp],
                            **styledict_fullt0_dict["vol"],
                        )
                        if fit.int_Q is True:
                            linesfull += ax.plot(
                                newincs[day][sortp],
                                newsig0_vals["inter"][day][sortp],
                                **styledict_fullt0_dict["inter"],
                            )

                    lintot = newsig0_vals_I_linear["tot"][day]
                    linsurf = newsig0_vals_I_linear["surf"][day]
                    linvol = newsig0_vals_I_linear["vol"][day]
                    if fit.int_Q is True:
                        lininter = newsig0_vals_I_linear["inter"][day]

                    lines_frac_full += ax1.plot(
                        newincs[day][sortp],
                        (linsurf / lintot)[sortp],
                        **styledict_fullt0_dict["surf"],
                    )
                    lines_frac_full += ax1.plot(
                        newincs[day][sortp],
                        (linvol / lintot)[sortp],
                        **styledict_fullt0_dict["vol"],
                    )
                    if fit.int_Q is True:
                        lines_frac_full += ax1.plot(
                            newincs[day][sortp],
                            (lininter / lintot)[sortp],
                            **styledict_fullt0_dict["inter"],
                        )

            # add unique legend entries
            ha, la = ax.get_legend_handles_labels()
            unila, wherela = np.unique(la, return_index=True)
            # sort legend entries
            sort_order = dict(
                data_1=0, data_2=1, total=2, surface=3, volume=4, interaction=5
            )

            halas = [
                [ha, la] for ha, la in zip(np.array(la)[wherela], np.array(ha)[wherela])
            ]
            halas.sort(key=lambda val: sort_order[val[0]])
            ax.legend(handles=[i[1] for i in halas], labels=[i[0] for i in halas])

            return lines, lines_frac, linesfull, lines_frac_full

        # plot first set of lines
        styletot = {
            "lw": 1,
            "marker": "o",
            "ms": 3,
            "color": "k",
            "label": "total",
        }
        stylevol = {
            "lw": 1,
            "marker": "o",
            "ms": 3,
            "color": "g",
            "markerfacecolor": "gray",
            "label": "volume",
        }
        stylesurf = {
            "lw": 1,
            "marker": "o",
            "ms": 3,
            "color": "y",
            "markerfacecolor": "gray",
            "label": "surface",
        }
        styleinter = {
            "lw": 1,
            "marker": "o",
            "ms": 3,
            "color": "c",
            "markerfacecolor": "gray",
            "label": "interaction",
        }
        styledata = {
            "lw": 0,
            "marker": "s",
            "ms": 5,
            "color": "k",
            "markerfacecolor": "gray",
            "label": "data_1",
        }
        styleindicator = {"c": "k"}

        stylefullt0tot = {"lw": 0.25, "color": "k"}
        stylefullt0vol = {"lw": 0.25, "color": "g"}
        stylefullt0surf = {"lw": 0.25, "color": "y"}
        stylefullt0inter = {"lw": 0.25, "color": "c"}

        styledict_dict = dict(
            zip(
                ["tot", "surf", "vol", "inter", "data", "indicator"],
                [
                    styletot,
                    stylesurf,
                    stylevol,
                    styleinter,
                    styledata,
                    styleindicator,
                ],
            )
        )
        styledict_fullt0_dict = dict(
            zip(
                ["tot", "surf", "vol", "inter"],
                [
                    stylefullt0tot,
                    stylefullt0surf,
                    stylefullt0vol,
                    stylefullt0inter,
                ],
            )
        )

        lines, lines_frac, linesfull, lines_frac_full = plotlines(
            range1,
            printcomponents1,
            printfullt_0,
            styledict_dict,
            styledict_fullt0_dict,
        )

        if range2 is not None:
            # plot second set of lines
            styletot = {
                "lw": 1,
                "marker": "o",
                "ms": 3,
                "color": "r",
                "dashes": [5, 5],
                "markerfacecolor": "none",
            }
            stylevol = {
                "lw": 1,
                "marker": "o",
                "ms": 3,
                "color": "g",
                "dashes": [5, 5],
                "markerfacecolor": "r",
            }
            stylesurf = {
                "lw": 1,
                "marker": "o",
                "ms": 3,
                "color": "y",
                "dashes": [5, 5],
                "markerfacecolor": "r",
            }
            styleinter = {
                "lw": 1,
                "marker": "o",
                "ms": 3,
                "color": "c",
                "dashes": [5, 5],
                "markerfacecolor": "r",
            }
            styledata = {
                "lw": 0,
                "marker": "s",
                "ms": 5,
                "color": "r",
                "markerfacecolor": "none",
                "label": "data_2",
            }
            styleindicator = {"c": "gray", "ls": "--"}

            stylefullt0tot = {"lw": 0.25, "color": "r", "dashes": [5, 5]}
            stylefullt0vol = {"lw": 0.25, "color": "g", "dashes": [5, 5]}
            stylefullt0surf = {"lw": 0.25, "color": "y", "dashes": [5, 5]}
            stylefullt0inter = {"lw": 0.25, "color": "c", "dashes": [5, 5]}

            styledict_dict = dict(
                zip(
                    ["tot", "surf", "vol", "inter", "data", "indicator"],
                    [
                        styletot,
                        stylesurf,
                        stylevol,
                        styleinter,
                        styledata,
                        styleindicator,
                    ],
                )
            )
            styledict_fullt0_dict = dict(
                zip(
                    ["tot", "surf", "vol", "inter"],
                    [
                        stylefullt0tot,
                        stylefullt0surf,
                        stylefullt0vol,
                        stylefullt0inter,
                    ],
                )
            )

            lines2, lines_frac2, linesfull2, lines_frac_full2 = plotlines(
                range2,
                printcomponents2,
                printfullt_0,
                styledict_dict,
                styledict_fullt0_dict,
            )

        # define function to update lines based on slider-input
        def animate(
            day0,
            lines,
            linesfull,
            lines_frac,
            lines_frac_full,
            dayrange,
            printcomponents,
            label,
        ):

            day0 = int(day0)

            label.set_position([day0, label.get_position()[1]])
            if dayrange == 1:
                label.set_text(sig0_vals["indexes"][day0].strftime("%d. %b %Y %H:%M"))
            elif dayrange > 1:

                lday_0 = sig0_vals["indexes"][day0].strftime("%d. %b %Y %H:%M")
                lday_1 = sig0_vals["indexes"][day0 + dayrange - 1].strftime(
                    "%d. %b %Y %H:%M"
                )
                label.set_text(f"{lday_0} - {lday_1}")

            maxdays = len(sig0_vals["incs"])
            i = 0
            for day in np.arange(day0, day0 + dayrange, 1):
                if day >= maxdays:
                    continue
                lines[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                lines[i].set_ydata(sig0_vals["tot"][day])
                i += 1
                if printcomponents:
                    lines[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                    lines[i].set_ydata(sig0_vals["surf"][day])
                    i += 1
                    lines[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                    lines[i].set_ydata(sig0_vals["vol"][day])
                    if fit.int_Q is True:
                        i += 1
                        lines[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                        lines[i].set_ydata(sig0_vals["inter"][day])
                    i += 1

                # update data measurements
                lines[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                lines[i].set_ydata(sig0_vals["data"][day])
                i += 1
                # update day-indicator line
                lines[i].set_xdata([sig0_vals["indexes"][day]] * 2)
                i += 1

            i = 0
            for day in np.arange(day0, day0 + dayrange, 1):
                if day >= maxdays:
                    continue
                lintot = sig0_vals_I_linear["tot"][day]
                linsurf = sig0_vals_I_linear["surf"][day]
                linvol = sig0_vals_I_linear["vol"][day]
                if fit.int_Q is True:
                    lininter = sig0_vals_I_linear["inter"][day]

                lines_frac[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                lines_frac[i].set_ydata(linsurf / lintot)
                i += 1
                lines_frac[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                lines_frac[i].set_ydata(linvol / lintot)
                if fit.int_Q is True:
                    i += 1
                    lines_frac[i].set_xdata(np.rad2deg(sig0_vals["incs"][day]))
                    lines_frac[i].set_ydata(lininter / lintot)
                i += 1

            if printfullt_0 is True:
                i = 0
                for day in np.arange(day0, day0 + dayrange, 1):
                    if day >= maxdays:
                        continue
                    day_inc_new = np.rad2deg(newsig0_vals["incs"][day])
                    sortp = np.argsort(day_inc_new)

                    linesfull[i].set_xdata(day_inc_new[sortp])
                    linesfull[i].set_ydata(newsig0_vals["tot"][day][sortp])
                    i += 1
                    if printcomponents:
                        linesfull[i].set_xdata(day_inc_new[sortp])
                        linesfull[i].set_ydata(newsig0_vals["surf"][day][sortp])
                        i += 1
                        linesfull[i].set_xdata(day_inc_new[sortp])
                        linesfull[i].set_ydata(newsig0_vals["vol"][day][sortp])
                        if fit.int_Q is True:
                            i += 1
                            linesfull[i].set_xdata(day_inc_new[sortp])
                            linesfull[i].set_ydata(newsig0_vals["inter"][day][sortp])
                        i += 1
                i = 0
                for day in np.arange(day0, day0 + dayrange, 1):
                    if day >= maxdays:
                        continue
                    day_inc_new = np.rad2deg(newsig0_vals["incs"][day])
                    sortp = np.argsort(day_inc_new)

                    lintot = newsig0_vals_I_linear["tot"][day]
                    linsurf = newsig0_vals_I_linear["surf"][day]
                    linvol = newsig0_vals_I_linear["vol"][day]
                    if fit.int_Q is True:
                        lininter = newsig0_vals_I_linear["inter"][day]

                    lines_frac_full[i].set_xdata(day_inc_new[sortp])
                    lines_frac_full[i].set_ydata((linsurf / lintot)[sortp])
                    i += 1
                    lines_frac_full[i].set_xdata(day_inc_new[sortp])
                    lines_frac_full[i].set_ydata((linvol / lintot)[sortp])
                    if fit.int_Q is True:
                        i += 1
                        lines_frac_full[i].set_xdata(day_inc_new[sortp])
                        lines_frac_full[i].set_ydata((lininter / lintot)[sortp])
                    i += 1

            return lines

        # define function to update slider-range based on zoom
        def updatesliderboundary(evt, slider):
            indexes = sig0_vals["indexes"]
            # Get the range for the new area
            xstart, ystart, xdelta, ydelta = ax2.viewLim.bounds
            xend = xstart + xdelta

            # convert to datetime-objects and ensure that they are in the
            # same time-zone as the sig0_vals indexes
            xend = mpl.dates.num2date(xend).replace(tzinfo=sig0_vals["indexes"].tzinfo)
            xstart = mpl.dates.num2date(xstart).replace(
                tzinfo=sig0_vals["indexes"].tzinfo
            )

            zoomindex = np.where(np.logical_and(indexes > xstart, indexes < xend))[0]
            slider.valmin = zoomindex[0] - 1
            slider.valmax = zoomindex[-1] + 1

            slider.ax.set_xlim(slider.valmin, slider.valmax)

        # create the slider
        a_slider = Slider(
            slider_ax,  # axes object for the slider
            "solid lines",  # name of the slider parameter
            0,  # minimal value of parameter
            len(sig0_vals["tot"]) - 1,  # maximal value of parameter
            valinit=0,  # initial value of parameter
            valfmt="%i",  # print slider-value as integer
            valstep=1,
            closedmax=True,
        )
        a_slider.valtext.set_visible(False)

        slider_ax.set_xticks(
            np.arange(
                slider_ax.get_xlim()[0] - 1,
                slider_ax.get_xlim()[1] + 1,
                1,
                dtype=int,
            )
        )
        slider_ax.tick_params(bottom=False, labelbottom=False)
        slider_ax.grid()

        label = slider_ax.text(
            0,
            0.5,
            f"{sig0_vals['indexes'][0].strftime('%d. %b %Y %H:%M')}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            bbox=dict(facecolor="w", alpha=0.75, boxstyle="round,pad=.2"),
        )

        # set slider to call animate function when changed
        a_slider.on_changed(
            partial(
                animate,
                lines=lines,
                linesfull=linesfull,
                lines_frac=lines_frac,
                lines_frac_full=lines_frac_full,
                dayrange=range1,
                printcomponents=printcomponents1,
                label=label,
            )
        )

        # update slider boundary with respect to zoom of second plot
        ax2.callbacks.connect(
            "xlim_changed", partial(updatesliderboundary, slider=a_slider)
        )
        if range2 is not None:

            # here we create the slider
            b_slider = Slider(
                slider_bx,  # axes object for the slider
                "dashed lines",  # name of the slider parameter
                0,  # minimal value of parameter
                len(sig0_vals["tot"]) - 1,  # maximal value of parameter
                valinit=0,  # initial value of parameter
                valfmt="%i",
                valstep=1,
                closedmax=True,
            )

            b_slider.valtext.set_visible(False)

            slider_bx.set_xticks(
                np.arange(
                    slider_bx.get_xlim()[0] - 1,
                    slider_bx.get_xlim()[1] + 1,
                    1,
                    dtype=int,
                )
            )
            slider_bx.tick_params(bottom=False, labelbottom=False)
            slider_bx.grid()

            label2 = slider_bx.text(
                0,
                0.5,
                f"{sig0_vals['indexes'][0].strftime('%d. %b %Y %H:%M')}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=8,
                bbox=dict(facecolor="w", alpha=0.75, boxstyle="round,pad=.2"),
            )

            b_slider.on_changed(
                partial(
                    animate,
                    lines=lines2,
                    linesfull=linesfull2,
                    lines_frac=lines_frac2,
                    lines_frac_full=lines_frac_full2,
                    dayrange=range2,
                    printcomponents=printcomponents2,
                    label=label2,
                )
            )

            ax2.callbacks.connect(
                "xlim_changed", partial(updatesliderboundary, slider=b_slider)
            )

        # a reference to the sliders must be returned in order to
        # remain interactive
        if range2 is not None:
            return f, a_slider, b_slider
        else:
            return f, a_slider

    def analyzemodel(
        self,
        fit=None,
        defdict=None,
        inc=None,
        labels=None,
        dB=True,
        sig0=True,
        int_Q=None,
        fillcomponents=True,
    ):
        """
        Analyze the range of backscatter for a given model-configuration
        based on the defined parameter-ranges

        Parameters
        ----------
        fit : rt1.rtfits.Fits, optional
            the fits-object to use
            The default is None.
        defdict : dict, optional
            A defdict used to define the rt1-configuration.
            (see rt1.rtfits.Fits for details). If none, the defdict provided
            in the fits-object will be used. The default is None.
        inc : array-like, optional
            The incidence-angles to be used in the calculation.
            If None,  `np.deg2rad(np.linspace(1, 89, 100))` is used
            The default is None
        labels : dict, optional
            A dict with labels that will be used to replace the parameter-names.
            (e.g. {'parameter' : 'parameter_label', ....})
            The default is {}.
        dB : bool, optional
            Indicator if the plot is in linear-units or dB. The default is True.
        sig0 : bool, optional
            Indicator if intensity- or sigma-0 values should be plotted.
            ( sig0 = 4 pi cos(theta) I ). The default is True.
        int_Q : bool, optional
            Indicator if the interaction-term should be evaluated.
            The default is False.
        fillcomponents : bool, optional
            Indicator if the variabilities of the components should
            be indicated by fillings. The default is True.

        Returns
        -------
        f, slider, buttons
            the matplotlib figure, slider and button instances

        """

        if fit is None:
            fit = self.fit

        res_dict = getattr(fit, "res_dict", None)

        if defdict is None:
            defdict = fit.defdict

        if inc is None:
            inc = np.deg2rad(np.linspace(1, 89, 100))

        if labels is None:
            labels = dict()

        if int_Q is None:
            int_Q = fit.int_Q

        # get parameter ranges from defdict and fit
        minparams, maxparams, startparams, fixparams = {}, {}, {}, {}
        for key, val in defdict.items():
            if val[0] is True:
                minparams[key] = val[3][0][0]
                maxparams[key] = val[3][1][0]
                # try to use fitted-values as start values for the parameters
                if res_dict is not None and key in res_dict:
                    startparams[key] = np.mean(res_dict[key][0])
                else:
                    startparams[key] = val[1]
            if val[0] is False:
                if isinstance(val[1], (int, float)):
                    # don't add constants to startparams (they are
                    # directly inserted into the functions)
                    # startparams[key] = val[1]
                    fixparams[key] = val[1]
                elif val[1] == "auxiliary":
                    assert key in fit.dataset or key in fit.fixed_dict, (
                        f"auxiliary dataset for {key} "
                        + "not found in fit.dataset or "
                        + "fit.fixed_dict"
                    )
                    if key in fit.dataset:
                        minparams[key] = fit.dataset[key].min()
                        maxparams[key] = fit.dataset[key].max()
                        startparams[key] = fit.dataset[key].mean()

                    elif key in fit.fixed_dict:
                        minparams[key] = fit.fixed_dict[key].min()
                        maxparams[key] = fit.fixed_dict[key].max()
                        startparams[key] = fit.fixed_dict[key].mean()

        if "bsf" not in defdict:
            startparams["bsf"] = fit.R.bsf
            fixparams["bsf"] = fit.R.bsf

        modelresult = dict(
            zip(["tot", "surf", "vol", "inter"], fit.calc(startparams, inc=inc))
        )
        # convert to sig0 and dB if required
        for key, val in modelresult.items():
            modelresult[key] = dBsig0convert(val[0], inc, dB, sig0, fit.dB, fit.sig0)

        f = plt.figure(figsize=(12, 9))
        f.subplots_adjust(top=0.93, right=0.98, left=0.07)
        # generate figure grid and populate with axes
        gs = GridSpec(
            1 + len(minparams) // 2,
            1 + 3,
            height_ratios=[8] + [1] * (len(minparams) // 2),
            width_ratios=[0.75, 1, 1, 1],
        )
        gs.update(wspace=0.3)

        gsslider = GridSpec(
            1 + len(minparams) // 2,
            1 + 3,
            height_ratios=[8] + [1] * (len(minparams) // 2),
            width_ratios=[0.75, 1, 1, 1],
        )
        gsslider.update(wspace=0.3, bottom=0.05)

        gsbutton = GridSpec(
            1 + len(minparams) // 2,
            1 + 3,
            height_ratios=[8] + [1] * (len(minparams) // 2),
            width_ratios=[0.75, 1, 1, 1],
        )
        gsbutton.update(hspace=0.75, wspace=0.1, bottom=0.05)

        ax = f.add_subplot(gs[0, 0:])

        paramaxes = {}
        col = 0
        for i, key in enumerate(minparams):
            if i % 3 == 0:
                col += 1
            paramaxes[key] = f.add_subplot(gsslider[col, 1 + i % 3])

        buttonax = f.add_subplot(gsbutton[1:, 0])
        # hide frame of button-axes
        buttonax.axis("off")
        # add values of fixed parameters
        if len(fixparams) > 0:
            ax.text(
                0.01,
                0.98,
                "fixed parameters:\n"
                + "".join(
                    [f"{key}={round(val, 5)}   " for key, val in fixparams.items()]
                ),
            )

        # overplot data used in fit
        try:
            ax.plot(
                fit.dataset.inc,
                dBsig0convert(
                    fit.dataset.sig,
                    fit.dataset.inc,
                    dB,
                    sig0,
                    fit.dB,
                    fit.sig0,
                ),
                zorder=0,
                marker=".",
                alpha=0.5,
                lw=0,
                markerfacecolor="none",
                markeredgecolor="k",
            )

        except:
            pass

        # plot initial curves
        (ltot,) = ax.plot(inc, modelresult["tot"], "k", label="total contribution")

        (lsurf,) = ax.plot(inc, modelresult["surf"], "b", label="surface contribution")

        (lvol,) = ax.plot(inc, modelresult["vol"], "g", label="volume contribution")

        if int_Q is True:
            (lint,) = ax.plot(
                inc,
                modelresult["inter"],
                "y",
                label="interaction contribution",
            )

        if dB is True:
            ax.set_ylim(-35, 5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, y: f"{np.rad2deg(x):.1f}")
        )
        # a legend for the lines
        leg0 = ax.legend(ncol=4, bbox_to_anchor=(0.5, 1.1), loc="upper center")
        # add the line-legend as individual artist
        ax.add_artist(leg0)

        if dB is True and sig0 is True:
            ax.set_ylabel(r"$\sigma_0$ [dB]")
        if dB is True and sig0 is False:
            ax.set_ylabel(r"$I/I_0$ [dB]")
        if dB is False and sig0 is True:
            ax.set_ylabel(r"$\sigma_0$")
        if dB is False and sig0 is False:
            ax.set_ylabel(r"$I/I_0$")

        ax.set_xlabel(r"$\theta_0$ [deg]")

        # create the slider for the parameter
        paramslider = {}
        buttonlabels = []
        for key, val in minparams.items():
            # replace label of key with provided label
            if key in labels:
                keylabel = labels[key]
            else:
                keylabel = key
            buttonlabels += [keylabel]

            startval = startparams[key]

            paramslider[key] = Slider(
                paramaxes[key],  # axes object for the slider
                keylabel,  # name of the slider
                minparams[key],  # minimal value
                maxparams[key],  # maximal value
                startval,  # initial value
                # valfmt="%i"            # slider-value as integer
                color="gray",
            )
            paramslider[key].label.set_position([0.05, 0.5])
            paramslider[key].label.set_bbox(
                dict(boxstyle="round,pad=0.5", facecolor="w")
            )
            paramslider[key].label.set_horizontalalignment("left")
            paramslider[key].valtext.set_position([0.8, 0.5])

        buttons = CheckButtons(buttonax, buttonlabels, [False for i in buttonlabels])

        params = startparams.copy()
        # define function to update lines based on slider-input
        def animate(value, key):
            # params = copy.deepcopy(startparams)

            params[key] = value
            modelresult = dict(
                zip(["tot", "surf", "vol", "inter"], fit.calc(params, inc=inc))
            )
            # convert to sig0 and dB if required
            for key, val in modelresult.items():
                modelresult[key] = dBsig0convert(
                    val[0], inc, dB, sig0, fit.dB, fit.sig0
                )

            # update the data
            ltot.set_ydata(modelresult["tot"].T)
            lsurf.set_ydata(modelresult["surf"].T)
            lvol.set_ydata(modelresult["vol"].T)
            if int_Q is True:
                lint.set_ydata(modelresult["inter"].T)

            # poverprint boundaries
            hatches = [r"//", r"\\ ", "+", "oo", "--", ".."]
            colors = ["C" + str(i) for i in range(10)]
            ax.collections.clear()
            legendhandles = []
            for i, [key_i, key_Q] in enumerate(printvariationQ.items()):
                # replace label of key_i with provided label
                if key_i in labels:
                    keylabel = labels[key_i]
                else:
                    keylabel = key_i

                # reset color of text-backtround
                paramslider[key_i].label.get_bbox_patch().set_facecolor("w")
                if key_Q is True:
                    # set color of text-background to hatch-color
                    # paramslider[key_i].label.set_color(colors[i%len(colors)])
                    paramslider[key_i].label.get_bbox_patch().set_facecolor(
                        colors[i % len(colors)]
                    )

                    fillparams = params.copy()
                    fillparams[key_i] = minparams[key_i]

                    # don't use bsf=1 since no vegetation-term would be present
                    if fillparams.get("bsf", 0.0) == 1.0:
                        fillparams["bsf"] = 0.999

                    modelresultmin = dict(
                        zip(
                            ["tot", "surf", "vol", "inter"],
                            fit.calc(fillparams, inc=inc),
                        )
                    )
                    # convert to sig0 and dB if required
                    for key, val in modelresultmin.items():
                        modelresultmin[key] = dBsig0convert(
                            val[0], inc, dB, sig0, fit.dB, fit.sig0
                        )

                    fillparams[key_i] = maxparams[key_i]

                    # don't use bsf=1 since no vegetation-term would be present
                    if fillparams.get("bsf", 0.0) == 1.0:
                        fillparams["bsf"] = 0.999

                    modelresultmax = dict(
                        zip(
                            ["tot", "surf", "vol", "inter"],
                            fit.calc(fillparams, inc=inc),
                        )
                    )
                    # convert to sig0 and dB if required
                    for key, val in modelresultmax.items():
                        modelresultmax[key] = dBsig0convert(
                            val[0], inc, dB, sig0, fit.dB, fit.sig0
                        )

                    legendhandles += [
                        ax.fill_between(
                            inc,
                            modelresultmax["tot"],
                            modelresultmin["tot"],
                            facecolor="none",
                            hatch=hatches[i % len(hatches)],
                            edgecolor=colors[i % len(colors)],
                            label="total variability (" + keylabel + ")",
                        )
                    ]

                    if fillcomponents is True:

                        legendhandles += [
                            ax.fill_between(
                                inc,
                                modelresultmax["surf"],
                                modelresultmin["surf"],
                                color="b",
                                alpha=0.1,
                                label="surf variability (" + keylabel + ")",
                            )
                        ]

                        legendhandles += [
                            ax.fill_between(
                                inc,
                                modelresultmax["vol"],
                                modelresultmin["vol"],
                                color="g",
                                alpha=0.1,
                                label="vol variability (" + keylabel + ")",
                            )
                        ]

                        if int_Q is True:
                            legendhandles += [
                                ax.fill_between(
                                    inc,
                                    modelresultmax["inter"],
                                    modelresultmin["inter"],
                                    color="y",
                                    alpha=0.1,
                                    label="int variability (" + keylabel + ")",
                                )
                            ]

                # a legend for the hatches
                leg1 = ax.legend(
                    handles=legendhandles,
                    labels=[i.get_label() for i in legendhandles],
                )

                if len(legendhandles) == 0:
                    leg1.remove()

        printvariationQ = {key: False for key in minparams}

        def buttonfunc(label):
            # if labels of the buttons have been changed by the labels-argument
            # set the name to the corresponding key (= the actual parameter name)
            for key, val in labels.items():
                if label == val:
                    label = key

            # ax.collections.clear()
            if printvariationQ[label] is True:
                ax.collections.clear()
                printvariationQ[label] = False
            elif printvariationQ[label] is False:
                printvariationQ[label] = True

            animate(paramslider[label].val, key=label)

            plt.draw()

        buttons.on_clicked(buttonfunc)

        for key, slider in paramslider.items():
            slider.on_changed(partial(animate, key=key))

        # define textboxes that allow changing the slider-boundaries
        bounds = dict(
            zip(
                minparams.keys(),
                np.array([list(minparams.values()), list(maxparams.values())]).T,
            )
        )

        def submit(val, key, minmax):
            slider = paramslider[key]
            if minmax == 0:
                bounds[key][0] = float(val)
                slider.valmin = float(val)
                slider.ax.set_xlim(slider.valmin, None)
                minparams[key] = float(val)
            if minmax == 1:
                bounds[key][1] = float(val)
                slider.valmax = float(val)
                slider.ax.set_xlim(None, slider.valmax)
                maxparams[key] = float(val)

            # call animate to update ranges
            animate(params[key], key=key)

            f.canvas.draw_idle()
            plt.draw()

        from matplotlib.widgets import TextBox

        textboxes_buttons = {}
        for i, [key, val] in enumerate(paramslider.items()):

            axbox0 = plt.axes(
                [
                    val.ax.get_position().x0,
                    val.ax.get_position().y1,
                    0.05,
                    0.025,
                ]
            )
            text_box0 = TextBox(axbox0, "", initial=str(round(bounds[key][0], 4)))
            text_box0.on_submit(partial(submit, key=key, minmax=0))

            axbox1 = plt.axes(
                [
                    val.ax.get_position().x1 - 0.05,
                    val.ax.get_position().y1,
                    0.05,
                    0.025,
                ]
            )
            text_box1 = TextBox(axbox1, "", initial=str(round(bounds[key][1], 4)))
            text_box1.on_submit(partial(submit, key=key, minmax=1))

            textboxes_buttons[key + "_min"] = text_box0
            textboxes_buttons[key + "_max"] = text_box1

        textboxes_buttons["buttons"] = buttons

        return f, paramslider, textboxes_buttons

    def intermediate_residuals(
        self,
        fit=None,
        grp="M",
        err="relerr",
        label_formatter=None,
        plottype="3D",
        iter_slice=slice(2, None),
        f_gs=None,
        colorbar=True,
        project_contour=False,
        fmt="%d.%m.%y %H:%M:%S",
        axtitle=None,
        cmap="coolwarm",
    ):
        """
        generate a 2D or 3D  plot of the intermediate residuals during the fit

        Parameters
        ----------
        fit : rt1.rtfits.Fits object
            the rtfits object to use.
        grp : `str` or `tuple`
            the grouping to use, you can either provide

            - a `string`
                - if 'groups', the parameter-groups will be used
                - if 'dataset', the unique dataset-index values will be used
                - all other strings will be interpreted as a pandas datetime
                  offset string (e.g. like 'D', 'M', '10D' etc.)
            - a `tuple`
                - it is interpreted as (key, values or bins) where the
                  key must correspond to a column in fit.dataset or fit.res_df
                  and the second entry is one of the following (see pandas.cut)
                  - `int`, e.g. the number of bins to use
                  - `array-like` e.g. the group-boundaries to use

              Note that only dynamic parameters can be used here since
              a constant can not be grouped!

            The default is 'M'
        err : str, optional
            - 'abserr' for absolute errors
            - 'relerr' for relative errors.

            The default is 'relerr'.
        label_formatter : callable, optional
            A formatter that will be applied to the group-column.
            See "matplotlib.ticker.FuncFormatter" for details.
            The default is None.
        plottype : str, optional
            either '2D' or '3D'. The default is '3D'.
        iter_slice : slice, optional
            a slice to display only a part of the iterations.
            The default is slice(2, None).
        f_gs : (matplotlib.figure, matpltolib.gridspec), optional
            a figure and gridspec object to be used instead of generating a new
            figure. The default is None.
        colorbar : bool, optional
            indicator if a colorbar should be generated. The default is True.
        project_contour : bool, optional
            only if plottype = '3D'. Indicator if the surface should be
            projected to the bottom or not.
            The default is False.
        fmt : str, optional
            only if grp is a string. the datetime-format used for the labels
            The default is '%d.%m.%y %H:%M:%S'.
        axtitle : str, optional
            The axes-label (if None, the provided key will be used)
            The default is None
        Returns
        -------
        ax : matplotlib.axes
            the matplotlib axes used

        """
        if fit is None:
            fit = self.fit
        try:
            fit.intermediate_results
        except AttributeError:
            assert False, (
                "No intermediate results are found, you must run"
                + " performfit() with intermediate_results=True!"
            )

        # get the step size of the iterator to correctly assign the labels
        if iter_slice and iter_slice.step:
            iterstep = iter_slice.step
        else:
            iterstep = 1

        if isinstance(grp, tuple):
            if grp[0] in fit.res_df:
                grps = pd.cut(
                    fit.res_df[grp[0]].reindex(fit.dataset.index),
                    grp[1],
                    include_lowest=True,
                )
            elif grp[0] in fit.dataset:
                grps = pd.cut(fit.dataset[grp[0]], grp[1], include_lowest=True)
            grpvals = grps.cat.categories.mid.values
            ymin, ymax = grps.min().left, grps.max().right

            # group the residuals with respect to the defined group
            resarr = np.abs(
                [
                    pd.DataFrame(i[err], fit.dataset.index)
                    .groupby(grps)
                    .mean()
                    .values.flatten()
                    for i in fit.intermediate_results["residuals"][iter_slice]
                ]
            )
            if axtitle is None:
                use_label = grp[0]
            else:
                use_label = axtitle

        elif isinstance(grp, str):
            if grp == "groups":
                # group the residuals with respect to the defined group
                grplabels = pd.to_datetime(fit.meandatetimes_group)
                resarr = np.abs(
                    [
                        pd.DataFrame(i[err], fit.dataset.index)
                        .groupby(fit._groupindex)
                        .mean()
                        .values.flatten()
                        for i in fit.intermediate_results["residuals"][iter_slice]
                    ]
                )
                grpvals = np.arange(resarr.shape[1])
                ymin, ymax = grpvals.min() - 0.5, grpvals.max() + 0.5

                def label_formatter(x, pos=None):
                    if x in grpvals:
                        return grplabels[int(x)].strftime(fmt)
                    else:
                        return ""

                use_label = ""
            elif grp == "dataset":
                # group the residuals with respect to the dataset-index
                grplabels = pd.to_datetime(fit.index)
                resarr = np.abs(
                    [
                        pd.DataFrame(i[err], fit.dataset.index)
                        .groupby(level=0)
                        .mean()
                        .values.flatten()
                        for i in fit.intermediate_results["residuals"][iter_slice]
                    ]
                )
                grpvals = np.arange(resarr.shape[1])
                ymin, ymax = grpvals.min() - 0.5, grpvals.max() + 0.5

                def label_formatter(x, pos=None):
                    if x in grpvals:
                        return grplabels[int(x)].strftime(fmt)
                    else:
                        return ""

                use_label = ""
            else:
                # group the residuals with respect to the defined group
                grplabels = (
                    pd.DataFrame(
                        fit.intermediate_results["residuals"][0][err],
                        fit.dataset.index,
                    )
                    .groupby(pd.Grouper(freq=grp))
                    .mean()
                    .index
                )

                resarr = np.abs(
                    [
                        pd.DataFrame(i[err], fit.dataset.index)
                        .groupby(pd.Grouper(freq=grp))
                        .mean()
                        .values.flatten()
                        for i in fit.intermediate_results["residuals"][iter_slice]
                    ]
                )
                grpvals = np.arange(resarr.shape[1])
                ymin, ymax = grpvals.min() - 0.5, grpvals.max() + 0.5

                def label_formatter(x, pos=None):
                    if x in grpvals:
                        return grplabels[int(x)].strftime(fmt)

                    else:
                        return ""

                use_label = ""

        # mask for nan-values
        resarr = np.ma.masked_array(resarr, np.isnan(resarr))

        xvals = np.arange(1, len(fit.intermediate_results["residuals"]) + 1)[iter_slice]
        yvals = np.sort(pd.unique(grpvals))

        if plottype == "3D":
            # generate a 3D surface plot

            Y, X = np.meshgrid(yvals, xvals)
            if f_gs is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig = f_gs[0]
                ax = fig.add_subplot(f_gs[1], projection="3d")
            ax.grid()
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a.pane.set_color("none")

            if project_contour:
                # project a contour-plot to the bottom
                ax.contourf(
                    X,
                    Y,
                    resarr,
                    zdir="z",
                    offset=-(np.nanmax(resarr) - np.nanmin(resarr)) * 0.1,
                    cmap=plt.cm.coolwarm,
                    alpha=0.5,
                    extent=[xvals.min() - iterstep, xvals.max(), ymin, ymax],
                    origin="lower",
                )

                ax.set_zlim(
                    np.nanmin(resarr) - (np.nanmax(resarr) - np.nanmin(resarr)) * 0.1,
                    np.nanmax(resarr),
                )

            # plot a 3D surface
            surf = ax.plot_surface(
                X,
                Y,
                resarr,
                cmap=cmap,
                linewidth=0,
                antialiased=True,
                vmin=np.nanmin(resarr),
                vmax=np.nanmax(resarr),
                rcount=500,
                ccount=500,
                alpha=1,
            )

            # find points that are surrounded by a mask and
            # plot a connection-line and scatterpoints
            maskp = [
                i == (True, False, True)
                for i in pairwise([True, *np.all(resarr.mask, axis=0), True], 3)
            ]

            for i in np.arange(len(maskp))[maskp]:
                ax.plot_wireframe(
                    X.T[[i]],
                    Y.T[[i]],
                    resarr.T[[i]],
                    linewidth=0.25,
                    antialiased=False,
                    linestyle=(0, (10, 10)),
                    alpha=0.25,
                    color="k",
                )

                surf = ax.scatter(
                    X.T[i],
                    Y.T[i],
                    resarr.T[i],
                    c=resarr.T[i],
                    cmap=cmap,
                    vmin=np.nanmin(resarr),
                    vmax=np.nanmax(resarr),
                )

            # incorporate the ticker if provided
            if label_formatter is not None:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_formatter))

        elif plottype == "2D":
            # generate a 2D imshow plot
            if f_gs is None:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111)
            else:
                fig = f_gs[0]
                ax = f_gs[0].add_subplot(f_gs[1])

            surf = ax.imshow(
                resarr.T,
                cmap=cmap,
                vmin=np.nanmin(resarr),
                vmax=np.nanmax(resarr),
                extent=[
                    xvals.min() - 0.5 * iterstep,
                    xvals.max() + 0.5 * iterstep,
                    ymin,
                    ymax,
                ],
                origin="lower",
                aspect="auto",
            )

            def xformatter(x, pos=None):
                if x in xvals:
                    return int(x)
                else:
                    return ""

            # show only relevant tick-labels on x-axis
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(xformatter))

            if label_formatter is not None:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(label_formatter))

            ax.set_ylim(ymin, ymax)

        if colorbar is True:
            cb = plt.colorbar(surf)
            if err == "abserr":
                cb.set_label(r"Absolute error   $(x_{fit} - x_{data})$")
            if err == "relerr":
                cb.set_label(
                    r"Relative error   $\frac{(x_{fit} - x_{data})}{x_{data}}$"
                )

        ax.set_ylabel(use_label)
        ax.set_xlabel("# fit-iteration")
        fig.tight_layout()
        return fig

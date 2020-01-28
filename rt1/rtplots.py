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

from .general_functions import rectangularize, dBsig0convert
from rt1.rt1 import RT1

# plot of 3d scattering distribution
#import mpl_toolkits.mplot3d as plt3d


def _evalfit(fit, inc=None, return_components=True):
    '''
    get backscatter timeseries values from a rtfits.Fits object
    (possibly replace the incidence-angle array by an alternative one)
    (used in printsig0analysis)

    Parameters
    ----------
    fit : rt1.rtfits.Fits object
        The fit-object whose backscatter should be returned.
    inc : array-like, optional
        The incidence-angles at which the backscatter values are intended
        to be evaluated. If None, the incidence-angles of the provided
        fit-object will be used (e.g. fit.R.t_0). The default is None.
    return_components : bool, optional
        Indicator if all components (tot, surf, vol, inter) or just the total
        backscatter should be returned. The default is True.

    Returns
    -------
    totsurfvolinter : dict
        A dict with columns 'inc' and 'tot' ('surf','vol','inter')
        corresponding to the incidence-angles and the backscatter-values
        respectively.
    '''

    if inc is not None:
        # copy initial values
        orig_fixed_dict = copy.deepcopy(fit.fixed_dict)
        orig_t_0 = copy.deepcopy(fit.R.t_0)
        orig_p_0 = copy.deepcopy(fit.R.p_0)

        assert len(fit.R.t_0) == len(inc), 'inc and fit.R.t_0 must have the ' \
                                            + 'same length to allow correct ' \
                                            + 'array-broadcasting'

        fit.R.t_0 = inc
        fit.R.p_0 = np.full_like(inc, 0.)

        for key, val in fit.fixed_dict.items():
            if val.shape != inc.shape:
                print(f'shape {val.shape} of fixed_input for "{key}" does ' +
                      f'not match shape {inc.shape} of "inc" and is updated')
                fit.fixed_dict[key] = np.repeat(np.mean(val, axis=1)[:,np.newaxis],
                                                inc.shape[1], axis=1)

    totsurfvolinter = fit._calc_model(return_components=return_components)
    if return_components is True:
        totsurfvolinter = dict(zip(['tot', 'surf', 'vol', 'inter'],
                                   totsurfvolinter))
    else:
        totsurfvolinter = dict(tot=totsurfvolinter)

    # get incidence-angles
    totsurfvolinter['incs'] = fit.R.t_0

    # revert changes to the dicts
    if inc is not None:
        fit.fixed_dict = orig_fixed_dict
        fit.R.t_0 = orig_t_0
        fit.R.p_0 = orig_p_0

    return totsurfvolinter


def _getbackscatter(params=dict(), fit=None, set_V_SRF=None, inc=None,
                    dB=True, sig0=True, int_Q = False, return_fnevals=False,
                    **kwargs):
    '''
    get backscatter values based on given configuration and parameter values
    (used in analyzemodel)

    Parameters
    ----------
    params : dict
        A dict containing the values for ALL parameters involved.
    fit : rt1.rtfits.Fits, optional
        Optionally provide a fits-object from which `set_V_SRF`, `inc` and
        `params['bsf']` will be retrieved if not provided explicitly.
        The default is None.
    set_V_SRF : callable, optional
        A setter function for V and SRF. (see rt1.rtfits.Fits for details)
        The default is None.
    inc : array-like, optional
        The incidence-angles to be used in the calculation. The default is None
    dB : bool, optional
        Indicator if the values should be returned in linear-units or dB.
        The default is True.
    sig0 : bool, optional
        Indicator if intensity- or sigma-0 values should be returned.
        ( sig0 = 4 pi cos(theta) I ). The default is True.
    int_Q : bool, optional
        Indicator if the interaction-term should be evaluated.
        The default is False.
    return_fnevals : bool, optional
        Indicator if the obtained _fnevals functions should be returned.
        (if true, the returned dict will contain a key '_fnevals' with
        the fnevals-function). The default is False.
    **kwargs :
        kwargs passed to the initialization of the rt1.RT1 object.

    Returns
    -------
    tsvi : dict
           a dict with keys 'tot', ('surf', 'vol', ('inter'), ('_fnevals'))

    '''

    params = params.copy()

    if fit is not None:
        if set_V_SRF is None: set_V_SRF = fit.set_V_SRF
        if 'bsf' not in params: params['bsf'] = fit.R.bsf
        # get incidence-angle from fit if inc is not provided explicitly
        if inc is None: inc = copy.deepcopy(fit.inc)

    bsf = params.pop('bsf', 0)

    # set V and SRF
    V, SRF = set_V_SRF(**params)

    R = RT1(1., inc, inc, np.zeros_like(inc), np.full_like(inc, np.pi),
            V=V, SRF=SRF, geometry='mono', bsf = bsf, param_dict=params,
            int_Q=int_Q, **kwargs)

    tsvi = dict(zip(['tot','surf','vol','inter'], R.calc()))

    # convert to sig0 and dB if required
    for key, val in tsvi.items():
        tsvi[key] = dBsig0convert(tsvi[key], inc, dB, sig0, False, False)

    if return_fnevals is True: tsvi['_fnevals'] = R._fnevals

    return tsvi


def polarplot(R=None, SRF=None, V=None, incp=[15., 35., 55., 75.],
              incBRDF=[15., 35., 55., 75.], pmultip=2., BRDFmultip=1.,
              plabel='Volume-Scattering Phase Function',
              BRDFlabel='Surface-BRDF', paprox=True, BRDFaprox=True,
              plegend=True, plegpos=(0.75, 0.5), BRDFlegend=True,
              BRDFlegpos=(0.285, 0.5), groundcolor="none",
              Vparam_dict = [{}],
              BRDFparam_dict = [{}]):
    """
    Generation of polar-plots of the volume- and the surface scattering
    phase function as well as the used approximations in terms of
    legendre-polynomials.


    Parameters
    -----------
    R : RT1-class object
        If R is provided, SRF and V are taken from it
        as V = R.V and SRF = R.SRF
    SRF : RT1.surface class object
          Alternative direct specification of the surface BRDF,
          e.g. SRF = CosineLobe(i=3, ncoefs=5)
    V : RT1.volume class object
        Alternative direct specification of the volume-scattering
        phase-function  e.g. V = Rayleigh()

    Other Parameters
    -----------------
    incp : list of floats (default = [15.,35.,55.,75.])
           Incidence-angles in degree at which the volume-scattering
           phase-function will be plotted
    incBRDF : list of floats (default = [15.,35.,55.,75.])
              Incidence-angles in degree at which the BRDF will be plotted
    pmultip : float (default = 2.)
              Multiplicator to scale the plotrange for the plot of the
              volume-scattering phase-function
              (the max-plotrange is given by the max. value of V in
              forward-direction (for the chosen incp) )
    BRDFmultip : float (default = 1.)
                 Multiplicator to scale the plotrange for the plot of
                 the BRDF (the max-plotrange is given by the max. value
                 of SRF in specular-direction (for the chosen incBRDF) )
    plabel : string
             Manual label for the volume-scattering phase-function plot
    BRDFlabel : string
                Manual label for the BRDF plot
    paprox : boolean (default = True)
             Indicator if the approximation of the phase-function in terms
             of Legendre-polynomials will be plotted.
    BRDFaprox : boolean (default = True)
             Indicator if the approximation of the BRDF in terms of
             Legendre-polynomials will be plotted.
    plegend : boolean (default = True)
             Indicator if a legend should be shown that indicates the
             meaning of the different colors for the phase-function
    plegpos : (float,float) (default = (0.75,0.5))
             Positioning of the legend for the V-plot (controlled via
             the matplotlib.legend keyword  bbox_to_anchor = plegpos )
    BRDFlegend : boolean (default = True)
             Indicator if a legend should be shown that indicates the
             meaning of the different colors for the BRDF
    BRDFlegpos : (float,float) (default = (0.285,0.5))
             Positioning of the legend for the SRF-plot (controlled via
             the matplotlib.legend keyword  bbox_to_anchor = BRDFlegpos)
    groundcolor : string (default = "none")
             Matplotlib color-indicator to change the color of the lower
             hemisphere in the BRDF-plot possible values are:
             ('r', 'g' , 'b' , 'c' , 'm' , 'y' , 'k' , 'w' , 'none')

    Returns
    ---------
    polarfig : figure
               a matplotlib figure showing a polar-plot of the functions
               specified by V or SRF
    """

    assert isinstance(incp, list), 'Error: incidence-angles for ' + \
        'polarplot of p must be a list'
    assert isinstance(incBRDF, list), 'Error: incidence-angles for ' + \
        'polarplot of the BRDF must be a list'
    for i in incBRDF:
        assert i <= 90, 'ERROR: the incidence-angle of the BRDF in ' + \
            'polarplot must be < 90'

    assert isinstance(pmultip, float), 'Error: plotrange-multiplier ' + \
        'for polarplot of p must be a floating-point number'
    assert isinstance(BRDFmultip, float), 'Error: plotrange-' + \
        'multiplier for plot of the BRDF must be a floating-point number'

    assert isinstance(plabel, str), 'Error: plabel of V-plot must ' + \
        'be a string'
    assert isinstance(BRDFlabel, str), 'Error: plabel of SRF-plot ' + \
        'must be a string'

    if R is None and SRF is None and V is None:
        assert False, 'Error: You must either provide R or SRF and/or V'

    # if R is provided, use it to define SRF and V,
    # else use the provided functions
    if R is not None:
        SRF = R.SRF
        V = R.V

    # define functions for plotting that evaluate the used
    # approximations in terms of legendre-polynomials
    if V is not None:
        # if V is a scalar, make it a list
        if np.ndim(V) == 0:
            V = [V]

        # make new figure
        if SRF is None:
            # if SRF is None, plot only a single plot of p
            polarfig = plt.figure(figsize=(7, 7))
            polarax = polarfig.add_subplot(111, projection='polar')
        else:
            # plot p and the BRDF together
            polarfig = plt.figure(figsize=(14, 7))
            polarax = polarfig.add_subplot(121, projection='polar')

        # plot of volume-scattering phase-function's
        pmax = 0
        for n_V, V in enumerate(V):
            # define a plotfunction of the legendre-approximation of p
            if paprox is True:
                phasefunktapprox = sp.lambdify((
                    'theta_0', 'theta_s',
                    'phi_0', 'phi_s', *Vparam_dict[n_V].keys()),
                    V.legexpansion('theta_0', 'theta_s',
                                   'phi_0', 'phi_s', 'vvvv').doit(),
                    modules=["numpy", "sympy"])

            # set incidence-angles for which p is calculated
            plottis = np.deg2rad(incp)
            colors = ['k', 'r',
                      'g', 'b',
                      'c', 'm',
                      'y'] * int(round((len(plottis) / 7. + 1)))

            #pmax = pmultip * np.max(V.p(plottis, np.pi - plottis, 0., 0.))
            for i in plottis:
                ts = np.arange(0., 2. * np.pi, .01)
                pmax_i = pmultip * np.max(V.p(np.full_like(ts, i),
                                            ts,
                                            0.,
                                            0.,
                                            param_dict=Vparam_dict[n_V]))
                if pmax_i > pmax:
                    pmax = pmax_i


            if plegend is True:
                legend_lines = []

            # set color-counter to 0
            i = 0
            for ti in plottis:
                color = colors[i]
                i = i + 1
                thetass = np.arange(0., 2. * np.pi, .01)
                rad = V.p(ti, thetass, 0., 0., param_dict=Vparam_dict[n_V])
                if paprox is True:
                    # the use of np.pi-ti stems from the definition
                    # of legexpansion() in volume.py
                    radapprox = phasefunktapprox(theta_0=np.pi - ti,
                                                 theta_s=thetass,
                                                 phi_0=0.,
                                                 phi_s=0.,
                                                 **Vparam_dict[n_V])
                # set theta direction to clockwise
                polarax.set_theta_direction(-1)
                # set theta to start at z-axis
                polarax.set_theta_offset(np.pi / 2.)

                polarax.plot(thetass, rad, color)
                if paprox is True:
                    polarax.plot(thetass, radapprox, color + '--')
                polarax.arrow(-ti, pmax * 1.2, 0., -pmax * 0.8,
                              head_width=.0, head_length=.0,
                              fc=color, ec=color, lw=1, alpha=0.3)

                polarax.fill_between(thetass, rad, alpha=0.2, color=color)
                polarax.set_xticklabels(['$0^\circ$', '$45^\circ$',
                                         '$90^\circ$', '$135^\circ$',
                                         '$180^\circ$'])
                polarax.set_yticklabels([])
                polarax.set_rmax(pmax * 1.2)
                polarax.set_title(plabel + '\n')

        # add legend for covering layer phase-functions
        if plegend is True:
            i = 0
            for ti in plottis:
                color = colors[i]
                legend_lines += [mlines.Line2D(
                    [], [], color=color,
                    label='$\\theta_0$ = ' + str(
                        np.round_(np.rad2deg(ti),
                                  decimals=1)) + '${}^\circ$')]
                i = i + 1

            if paprox is True:
                legend_lines += [mlines.Line2D(
                    [], [], color='k',
                    linestyle='--', label='approx.')]

            legend = plt.legend(bbox_to_anchor=plegpos,
                                loc=2, handles=legend_lines)
            legend.get_frame().set_facecolor('w')
            legend.get_frame().set_alpha(.5)

    if SRF is not None:
        # if SRF is a scalar, make it a list
        if np.ndim(SRF) == 0:
            SRF = [SRF]

        # append to figure or make new figure
        if V is None:
            # if V is None, plot only a single plot of the BRDF
            polarfig = plt.figure(figsize=(7, 7))
            polarax = polarfig.add_subplot(111, projection='polar')
        else:
            # plot p and the BRDF together
            polarax = polarfig.add_subplot(122, projection='polar')

        if BRDFlegend is True:
            legend_lines = []

        # plot of BRDF
        brdfmax = 0
        for n_SRF, SRF in enumerate(SRF):
            # define a plotfunction of the analytic form of the BRDF
            if BRDFaprox is True:
                brdffunktapprox = sp.lambdify(
                    ('theta_ex', 'theta_s', 'phi_ex', 'phi_s', *BRDFparam_dict[n_SRF].keys()),
                    SRF.legexpansion(
                        'theta_ex', 'theta_s', 'phi_ex', 'phi_s', 'vvvv'
                        ).doit(), modules=["numpy", "sympy"])

            # set incidence-angles for which the BRDF is calculated
            plottis = np.deg2rad(incBRDF)
            colors = ['k', 'r',
                      'g', 'b',
                      'c', 'm',
                      'y'] * int(round((len(plottis) / 7. + 1)))

            #brdfmax = BRDFmultip * np.max(SRF.brdf(plottis,
            #                                       plottis, 0., 0.))

            for i in plottis:
                ts = np.arange(0., 2. * np.pi, .01)
                brdfmax_i = BRDFmultip * np.max(SRF.brdf(
                        np.full_like(ts, i), ts, 0., 0.,
                        param_dict=BRDFparam_dict[n_SRF]))
                if brdfmax_i > brdfmax:
                    brdfmax = brdfmax_i


            # set color-counter to 0
            i = 0
            for ti in plottis:
                color = colors[i]
                i = i + 1
                thetass = np.arange(-np.pi / 2., np.pi / 2., .01)
                rad = SRF.brdf(ti, thetass, 0., 0.,
                               param_dict=BRDFparam_dict[n_SRF])
                if BRDFaprox is True:
                    radapprox = brdffunktapprox(theta_ex=ti,
                                                theta_s=thetass,
                                                phi_ex=0.,
                                                phi_s=0.,
                                                **BRDFparam_dict[n_SRF])
                # set theta direction to clockwise
                polarax.set_theta_direction(-1)
                # set theta to start at z-axis
                polarax.set_theta_offset(np.pi / 2.)

                polarax.plot(thetass, rad, color=color)
                if BRDFaprox is True:
                    polarax.plot(thetass, radapprox, color + '--')

                polarax.fill(
                    np.arange(np.pi / 2., 3. * np.pi / 2., .01),
                    np.ones_like(np.arange(np.pi / 2.,
                                           3. * np.pi / 2.,
                                           .01)
                                 ) * brdfmax * 1.2, color=groundcolor)

                polarax.arrow(-ti, brdfmax * 1.2, 0.,
                              -brdfmax * 0.8, head_width=.0,
                              head_length=.0, fc=color,
                              ec=color, lw=1, alpha=0.3)

                polarax.fill_between(thetass, rad, alpha=0.2, color=color)
                polarax.set_xticklabels(['$0^\circ$',
                                         '$45^\circ$',
                                         '$90^\circ$'])
                polarax.set_yticklabels([])
                polarax.set_rmax(brdfmax * 1.2)
                polarax.set_title(BRDFlabel + '\n')

        # add legend for BRDF's
        if BRDFlegend is True:
            i = 0
            for ti in plottis:
                color = colors[i]
                legend_lines += [
                    mlines.Line2D([], [], color=color,
                                  label='$\\theta_0$ = ' + str(
                        np.round_(np.rad2deg(ti), decimals=1)) +
                        '${}^\circ$')]
                i = i + 1
            if BRDFaprox is True:
                legend_lines += [mlines.Line2D([], [], color='k',
                                               linestyle='--',
                                               label='approx.')]

            legend = plt.legend(bbox_to_anchor=BRDFlegpos,
                                loc=2, handles=legend_lines)
            legend.get_frame().set_facecolor('w')
            legend.get_frame().set_alpha(.5)

    plt.show()
    return polarfig


def linplot3d(theta, phi, Itot=None, Isurf=None, Ivol=None,
              Iint=None, surfmultip=1., zoom=2.):
    """
    Generation of a spherical 3D-plot to visualize bistatic
    scattering behaviour

    Parameters
    -----------

    theta, phi : 2d arrays of floats
                 The angular grid at which the scattered radiation has been
                 evaluated. If t_ex and p_ex are arrays of the azimuth-
                 and polar angles to be used, theta and phi can be
                 calculated via:     theta,phi = np.meshgrid(t_ex, p_ex)
    Itot, Ivol, Isurf, Iint : 2d arrays of floats
                              individual bistatic signal contributions
                              i.e. outputs from RT1.calc()  with
                              RT1.geometry = 'fvfv'   ,
                              RT1.t_ex = theta    ,
                              RT1.p_ex = phi.
                              At least one of the arrays must be provided
                              and they must have the same shape
                              as theta and phi!
    Other Parameters
    -----------------
    surfmultip : float (default = 1.)
                 Scaling factor for the plotted surface that indicates
                 the ground.
    zoom : float (default = 2.)
           Factor to scale the plot.
           The factor is included in order to be able to change the
           plotrange while keeping a constant aspect-ratio of 1. in
           the 3d-plot. (This workaround is necessary due to an open
           issue concerning aspect-ratio in matplotlib's 3d-plots)

    Returns
    --------
    fig : figure
          a matplotlib figure showing a 3d plot of the given input-arrays

    """

    assert isinstance(theta, np.ndarray), 'Error: theta must be' + \
        ' a numpy-array'
    assert isinstance(phi, np.ndarray), 'Error: phi must be a numpy-array'
    if Itot is not None:
        assert isinstance(Itot, np.ndarray), 'Error: Itot3d must be' + \
            ' a numpy-array'
    if Isurf is not None:
        assert isinstance(Isurf, np.ndarray), 'Error: Isurf3d must be' + \
            ' a numpy-array'
    if Ivol is not None:
        assert isinstance(Ivol, np.ndarray), 'Error: Ivol3d must be' + \
            ' a numpy-array'
    if Iint is not None:
        assert isinstance(Iint, np.ndarray), 'Error: Iint3d must be' + \
            ' a numpy-array'
    assert isinstance(surfmultip, float), 'Error: surfmultip must be' + \
        ' a floating-point number'
    assert surfmultip > 0., 'Error: surfmultip must be larger than 0.'

    assert isinstance(zoom, float), 'Error: zoom must be a ' + \
        'floating-point number'
    assert zoom > 0., 'Error: zoom must be larger than 0.'

    # transform values to spherical coordinate system
    def sphericaltransform(r):
        if r is None:
            return None

        X = r * np.sin(theta) * np.cos(phi)
        Y = r * np.sin(theta) * np.sin(phi)
        Z = r * np.cos(theta)
        return X, Y, Z

    fig = plt.figure(figsize=(7, 7))
    ax3d = fig.add_subplot(1, 1, 1, projection='3d')

    # ax3d.view_init(elev=20.,azim=45)

    # calculate maximum value of all given imput-arrays
    m = []
    if Itot is not None:
        m = m + [np.max(sphericaltransform(Itot)),
                 np.abs(np.min(sphericaltransform(Itot)))]
    if Isurf is not None:
        m = m + [np.max(sphericaltransform(Isurf)),
                 np.abs(np.min(sphericaltransform(Isurf)))]
    if Ivol is not None:
        m = m + [np.max(sphericaltransform(Ivol)),
                 np.abs(np.min(sphericaltransform(Ivol)))]
    if Iint is not None:
        m = m + [np.max(sphericaltransform(Iint)),
                 np.abs(np.min(sphericaltransform(Iint)))]
    maximum = np.max(m)

    xx = np.array([- surfmultip * maximum, surfmultip * maximum])
    yy = np.array([0., surfmultip * maximum])
    xxx, yyy = np.meshgrid(xx, yy)
    zzz = np.ones_like(xxx) * (0.)

    ax3d.plot_surface(xxx, yyy, zzz, alpha=0.2, color='k')

    if Itot is not None:
        ax3d.plot_surface(
            sphericaltransform(Itot)[0], sphericaltransform(Itot)[1],
            sphericaltransform(Itot)[2], rstride=1, cstride=1,
            color='Gray', linewidth=0, antialiased=True, alpha=.3)

    if Isurf is not None:
        ax3d.plot_surface(
            sphericaltransform(Isurf)[0], sphericaltransform(Isurf)[1],
            sphericaltransform(Isurf)[2], rstride=1, cstride=1,
            color='Red', linewidth=0, antialiased=True, alpha=.5)

    if Ivol is not None:
        ax3d.plot_surface(
            sphericaltransform(Ivol)[0], sphericaltransform(Ivol)[1],
            sphericaltransform(Ivol)[2], rstride=1, cstride=1,
            color='Green', linewidth=0, antialiased=True, alpha=.5)

    if Iint is not None:
        ax3d.plot_surface(
            sphericaltransform(Iint)[0], sphericaltransform(Iint)[1],
            sphericaltransform(Iint)[2], rstride=1, cstride=1,
            color='Blue', linewidth=0, antialiased=True, alpha=.5)

    ax3d.w_xaxis.set_pane_color((1., 1., 1., 0.))
    ax3d.w_xaxis.line.set_color((1., 1., 1., 0.))
    ax3d.w_yaxis.set_pane_color((1., 1., 1., 0.))
    ax3d.w_yaxis.line.set_color((1., 1., 1., 0.))
    # ax3d.w_zaxis.set_pane_color((0.,0.,0.,.1))
    ax3d.w_zaxis.set_pane_color((1., 1., 1., .0))
    ax3d.w_zaxis.line.set_color((1., 1., 1., 0.))
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    ax3d.set_xlim(-maximum / zoom, maximum / zoom)
    ax3d.set_ylim(-maximum / zoom, maximum / zoom)
    ax3d.set_zlim(0, 2 * maximum / zoom)

    # ensure display of correct aspect ratio (bug in mplot3d)
    # due to the bug it is only possible to have equally sized axes
    # (without changing the mplotlib code itself)

    # ax3d.auto_scale_xyz([np.min(sphericaltransform(Itot3d)),
    #                     np.max(sphericaltransform(Itot3d))],
    #                    [0., np.max(sphericaltransform(Itot3d)) +
    #                     np.abs(np.min(sphericaltransform(Itot3d)))],
    #                    [-np.max(sphericaltransform(Itot3d)) +
    #                     np.abs(np.min(sphericaltransform(Itot3d))),
    #                     np.max(sphericaltransform(Itot3d)) +
    #                     np.abs(np.min(sphericaltransform(Itot3d)))])

    plt.show()
    return fig

def hemreflect(R=None, SRF=None, phi_0=0., t_0_step=5., t_0_min=0.,
               t_0_max=90., simps_N=1000, showpoints=True,
               returnarray=False, param_dict={}):
    '''
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
    '''

    from scipy.integrate import simps

    # choose BRDF function to be evaluated
    if R is not None:
        BRDF = R.SRF.brdf

        try:
            Nsymb = R.SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, R.SRF.NormBRDF[0],
                                modules=['numpy'])
            NormBRDF = Nfunc(*[param_dict[str(i)] for i in Nsymb])
        except Exception:
            NormBRDF = R.SRF.NormBRDF
    elif SRF is not None:
        BRDF = SRF.brdf
        try:
            Nsymb = SRF.NormBRDF[0].free_symbols
            Nfunc = sp.lambdify(Nsymb, SRF.NormBRDF[0],
                                modules=['numpy'])
            NormBRDF = Nfunc(*[param_dict[str(i)] for i in Nsymb])
        except Exception:
            NormBRDF = SRF.NormBRDF
    else:
        assert False, 'Error: You must provide either R or SRF'

    # set incident (zenith-angle) directions for which the integral
    # should be evaluated!
    incnum = np.arange(t_0_min, t_0_max, t_0_step)

    # define grid for integration
    x = np.linspace(0., np.pi / 2., simps_N)
    y = np.linspace(0., 2 * np.pi, simps_N)

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
            return np.sin(theta_s) * np.cos(theta_s) * BRDF(theta_0,
                                                            theta_s,
                                                            phi_0, phi_s,
                                                            param_dict=param_dict)
        # evaluate the integral using Simpson's Rule twice
        z = integfunkt(x[:, None], y)
        sol = sol + [simps(simps(z, y), x)]

    sol = np.array(sol) * NormBRDF

    # print warning if the hemispherical reflectance exceeds 1
    if np.any(sol > 1.):
        print('ATTENTION, Hemispherical Reflectance > 1 !')

    if returnarray is True:
        return sol
    else:
        # generation of plot
        fig = plt.figure()
        axnum = fig.add_subplot(1, 1, 1)

        if len(sol.shape) > 1:
            for i, sol in enumerate(sol):
                axnum.plot(incnum, sol,
                           label='NormBRDF = ' + str(NormBRDF[i][0]))
                if showpoints is True:
                    axnum.plot(incnum, sol, 'r.')
        else:
            axnum.plot(incnum, sol, 'k',
                       label='NormBRDF = ' + str(NormBRDF[0]))
            if showpoints is True:
                axnum.plot(incnum, sol, 'r.')

        axnum.set_xlabel('$\\theta_0$ [deg]')
        axnum.set_ylabel('$R(\\theta_0)$')
        axnum.set_title('Hemispherical reflectance ')
        axnum.set_ylim(0., np.max(sol) * 1.1)

        axnum.legend()

        axnum.grid()
        plt.show()
        return fig


class plot:
    '''
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
        !!! only available if *performfit* has been called with
        *intermediate_results = True*

        generate a plot showing the development of the fitted parameters
        and the residuals for each fit-iteration

    - printsig0analysis
        a widget to analyze the fit-results of individual timestamps within
        the considered timeseries

    '''

    def __init__(self, fit=None, **kwargs):
        self.fit = fit

    def scatter(self, fit=None, mima=None, pointsize=0.5,
                regression=True, newcalc=True,  **kwargs):
        '''
        geerate a scatterplot of modelled vs. original backscatter data

        Parameters:
        ------------
        fit : list
              output of monofit()-function
        Other Parameters:
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

        Returns:
        --------------
        fig : matplotlib.figure
            the used matplotlib figure instance
        '''
        plot('asdf')
        if fit is None:
            fit = self.fit

        # reset incidence-angles in case they have been altered beforehand
        fit.R.t_0 = fit.inc
        fit.R.p_0 = np.zeros_like(fit.inc)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if newcalc is True:

            estimates = fit._calc_model(fit.R, fit.res_dict, fit.fixed_dict)

            # apply mask
            estimates = estimates[~fit.mask]
            measures = fit.data[~fit.mask]

        else:
            # get the residuals and apply mask
            residuals = np.reshape(fit.fit_output.fun, fit.data.shape)
            residuals = np.ma.masked_array(residuals, fit.mask)
            # prepare measurements
            measures = fit.data[~fit.mask]
            # calculate estimates
            estimates = residuals[~fit.mask] + measures

        if mima is None:
            mi = np.min((measures, estimates))
            ma = np.max((measures, estimates))
        else:
            mi, ma = mima

        ax.scatter(estimates, measures, s=pointsize, alpha=0.7, **kwargs)

        # plot 45degree-line
        ax.plot([mi, ma], [mi, ma], 'k--')

        if fit.sig0 is True:
            quantity = r'$\sigma_0$'
        else:
            quantity = 'Intensity'

        if fit.dB is True:
            scale = '[dB]'
        else:
            scale = ''

        ax.set_xlabel('modelled ' + quantity + scale)
        ax.set_ylabel('measured ' + quantity + scale)

        if regression is True:
            from scipy.stats import linregress
            # evaluate linear regression to get r-value etc.
            slope, intercept, r_value, p_value, std_err = linregress(estimates,
                                                                     measures)

            ax.plot(np.sort(measures),
                    intercept + slope * np.sort(measures), 'r--', alpha=0.4)

            ax.text(0.8, .1, '$R^2$ = ' + str(np.round(r_value**2, 2)),
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)

        return fig


    def fit_timeseries(self, fit=None, dB=True, sig0=True, params=None,
                       printtot = True, printsurf = True,
                       printvol = True, printint = True,
                       printorig = True, months = None, years = None,
                       ylim=None, printinc = True):
        '''
        Print individual contributions, resulting parameters and the
        reference dataset of an rt1.rtfits object as timeseries.

        Parameters:
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

        Returns:
        --------------
        f : matplotlib.figure
            the used matplotlib figure instance
        '''

        if fit is None:
            fit = self.fit


        # get incidence-angles
        inc_array = np.ma.masked_array(fit.inc, fit.mask)
        inc = inc_array.compressed()
        # get input dataset
        data = np.ma.masked_array(fit.data, fit.mask)

        # def dBsig0convert(val):
        #     # if results are provided in dB convert them to linear units
        #     if fit.dB is True: val = 10**(val/10.)
        #     # convert sig0 to intensity
        #     if sig0 is False and fit.sig0 is True:
        #         val = val/(4.*np.pi*np.cos(inc))
        #     # convert intensity to sig0
        #     if sig0 is True and fit.sig0 is False:
        #         val = 4.*np.pi*np.cos(inc)*val
        #     # if dB output is required, convert to dB
        #     if dB is True: val = 10.*np.log10(val)
        #     return val

        # calculate individual contributions
        contrib_array = fit._calc_model(R=fit.R,
                                        res_dict=fit.res_dict,
                                        fixed_dict=fit.fixed_dict,
                                        return_components=True)

        # apply mask and convert to pandas dataframe
        contrib_array = [np.ma.masked_array(con,
                                            fit.mask) for con in contrib_array]

        contrib = dict(zip(['tot', 'surf', 'vol', 'inter'],
                                contrib_array))

        contrib['$\\sigma_0$ dataset'] = data
        contrib['inc'] = inc_array

        contrib = {key:pd.DataFrame(val, fit.index).stack().droplevel(1)
                        for key, val in contrib.items()}
        contrib = pd.DataFrame(contrib)

        # convert units
        complist = [i for i in contrib.keys() if i not in ['inc']]
        contrib[complist] = contrib[complist].apply(dBsig0convert,
                                                    inc=inc, dB=dB, sig0=sig0,
                                                    fitdB=fit.dB,
                                                    fitsig0=fit.sig0)


        # drop unneeded columns
        if printint is False and 'inter' in contrib:
            contrib = contrib.drop('inter', axis=1)
        if printtot is False: contrib = contrib.drop('tot', axis=1)
        if printsurf is False: contrib = contrib.drop('surf', axis=1)
        if printvol is False: contrib = contrib.drop('vol', axis=1)
        if printorig is False:
            contrib = contrib.drop('$\\sigma_0$ dataset', axis=1)

        # select years and months
        if years is not None:
            contrib = contrib.loc[contrib.index.year.isin(years)]
        if months is not None:
            contrib = contrib.loc[contrib.index.month.isin(months)]

        # print incidence-angle dependency
        if printinc is True:
            f, [ax, ax_inc] = plt.subplots(ncols=2, figsize=(15,5),
                                           gridspec_kw={'width_ratios':[3,1]},
                                           sharey=True)
            f.subplots_adjust(left=0.05, right=0.98, top=0.98,
                              bottom=0.1, wspace=0.1)

            # -------------------
            color = {'tot':'r', 'surf':'b', 'vol':'g', 'inter':'y',
                     '$\\sigma_0$ dataset':'k'}

            groupedcontrib = contrib.groupby(contrib.index)

            #return contrib, groupedcontrib
            for label in contrib.keys():
                if label in ['inc']: continue
                a=np.rad2deg(rectangularize(
                    [x.values for _, x in groupedcontrib['inc']],
                    return_masked=True)).T
                b=np.array(rectangularize(
                    [x.values for _, x in groupedcontrib[label]],
                    return_masked=True)).T
                x = (np.array([a,b]).T)

                l_col = mpl.collections.LineCollection(x,linewidth =.25,
                                                       label='x',
                                                       color=color[label],
                                                       alpha = 0.5)
                ax_inc.add_collection(l_col)
                ax_inc.scatter(a, b, color=color[label], s=1)
                ax_inc.set_xlim(a.min(), a.max())
                ax_inc.set_xlabel('$\\theta_0$')
            ax_inc.set_xlabel('$\\theta_0$')


        else:
            f, ax = plt.subplots(figsize=(12,5))
            f.subplots_adjust(left=0.05, right=0.98, top=0.98,
                              bottom=0.1, wspace=0.05)

        for label, val in contrib.items():
            if label in ['inc']: continue
            color = {'tot':'r', 'surf':'b', 'vol':'g', 'inter':'y'}
            if printorig is True: color['$\\sigma_0$ dataset'] = 'k'
            ax.plot(val.sort_index(), linewidth =.25, marker='.',
                    ms=2, label=label, color=color[label], alpha = 0.5)
        # overprint parameters
        if params != None:
            paramdf_dict = {}
            # add fitted parameters
            paramdf_dict.update(fit.res_dict)
            # add constant values
            paramdf_dict.update(fit.fixed_dict)

            paramdf = pd.DataFrame(paramdf_dict,
                                   index = fit.index).sort_index()
            if years is not None:
                paramdf = paramdf.loc[paramdf.index.year.isin(years)]
            if months is not None:
                paramdf = paramdf.loc[paramdf.index.month.isin(months)]

            pax = ax.twinx()
            for k in params:
                pax.plot(paramdf[k], lw=1, marker='.', ms=2, label=k)
            pax.legend(loc='upper right', ncol=5)
            pax.set_ylabel('parameter-values')

        # format datetime index
        ax.xaxis.set_minor_locator(mpl.dates.MonthLocator())
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%m'))
        ax.xaxis.set_major_locator(mpl.dates.YearLocator())
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('\n%Y'))

        # set ylabels
        if sig0 is True:
            label = '$\\sigma_0$'
        else:
            label = 'Intensity'
        if dB is True: label += ' [dB]'
        ax.set_ylabel(label)

        # generate legend
        hand, lab = ax.get_legend_handles_labels()
        lab, unique_ind = np.unique(lab, return_index=True)
        ax.legend(handles = list(np.array(hand)[unique_ind]),
                  labels=list(lab), loc='upper left',
                  ncol=5)

        if ylim is not None:
            ax.set_ylim(ylim)

        return f


    def fit_errors(self, fit=None, newcalc=False, relative=False,
                   result_selection='all'):
        '''
        a function to quickly print residuals for each measurement
        and for each incidence-angle value

        Parametsrs:
        ------------
        fit : list
            output of performfit()-function
        newcalc : bool (default = False)
                  indicator whether the residuals shall be re-calculated
                  or not.

                  True:
                      the residuals are calculated using R, inc, mask,
                      res_dict and fixed_dict from the fit-argument
                  False:
                      the residuals are taken from the output of
                      res_lsq from the fit-argument
        relative : bool (default = False)
                   indicator if relative (True) or absolute (False) residuals
                   shall be plotted

        Returns:
        --------------
        fig : matplotlib.figure
            the used matplotlib figure instance
        '''

        if fit is None:
            fit = self.fit

        if result_selection == 'all':
            result_selection = range(len(fit.data))

        if newcalc is False:
            # get residuals from fit into desired shape for plotting
            # Attention -> incorporate weights and mask !
            res = np.ma.masked_array(np.reshape(fit.fit_output.fun,
                                                fit.data.shape), fit.mask)

            if relative is True:
                res = np.ma.abs(res / (res + np.ma.masked_array(fit.data,
                                                                fit.mask)))
            else:
                res = np.ma.abs(res)
        else:
            # Alternative way of calculating the residuals
            # (based on R, inc and res_dict)

            fit.R.t_0 = fit.inc
            fit.R.p_0 = np.zeros_like(fit.inc)

            estimates = fit._calc_model(fit.R, fit.res_dict, fit.fixed_dict)
            # calculate the residuals based on masked arrays
            masked_estimates = np.ma.masked_array(estimates, mask=fit.mask)
            masked_data = np.ma.masked_array(fit.data, mask=fit.mask)

            res = np.ma.sqrt((masked_estimates - masked_data)**2)

            if relative is True:
                res = res / masked_estimates

        # apply mask to data and incidence-angles (and convert to degree)
        inc = np.ma.masked_array(np.rad2deg(fit.inc), mask=fit.mask)

        # make new figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(212)
        if relative is True:
            ax.set_title('Mean relative residual per measurement')
        else:
            ax.set_title('Mean absolute residual per measurement')

        ax2 = fig.add_subplot(211)
        if relative is True:
            ax2.set_title('Relative residuals per incidence-angle')
        else:
            ax2.set_title('Residuals per incidence-angle')

        # the use of masked arrays might cause python 2 compatibility issues!
        ax.plot(fit.index[result_selection], res[result_selection], '.',
                alpha=0.5)

        # plot mean residual for each measurement
        ax.plot(fit.index[result_selection], np.ma.mean(res[result_selection],
                                                        axis=1),
                   'k', linewidth=3, marker='o', fillstyle='none')

        # plot total mean of mean residuals per measurement
        ax.plot(fit.index[result_selection],
                   [np.ma.mean(np.ma.mean(res[result_selection],
                                          axis=1))] * len(result_selection),
                   'k--')

        # add some legends
        res_h = mlines.Line2D(
            [], [], color='black', label='Mean res.  per measurement',
            linestyle='-', linewidth=3, marker='o', fillstyle='none')
        res_h_dash = mlines.Line2D(
            [], [], color='black', linestyle='--', label='Average mean res.',
            linewidth=1, fillstyle='none')

        res_h_dots = mlines.Line2D(
            [], [], color='black', label='Residuals',
            linestyle='-', linewidth=0, marker='.', alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [res_h_dots] + [res_h] + [res_h_dash],
                     loc=1)

        ax.set_ylabel('Residual')

    #        # evaluate mean residuals per incidence-angle
        meanincs = np.ma.unique(np.concatenate(inc[result_selection]))
        mean = np.full_like(meanincs, 0.)

        for a, incval in enumerate(meanincs):
            select = np.where(inc[result_selection] == incval)
            res_selected = res[result_selection][select[0][:, np.newaxis],
                               select[1][:, np.newaxis]]
            mean[a] = np.ma.mean(res_selected)

        sortpattern = np.argsort(meanincs)
        meanincs = meanincs[sortpattern]
        mean = mean[sortpattern]

        # plot residuals per incidence-angle for each measurement
        for i, resval in enumerate(res[result_selection]):
            sortpattern = np.argsort(inc[result_selection[i]])
            ax2.plot(inc[result_selection[i]][sortpattern], resval[sortpattern],
                        ':', alpha=0.5, marker='.')

        # plot mean residual per incidence-angle
        ax2.plot(meanincs, mean,
                    'k', linewidth=3, marker='o', fillstyle='none')

        # add some legends
        res_h2 = mlines.Line2D(
            [], [], color='black', label='Mean res.  per inc-angle',
            linestyle='-', linewidth=3, marker='o', fillstyle='none')
        res_h_lines = mlines.Line2D(
            [], [], color='black', label='Residuals',
            linestyle=':', alpha=0.5)

        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles2 + [res_h_lines] + [res_h2], loc=1)

        ax2.set_xlabel('$\\theta_0$ [deg]')
        ax2.set_ylabel('Residual')

        # find minimum and maximum incidence angle
        maxinc = np.max(inc)
        mininc = np.min(inc)

        ax2.set_xlim(np.floor(mininc) - 1,
                        np.ceil(maxinc) + 1)

        # set major and minor ticks
        ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(.25))

        # set ticks
        if isinstance(fit.index[0], datetime.datetime):
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        fig.tight_layout()

        return fig


    def results(self, fit=None, truevals=None, startvals=False,
                 legends=False, result_selection='all'):
        '''
        a function to quickly print the fit-results and the gained parameters

        Parametsrs:
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
        legends : bool (default = True)
                  indicator if legends should be plotted
        result_selection : list-like or 'all'
                           a list of the measurement-numbers that should be
                           plotted (indexed starting from 0) or 'all' in case
                           all measurements should be plotted
        Returns:
        --------------
        fig : matplotlib.figure
              the used matplotlib figure instance
        '''

        if fit is None:
            fit = self.fit

        # this is done to allow the usage of monofit-outputs as well

        if result_selection == 'all':
            result_selection = range(len(fit.data))

        # assign colors
        colordict = {key:f'C{i%10}' for
                     i, key in enumerate(fit.res_dict.keys())}

        # reset incidence-angles in case they have been altered beforehand
        fit.R.t_0 = fit.inc
        fit.R.p_0 = np.zeros_like(fit.inc)

        # evaluate number of measurements
        Nmeasurements = len(fit.inc)

        if truevals is not None:
            truevals = {**truevals}

            # generate a dictionary to assign values based on input
            for key in truevals:
                if np.isscalar(truevals[key]):
                    truevals[key] = np.array([truevals[key]] * Nmeasurements)
                else:
                    truevals[key] = truevals[key]

        # generate figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(211)
        ax.set_title('Fit-results')

        # plot datapoints
        for i, j in enumerate(np.ma.masked_array(fit.data,
                                                 fit.mask)[result_selection]):
            ax.plot(fit.inc[result_selection[i]], j, '.')

        # reset color-cycle
        plt.gca().set_prop_cycle(None)

        # define incidence-angle range for plotting
#        incplot = np.array([np.linspace(np.min(inc), np.max(inc), 100)]
#                           * Nmeasurements)
        incplot = fit.inc
        # set new incidence-angles
        fit.R.t_0 = incplot
        fit.R.p_0 = np.zeros_like(incplot)

        # calculate results
        fitplot = fit._calc_model(fit.R, fit.res_dict, fit.fixed_dict)

        # generate a mask that hides all measurements where no data has
        # been provided (i.e. whose parameter-results are still the startvals)
        newmask = np.ones_like(incplot) * np.all(fit.mask, axis=1)[:,
                                                                   np.newaxis]
        fitplot = np.ma.masked_array(fitplot, newmask)

        for i, val in enumerate(fitplot[result_selection]):
            ax.plot(incplot[i], val, alpha=0.4, label=result_selection[i])

        # ----------- plot start-values ------------
        if startvals is True:
            startplot = fit._calc_model(fit.R, fit.start_dict, fit.fixed_dict)
            for i, val in enumerate(startplot[result_selection]):
                if i == 0:
                    label = 'fitstart'
                else:
                    label = ''
                ax.plot(incplot[result_selection[i]], val, 'k--', linewidth=1,
                        alpha=0.5, label=label)

        if legends is True:
            ax.legend(loc=1)

        mintic = np.round(np.rad2deg(np.min(fit.inc)) + 4.9, -1)
        if mintic < 0.:
            mintic = 0.
        maxtic = np.round(np.rad2deg(np.max(fit.inc)) + 4.9, -1)
        if maxtic > 360.:
            maxtic = 360.

        ticks = np.arange(np.rad2deg(np.min(fit.inc)),
                          np.rad2deg(np.max(fit.inc)) + 1.,
                          (maxtic - mintic) / 10.)
        plt.xticks(np.deg2rad(ticks), np.array(ticks, dtype=int))
        plt.xlabel('$\\theta_0$ [deg]')
        plt.ylabel('$I_{tot}$')

        ax2 = fig.add_subplot(212)
        ax2.set_title('Estimated parameters')


        if truevals is not None:

            # plot actual values
            for key in truevals:
                ax2.plot(fit.index, truevals[key],
                         '--', alpha=0.75, color=colordict[key])
            for key in truevals:
                ax2.plot(fit.index, truevals[key], 'o',
                         color=colordict[key])

            param_errs = {}
            for key in truevals:
                param_errs[key] = fit.res_dict[key] - truevals[key]

            for key in truevals:
                ax2.plot(fit.index, param_errs[key],
                         ':', alpha=.25, color=colordict[key])
            for key in truevals:
                ax2.plot(fit.index, param_errs[key],
                         '.', alpha=.25, color=colordict[key])

            h2 = mlines.Line2D([], [], color='black', label='data',
                               linestyle='--', alpha=0.75, marker='o')
            h3 = mlines.Line2D([], [], color='black', label='errors',
                               linestyle=':', alpha=0.5, marker='.')


        # plot fitted values
        for key in fit.res_dict:
            ax2.plot(fit.index,
                     np.ma.masked_array(fit.res_dict[key],
                                        np.all(fit.mask, axis=1)),
                     alpha=1., label=key, color=colordict[key])

        if len(result_selection) < len(fit.data):
            for i, resid in enumerate(result_selection):
                ax2.text(fit.index[resid],
                         ax2.get_ylim()[1]*.9,
                         resid,
                         bbox=dict(facecolor=f'C{i}', alpha=0.5))



        h1 = mlines.Line2D([], [], color='black', label='estimates',
                           linestyle='-', alpha=0.75, marker='.')

        handles, labels = ax2.get_legend_handles_labels()
        if truevals is None:
            plt.legend(handles=handles + [h1], loc=1)
        else:
            plt.legend(handles=handles + [h1, h2, h3], loc=1)

        # set ticks
        if isinstance(fit.index[0], datetime.datetime):
            fig.autofmt_xdate()

        if truevals is None:
            plt.ylabel('Parameters')
        else:
            plt.ylabel('Parameters / Errors')

        fig.tight_layout()

        return fig


    def single_results(self, fit=None, fit_numbers=None, fit_indexes=None,
                    hexbinQ=True, hexbinargs={},
                    convertTodB=False):
        '''
        a function to investigate the quality of the individual fits


        Parameters:
        ------------
        fit : list
              output of the monofit()-function
        fit_numbers : list
                      a list containing the position of the measurements
                      that should be plotted (starting from 0)
        fit_indexes : list
                      a list containing the index-values of the measurements
                      that should be plotted

        Other Parameters:
        ------------------
        hexbinQ : bool (default = False)
                  indicator if a hexbin-plot should be underlayed
                  that shows the distribution of the datapoints
        hexbinargs : dict
                     a dict containing arguments to customize the hexbin-plot
        convertTodB : bool (default=False)
                      if set to true, the datasets will be converted to dB

        Returns:
        --------------
        fig : matplotlib.figure
              the used matplotlib figure instance
        '''
        if fit is None:
            fit = self.fit

        if fit_numbers is not None and fit_indexes is not None:
            assert False, 'please provide EITHER fit_numbers OR fit_indexes!'
        elif fit_indexes is not None:
            fit_numbers = np.where(fit.index.isin(fit_indexes))[0]
        elif fit_numbers is None and fit_indexes is None:
            fit_numbers = range(len(fit.index))

        # function to generate colormap that fades between colors
        def CustomCmap(from_rgb, to_rgb):

            # from color r,g,b
            r1, g1, b1 = from_rgb

            # to color r,g,b
            r2, g2, b2 = to_rgb

            cdict = {'red': ((0, r1, r1),
                             (1, r2, r2)),
                     'green': ((0, g1, g1),
                               (1, g2, g2)),
                     'blue': ((0, b1, b1),
                              (1, b2, b2))}

            cmap = LinearSegmentedColormap('custom_cmap', cdict)
            return cmap

        estimates = fit._calc_model(fit.R, fit.res_dict, fit.fixed_dict)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for m_i, m in enumerate(fit_numbers):

            if convertTodB is True:
                y = 10.*np.log10(estimates[m][~fit.mask[m]])
            else:
                y = estimates[m][~fit.mask[m]]

            # plot data
            label = fit.index[m]

            xdata = np.rad2deg(fit.inc[m][~fit.mask[m]])

            if convertTodB is True:
                ydata = 10.*np.log10(fit.data[m][~fit.mask[m]])
            else:
                ydata = fit.data[m][~fit.mask[m]]

            # get color that will be applied to the next line drawn
            dummy, = ax.plot(xdata[0], ydata[0], '.', alpha=0.)
            color = dummy.get_color()

            if hexbinQ is True:
                args = dict(gridsize=15, mincnt=1,
                            linewidths=0., vmin=0.5, alpha=0.7)
                args.update(hexbinargs)

                # evaluate the hexbinplot once to get the maximum number of
                # datapoints within a single hexagonal (used for normalization)
                dummyargs = args.copy()
                dummyargs.update({'alpha': 0.})
                hb = ax.hexbin(xdata, ydata, **dummyargs)

                # generate colormap that fades from white to the color
                # of the plotted data  (asdf.get_color())
                cmap = CustomCmap([1.00, 1.00, 1.00],
                                  plt.cm.colors.hex2color(color))
                # setup correct normalizing instance
                norm = Normalize(vmin=0, vmax=hb.get_array().max())

                ax.hexbin(xdata, ydata, cmap=cmap, norm=norm, **args)

            # plot datapoints
            asdf, = ax.plot(xdata, ydata, '.',
                            color=color, alpha=1.,
                            label=label, markersize=10)

            # plot results
            iii = fit.inc[m][~fit.mask[m]]
            ax.plot(np.rad2deg(iii[np.argsort(iii)]), y[np.argsort(iii)],
                    '-', color='w', linewidth=3)

            ax.plot(np.rad2deg(iii[np.argsort(iii)]), y[np.argsort(iii)],
                    '-', color=asdf.get_color(), linewidth=2)

        ax.set_xlabel('$\\theta_0$ [deg]')
        ax.set_ylabel('$\\sigma_0$ [dB]')

        ax.legend(title='# Measurement')

        return fig


    def intermediate_results(self, fit=None, params = None,
                             cmaps=None):
        '''
        a function to plot the intermediate-results
        (the data is only available if rtfits.monofit has been called with
        the argument intermediate_results=True!)

        Parameters:
        -------------
        fit : rtfits object
              the rtfits-object containing the fit-results
        params : list
            a list of parameter-names that are intended to be plotted
            as timeseries.
        cmaps : list
            a list of the colormaps used to plot the parameter variations

        Returns:
        --------------
        f : matplotlib.figure
            the used matplotlib figure instance
        '''

        if fit is None:
            fit = self.fit

        try:
            fit.intermediate_results
        except AttributeError:
            assert False, ('No intermediate results are found, you must run' +
                           ' monofit() with intermediate_results=True flag!')

        if params is None:
            params = fit.res_dict.keys()


        if cmaps is None:
            cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys',
                     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

        interparams = {}
        for i, valdict in enumerate(fit.intermediate_results['parameters']):
            for key, val in valdict.items():
                if key in interparams:
                    interparams[key] += [[i, np.mean(val)]]
                else:
                    interparams[key] = [[i, np.mean(val)]]
        intererrs = {}
        for i, valdict in enumerate(fit.intermediate_results['residuals']):
            for key, val in valdict.items():
                if key in intererrs:
                    intererrs[key] += [[i, np.mean(val)]]
                else:
                    intererrs[key] = [[i, np.mean(val)]]
        interjacs = {}
        for i, valdict in enumerate(fit.intermediate_results['jacobian']):
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
            interres_p = pd.concat([pd.DataFrame(valdict[key],
                                                 fit.index.drop_duplicates(),
                                    columns=[i])
                                    for i, valdict in enumerate(
                                            fit.intermediate_results[
                                                'parameters'])
                                    ], axis=1)
            interres_params[key] = interres_p


        f = plt.figure(figsize=(15,10))
        f.subplots_adjust(top=0.98, left=0.05, right=0.95)
        gs = GridSpec(4, len(interjacs),
                               #width_ratios=[1, 2],
                               #height_ratios=[1, 2, 1]
                               )
        axsm = plt.subplot(gs[0,:])
        axerr = plt.subplot(gs[1,:])
        paramaxes, jacaxes = [], []
        for i in range(len(interjacs)):
            paramaxes += [plt.subplot(gs[2,i])]
            jacaxes += [plt.subplot(gs[3,i])]

        smhandles, smlabels = [],[]
        for nparam, [parameter, paramdf] in enumerate(interres_params.items()):

            cmap = plt.get_cmap(cmaps[nparam])

            for key, val in paramdf.items():
                axsm.plot(val,
                          c=cmap((float(key)/len(paramdf.keys()))),
                          lw=0.25, marker='.', ms=2)

            # add colorbar
            axcb = f.add_axes([axsm.get_position().x1-.01*(nparam+1),
                               axsm.get_position().y0,
                               .01,
                               axsm.get_position().y1-axsm.get_position().y0])

            cbbounds = [1] + list(np.arange(2, len(paramdf.keys()) + 1, 1))


            cb = mpl.colorbar.ColorbarBase(
                axcb, cmap=cmap, orientation = 'vertical',
                boundaries = [0] + cbbounds + [cbbounds[-1]+1],
                spacing='proportional',
                norm=mpl.colors.BoundaryNorm(cbbounds, cmap.N))
            if nparam > 0: cb.set_ticks([])

            smhandles += [mpl.lines.Line2D([],[], color=cmap(.9))]
            smlabels += [parameter]
        axsm.legend(handles=smhandles, labels=smlabels, loc='upper left')

        axsmbounds = list(axsm.get_position().bounds)
        axsmbounds[2] = axsmbounds[2] - 0.015*len(interres_params)
        axsm.set_position(axsmbounds)

        for [pax, jax, [key, val]] in zip(paramaxes, jacaxes,
                                          interparams.items()):
            if key not in interjacs: continue
            pax.plot(*val, label=key, marker='.', ms=3, lw=0.5)
            pax.legend(loc='upper center')
            jax.plot(interjacs[key][0], interjacs[key][1], label=key,
                     marker='.', ms=3, lw=0.5)
            jax.legend(loc='upper center')

        for key, val in intererrs.items():
            if key == 'abserr':
                axerr.semilogy(val[0],np.abs(val[1]), label=key, marker='.',
                               ms=3, lw=0.5, c='r')
                axerr.legend(ncol=5, loc='upper left')
            if key == 'relerr':
                axrelerr = axerr.twinx()
                axrelerr.semilogy(val[0],np.abs(val[1]), label=key, marker='.',
                                  ms=3, lw=0.5, c='g')
                axrelerr.legend(ncol=5, loc='upper right')

        return f


    def printsig0analysis(self, fit=None,
                          dayrange1=10,
                          dayrange2=1,
                          printcomponents1=True,
                          printcomponents2=True,
                          secondslider=False,
                          printfullt_0=True,
                          printfulldata=True,
                          dB = True,
                          sig0=True,
                          printparamnames=None):
        '''
        A widget to analyze the results of a rt1.rtfits.Fits object.
        (In order to keep the widget responsive, a reference to the returns
        must be kept!)


        Parameters
        ----------
        fit : rt1.rtfits.Fits object
            The used fit-object.
        dayrange1, dayrange2 : int, optional
            The number of consecutive measurements considered by the
            first/second slider. The default is (10 / 1).
        printcomponents1, printcomponents2 : bool, optional
            Indicator if individual backscatter contributions (surface, volume,
            interaction) should be plotted or not. The default is (True, True).
        secondslider : bool, optional
            Indicator if a second slider should be added. The default is False.
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

        Returns
        -------
        list
            a reference to the matplotlib-objects corresponding to:
            [figure, first_slider, (second_slider)].

        '''

        if fit is None:
            fit = self.fit

        # deepcopy the fit-object to avoid altering data
        # TODO avoid copying if possible!
        fit = copy.deepcopy(fit)
        #int_Q = fit.R.int_Q

        if printparamnames is None:
            printparamnames = fit.res_dict.keys()

        # gridspec for incidence-angle and parameter plots
        gs = GridSpec(3, 2, height_ratios=[.65, .25, .1])
        gs.update(top=0.98, left=0.1, right=0.9, bottom=0.025)
        # sub-gridspec for sliders
        gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, :])


        f = plt.figure(figsize=(10, 6))
        # add plot-axes
        ax = f.add_subplot(gs[0,0])
        ax1 = f.add_subplot(gs[0,1])
        ax2 = f.add_subplot(gs[1,:])

        # add slider axes
        slider_ax = f.add_subplot(gs2[0])
        if secondslider:
            slider_bx = f.add_subplot(gs2[1])

        ax.grid()
        ax1.grid()


        # calculate backscatter values
        data = fit.data
        mask = fit.mask
        sig0_vals = fit._calc_model(return_components=True)

        # apply mask and convert to pandas dataframe
        sig0_vals = [np.ma.masked_array(con, mask) for con in sig0_vals]
        sig0_vals = dict(zip(['tot', 'surf', 'vol', 'inter'], sig0_vals))

        sig0_vals['data'] = np.ma.masked_array(data, mask)
        sig0_vals['incs'] = np.ma.masked_array(fit.R.t_0, mask)
        sig0_vals['indexes'] = fit.index

        # convert to sig0 and dB if necessary
        sig0_vals_I_linear = dict()
        for key in ['tot', 'surf', 'vol', 'inter', 'data']:
            if key not in sig0_vals: continue
            sig0_vals[key] = dBsig0convert(sig0_vals[key], sig0_vals['incs'],
                                           dB = dB, sig0=sig0,
                                           fitdB=fit.dB,
                                           fitsig0=fit.sig0)
            sig0_vals_I_linear[key] = dBsig0convert(sig0_vals[key],
                                                    sig0_vals['incs'],
                                                    dB = False, sig0=False,
                                                    fitdB=dB, fitsig0=sig0)
        if printfullt_0 is True:
            inc = np.array([np.deg2rad(np.arange(1, 89, 1))]*len(fit.index))
            newsig0_vals = _evalfit(fit, inc)
            newsig0_vals_I_linear = dict()
            for key in ['tot', 'surf', 'vol', 'inter']:
                if key not in newsig0_vals: continue
                newsig0_vals[key] = dBsig0convert(newsig0_vals[key],
                                                  newsig0_vals['incs'],
                                                  fitdB=fit.dB,
                                                  fitsig0=fit.sig0,
                                                  dB = dB, sig0=sig0)
                newsig0_vals_I_linear[key] = dBsig0convert(
                    newsig0_vals[key], newsig0_vals['incs'], dB = False,
                    sig0=False, fitdB=dB, fitsig0=sig0)


            ax.set_xlim([-2 + np.rad2deg(np.ma.min(newsig0_vals['incs'])),
                          2 + np.rad2deg(np.ma.max(newsig0_vals['incs']))])
            ax.set_ylim([np.min([np.ma.min(newsig0_vals['tot']),
                                 np.ma.min(sig0_vals['data'])]),
                         np.max([np.ma.max(newsig0_vals['tot']),
                                 np.ma.max(sig0_vals['data'])])])
        else:
            ax.set_xlim([-2 + np.rad2deg(np.ma.min(sig0_vals['incs'][0])),
                          2 + np.rad2deg(np.ma.max(sig0_vals['incs'][0]))])
            ax.set_ylim([np.min([np.ma.min(sig0_vals['tot']),
                                 np.ma.min(sig0_vals['data'])]),
                         np.max([np.ma.max(sig0_vals['tot']),
                                 np.ma.max(sig0_vals['data'])])])

        # print full data points in the background
        if printfulldata is True:
            ax.plot(np.rad2deg(sig0_vals['incs']),
                    sig0_vals['data'], lw=0., marker='.', ms=.5,  color = 'k',
                    alpha = 0.5)


        # get upper and lower boundaries for the indicator-lines
        indicator_bounds = [0,1]
        try:
            for key in printparamnames:
                indicator_bounds = [np.min({**fit.res_dict,
                                            **fit.fixed_dict}[key]),
                                    np.max({**fit.res_dict,
                                            **fit.fixed_dict}[key])]
                # if a constant value is plotted, ensure that the boundaries
                # are not equal
                if indicator_bounds[0] == indicator_bounds[1]:
                    indicator_bounds[1] = indicator_bounds[0]*1.1
                    indicator_bounds[0] = indicator_bounds[0]*0.9
                break
        except:
            pass

        # plot parameters as specified in printparamnames
        axparamplot = ax2
        handles, labels = [], []
        i = 0
        pos = 1 - len(printparamnames)//2 * 0.035
        for key in printparamnames:
            try:
                if i > 0:
                    axparamplot = ax2.twinx()
                    axparamplot.tick_params(axis='y', which='both',
                                            labelsize=5, length=2)
                    pos += 0.035
                    axparamplot.spines["right"].set_position(('axes', pos))
                    axparamplot.tick_params(axis='y', which='both',
                                            labelsize=5, length=2)

                l, = axparamplot.plot(fit.index,
                                      {**fit.res_dict,
                                       **fit.fixed_dict}[key],
                                       label = key, color='C' + str(i))
                # add handles and labels to legend
                handles += axparamplot.get_legend_handles_labels()[0]
                labels += axparamplot.get_legend_handles_labels()[1]

                # change color of axis to fit color of lines
                axparamplot.yaxis.label.set_color(l.get_color())
                axparamplot.tick_params(axis='y', colors=l.get_color())
                # shift twin-axes if necessary
                i += 1
            except:
                pass
        axparamplot.legend(handles=handles, labels=labels, loc='upper center',
                           ncol=len(printparamnames))

        ax2.xaxis.set_minor_locator(mpl.dates.MonthLocator())
        ax2.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%m'))
        ax2.xaxis.set_major_locator(mpl.dates.YearLocator())
        ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('\n%Y'))


        # -----------------------------------------------------------
        # plot lines
        def plotlines(dayrange, printcomponents, printfullt_0,
                      styledict_dict, styledict_fullt0_dict):

            incs = np.rad2deg(sig0_vals['incs'])

            lines = []
            for day in np.arange(0, dayrange, 1):
                lines += ax.plot(incs[day], sig0_vals['tot'][day],
                                 **styledict_dict['tot'])
                if printcomponents:
                    lines += ax.plot(incs[day], sig0_vals['surf'][day],
                                     **styledict_dict['surf'])
                    lines += ax.plot(incs[day], sig0_vals['vol'][day],
                                     **styledict_dict['vol'])
                    if fit.R.int_Q is True:
                        lines += ax.plot(incs[day], sig0_vals['inter'][day],
                                         **styledict_dict['inter'])

                lines += ax.plot(incs[day], sig0_vals['data'][day],
                                 **styledict_dict['data'])
                lines += ax2.plot([sig0_vals['indexes'][day]]*2,
                                  indicator_bounds,
                                  **styledict_dict['indicator'])

            lines_frac = []
            for day in np.arange(0, dayrange, 1):
                lintot = sig0_vals_I_linear['tot'][day]
                linsurf = sig0_vals_I_linear['surf'][day]
                linvol = sig0_vals_I_linear['vol'][day]

                if fit.R.int_Q is True:
                    lininter = sig0_vals_I_linear['inter'][day]

                lines_frac += ax1.plot(incs[day], linsurf/lintot,
                                       **styledict_dict['surf'])
                lines_frac += ax1.plot(incs[day], linvol/lintot,
                                       **styledict_dict['vol'])
                if fit.R.int_Q is True:
                    lines_frac += ax1.plot(incs[day], lininter/lintot,
                                           **styledict_dict['inter'])

            # plot full-incidence-angle lines
            linesfull = []
            lines_frac_full = []
            if printfullt_0 is True:
                newincs = np.rad2deg(newsig0_vals['incs'])

                for day in np.arange(0, dayrange, 1):
                    linesfull += ax.plot(np.rad2deg(newsig0_vals['incs'][day]),
                                         newsig0_vals['tot'][day],
                                         **styledict_fullt0_dict['tot'])
                    if printcomponents:
                        linesfull += ax.plot(newincs[day],
                                             newsig0_vals['surf'][day],
                                             **styledict_fullt0_dict['surf'])
                        linesfull += ax.plot(newincs[day],
                                             newsig0_vals['vol'][day],
                                             **styledict_fullt0_dict['vol'])
                        if fit.R.int_Q is True:
                            linesfull += ax.plot(
                                newincs[day], newsig0_vals['inter'][day],
                                **styledict_fullt0_dict['inter'])
                for day in np.arange(0, dayrange, 1):
                    lintot = newsig0_vals_I_linear['tot'][day]
                    linsurf = newsig0_vals_I_linear['surf'][day]
                    linvol = newsig0_vals_I_linear['vol'][day]
                    if fit.R.int_Q is True:
                        lininter = newsig0_vals_I_linear['inter'][day]

                    lines_frac_full += ax1.plot(
                        newincs[day], linsurf/lintot,
                        **styledict_fullt0_dict['surf'])
                    lines_frac_full += ax1.plot(newincs[day],
                                                linvol/lintot,
                                                **styledict_fullt0_dict['vol'])
                    if fit.R.int_Q is True:
                        lines_frac_full += ax1.plot(
                            newincs[day], lininter/lintot,
                            **styledict_fullt0_dict['inter'])

            # add unique legend entries
            ha, la = ax.get_legend_handles_labels()
            unila, wherela = np.unique(la, return_index=True)
            # sort legend entries
            sort_order = dict(data_1=0, data_2=1, total=2,
                              surface=3, volume=4, interaction=5)

            halas = [[ha, la] for ha, la in zip(np.array(la)[wherela],
                                                np.array(ha)[wherela])]
            halas.sort(key=lambda val: sort_order[val[0]])
            ax.legend(handles=[i[1] for i in halas],
                      labels=[i[0] for i in halas])

            return lines, lines_frac, linesfull, lines_frac_full

        # plot first set of lines
        styletot      = {'lw':1, 'marker':'o', 'ms':3, 'color':'k',
                         'label':'total'}
        stylevol   = {'lw':1, 'marker':'o', 'ms':3, 'color':'g',
                      'markerfacecolor':'gray', 'label':'volume'}
        stylesurf  = {'lw':1, 'marker':'o', 'ms':3, 'color':'y',
                      'markerfacecolor':'gray', 'label':'surface'}
        styleinter = {'lw':1, 'marker':'o', 'ms':3, 'color':'c',
                      'markerfacecolor':'gray', 'label':'interaction'}
        styledata  = {'lw':0, 'marker':'s', 'ms':5, 'color':'k',
                      'markerfacecolor':'gray', 'label':'data_1'}
        styleindicator = {'c':'k'}

        stylefullt0tot = {'lw':.25, 'color':'k'}
        stylefullt0vol = {'lw':.25, 'color':'g'}
        stylefullt0surf = {'lw':.25, 'color':'y'}
        stylefullt0inter = {'lw':.25, 'color':'c'}

        styledict_dict = dict(zip(['tot', 'surf', 'vol', 'inter',
                                   'data', 'indicator'],
                                  [styletot, stylesurf, stylevol, styleinter,
                                   styledata, styleindicator]))
        styledict_fullt0_dict = dict(zip(['tot', 'surf', 'vol', 'inter'],
                                  [stylefullt0tot, stylefullt0surf,
                                   stylefullt0vol, stylefullt0inter]))

        lines, lines_frac, linesfull, lines_frac_full = plotlines(
            dayrange1, printcomponents1, printfullt_0, styledict_dict,
            styledict_fullt0_dict)



        if secondslider:
            # plot second set of lines
            styletot       = {'lw':1, 'marker':'o', 'ms':3, 'color':'r',
                              'dashes':[5, 5],  'markerfacecolor':'none'}
            stylevol    = {'lw':1, 'marker':'o', 'ms':3, 'color':'g',
                           'dashes':[5, 5],  'markerfacecolor':'r'}
            stylesurf   = {'lw':1, 'marker':'o', 'ms':3, 'color':'y',
                           'dashes':[5, 5],  'markerfacecolor':'r'}
            styleinter  = {'lw':1, 'marker':'o', 'ms':3, 'color':'c',
                           'dashes':[5, 5],  'markerfacecolor':'r'}
            styledata   = {'lw':0, 'marker':'s', 'ms':5, 'color':'r',
                           'markerfacecolor':'none', 'label':'data_2'}
            styleindicator = {'c':'gray', 'ls':'--'}

            stylefullt0tot = {'lw':.25, 'color':'r', 'dashes':[5, 5]}
            stylefullt0vol = {'lw':.25, 'color':'g', 'dashes':[5, 5]}
            stylefullt0surf = {'lw':.25, 'color':'y', 'dashes':[5, 5]}
            stylefullt0inter = {'lw':.25, 'color':'c', 'dashes':[5, 5]}

            styledict_dict = dict(zip(['tot', 'surf', 'vol', 'inter',
                                       'data', 'indicator'],
                                  [styletot, stylevol, stylesurf, styleinter,
                                   styledata, styleindicator]))
            styledict_fullt0_dict = dict(zip(['tot', 'surf', 'vol', 'inter'],
                                      [stylefullt0tot, stylefullt0vol,
                                       stylefullt0surf, stylefullt0inter]))

            lines2, lines_frac2, linesfull2, lines_frac_full2 = plotlines(
                dayrange2, printcomponents2, printfullt_0, styledict_dict,
                styledict_fullt0_dict)



        # define function to update lines based on slider-input
        def animate(day0, lines, linesfull,
                    lines_frac, lines_frac_full,
                    dayrange, printcomponents,
                    label):

            day0 = int(day0)


            label.set_position([day0, label.get_position()[1]])
            if dayrange == 1:
                label.set_text(
                    sig0_vals['indexes'][day0].strftime('%d. %b %Y %H:%M'))
            elif dayrange > 1:

                lday_0 = sig0_vals['indexes'][day0].strftime('%d. %b %Y %H:%M')
                lday_1 = sig0_vals['indexes'][day0 + dayrange -1
                                              ].strftime('%d. %b %Y %H:%M')
                label.set_text(f"{lday_0} - {lday_1}")


            maxdays = len(sig0_vals['incs'])
            i = 0
            for day in np.arange(day0, day0 + dayrange, 1):
                if day >= maxdays: continue
                lines[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                lines[i].set_ydata(sig0_vals['tot'][day])
                i += 1
                if printcomponents:
                    lines[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                    lines[i].set_ydata(sig0_vals['surf'][day])
                    i += 1
                    lines[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                    lines[i].set_ydata(sig0_vals['vol'][day])
                    if fit.R.int_Q is True:
                       i += 1
                       lines[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                       lines[i].set_ydata(sig0_vals['inter'][day])
                    i += 1

                # update data measurements
                lines[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                lines[i].set_ydata(sig0_vals['data'][day])
                i += 1
                # update day-indicator line
                lines[i].set_xdata([sig0_vals['indexes'][day]]*2)
                i += 1

            i = 0
            for day in np.arange(day0, day0 + dayrange, 1):
                if day >= maxdays: continue
                lintot = sig0_vals_I_linear['tot'][day]
                linsurf = sig0_vals_I_linear['surf'][day]
                linvol = sig0_vals_I_linear['vol'][day]
                if fit.R.int_Q is True:
                    lininter = sig0_vals_I_linear['inter'][day]

                lines_frac[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                lines_frac[i].set_ydata(linsurf/lintot)
                i += 1
                lines_frac[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                lines_frac[i].set_ydata(linvol/lintot)
                if fit.R.int_Q is True:
                    i += 1
                    lines_frac[i].set_xdata(np.rad2deg(sig0_vals['incs'][day]))
                    lines_frac[i].set_ydata(lininter/lintot)
                i += 1

            if printfullt_0 is True:
                i = 0
                for day in np.arange(day0, day0 + dayrange, 1):
                    if day >= maxdays: continue
                    linesfull[i].set_xdata(np.rad2deg(
                        newsig0_vals['incs'][day]))
                    linesfull[i].set_ydata(newsig0_vals['tot'][day])
                    i += 1
                    if printcomponents:
                        linesfull[i].set_xdata(np.rad2deg(
                            newsig0_vals['incs'][day]))
                        linesfull[i].set_ydata(newsig0_vals['surf'][day])
                        i += 1
                        linesfull[i].set_xdata(np.rad2deg(
                            newsig0_vals['incs'][day]))
                        linesfull[i].set_ydata(newsig0_vals['vol'][day])
                        if fit.R.int_Q is True:
                            i += 1
                            linesfull[i].set_xdata(np.rad2deg(
                                newsig0_vals['incs'][day]))
                            linesfull[i].set_ydata(newsig0_vals['inter'][day])
                        i += 1
                i = 0
                for day in np.arange(day0, day0 + dayrange, 1):
                    if day >= maxdays: continue
                    lintot = newsig0_vals_I_linear['tot'][day]
                    linsurf = newsig0_vals_I_linear['surf'][day]
                    linvol = newsig0_vals_I_linear['vol'][day]
                    if fit.R.int_Q is True:
                        lininter = newsig0_vals_I_linear['inter'][day]

                    lines_frac_full[i].set_xdata(np.rad2deg(
                        newsig0_vals['incs'][day]))
                    lines_frac_full[i].set_ydata(linsurf/lintot)
                    i += 1
                    lines_frac_full[i].set_xdata(np.rad2deg(
                        newsig0_vals['incs'][day]))
                    lines_frac_full[i].set_ydata(linvol/lintot)
                    if fit.R.int_Q is True:
                        i += 1
                        lines_frac_full[i].set_xdata(np.rad2deg(
                            newsig0_vals['incs'][day]))
                        lines_frac_full[i].set_ydata(lininter/lintot)
                    i += 1

            return lines

        # define function to update slider-range based on zoom
        def updatesliderboundary(evt, slider):
            indexes = sig0_vals['indexes']
            # Get the range for the new area
            xstart, ystart, xdelta, ydelta = ax2.viewLim.bounds
            xend = xstart + xdelta

            # convert to datetime-objects and ensure that they are in the
            # same time-zone as the sig0_vals indexes
            xend = mpl.dates.num2date(xend).replace(
                tzinfo=sig0_vals['indexes'].tzinfo)
            xstart = mpl.dates.num2date(xstart).replace(
                tzinfo=sig0_vals['indexes'].tzinfo)

            zoomindex = np.where(np.logical_and(indexes > xstart,
                                                indexes < xend))[0]
            slider.valmin = zoomindex[0] - 1
            slider.valmax = zoomindex[-1] + 1

            slider.ax.set_xlim(slider.valmin,
                               slider.valmax)

        # create the slider
        a_slider = Slider(slider_ax,            # axes object for the slider
                          'solid lines',        # name of the slider parameter
                          0,                    # minimal value of parameter
                          len(fit.index) - 1,   # maximal value of parameter
                          valinit=0,            # initial value of parameter
                          valfmt="%i",          # print slider-value as integer
                          valstep=1,
                          closedmax=True)
        a_slider.valtext.set_visible(False)

        slider_ax.set_xticks(np.arange(slider_ax.get_xlim()[0] - 1,
                                       slider_ax.get_xlim()[1] + 1, 1,
                                       dtype=int))
        slider_ax.tick_params(bottom=False, labelbottom=False)
        slider_ax.grid()

        label = slider_ax.text(
            0, 0.5, f"{sig0_vals['indexes'][0].strftime('%d. %b %Y %H:%M')}",
            horizontalalignment='center', verticalalignment='center',
            fontsize=8, bbox=dict(facecolor='w', alpha=0.75,
                                  boxstyle='round,pad=.2'))

        # set slider to call animate function when changed
        a_slider.on_changed(partial(animate,
                                    lines=lines,
                                    linesfull=linesfull,
                                    lines_frac = lines_frac,
                                    lines_frac_full = lines_frac_full,
                                    dayrange=dayrange1,
                                    printcomponents=printcomponents1,
                                    label=label))

        # update slider boundary with respect to zoom of second plot
        ax2.callbacks.connect('xlim_changed', partial(updatesliderboundary,
                                                      slider=a_slider))

        if secondslider:

            # here we create the slider
            b_slider = Slider(slider_bx,         # axes object for the slider
                              'dashed lines',    # name of the slider parameter
                              0,                 # minimal value of parameter
                              len(fit.index) - 1,# maximal value of parameter
                              valinit=0,         # initial value of parameter
                              valfmt="%i",
                              valstep=1,
                              closedmax=True)

            b_slider.valtext.set_visible(False)

            slider_bx.set_xticks(np.arange(slider_bx.get_xlim()[0] - 1,
                                           slider_bx.get_xlim()[1] + 1, 1,
                                           dtype=int))
            slider_bx.tick_params(bottom=False, labelbottom=False)
            slider_bx.grid()

            label2 = slider_bx.text(
                0, 0.5,
                f"{sig0_vals['indexes'][0].strftime('%d. %b %Y %H:%M')}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, bbox=dict(facecolor='w', alpha=0.75,
                                      boxstyle='round,pad=.2'))

            b_slider.on_changed(partial(animate,
                                        lines=lines2,
                                        linesfull=linesfull2,
                                        lines_frac = lines_frac2,
                                        lines_frac_full = lines_frac_full2,
                                        dayrange=dayrange2,
                                        printcomponents=printcomponents2,
                                        label=label2))

            ax2.callbacks.connect('xlim_changed', partial(updatesliderboundary,
                                                          slider=b_slider))


        # !!! a reference to the sliders must be returned in order to
        # !!! remain interactive
        if secondslider:
            return f, a_slider, b_slider
        else:
            return f, a_slider


    def analyzemodel(self, fit=None, set_V_SRF=None, defdict=None, inc=None,
                     labels = dict(), dB=True, sig0=True, int_Q=False,
                     fillcomponents=True):
        '''
        Analyze the range of backscatter for a given model-configuration
        based on the defined parameter-ranges

        Parameters
        ----------
        fit : rt1.rtfits.Fits, optional
            Optionally provide a fits-object from which `set_V_SRF`, `inc` and
            `params['bsf']` will be retrieved if not provided explicitly.
            The default is None.
        set_V_SRF : callable, optional
            A setter function for V and SRF. (see rt1.rtfits.Fits for details)
            The default is None.
        defdict : dict, optional
            A defdict used to define the rt1-configuration.
            (see rt1.rtfits.Fits for details). The default is None.
        inc : array-like, optional
            The incidence-angles to be used in the calculation. The default is None
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

        '''

        if fit is None:
            fit = getattr(self, 'fit', None)

        if fit is not None:
            res_dict = getattr(fit, 'res_dict', None)

            try:
                _fnevals_input = fit.R._fnevals
            except Exception:
                pass

            if defdict is None:
                defdict = fit.defdict

        else:
            _fnevals_input = None

        # number of measurements
        N_param = 100

        inc = np.array(np.deg2rad(np.linspace(1, 89, N_param)))

        # get parameter ranges from defdict and fit
        minparams, maxparams, startparams, fixparams = {}, {}, {}, {}
        for key, val in defdict.items():
            if val[0] is True:
                 minparams[key] = val[3][0][0]
                 maxparams[key] = val[3][1][0]
                 # try to use fitted-values as start values for the parameters
                 if res_dict is not None and key in res_dict:
                     startparams[key] = np.mean(res_dict[key])
                 else:
                     startparams[key] = val[1]
            if val[0] is False:
                if isinstance(val[1], (int, float)):
                         startparams[key] = val[1]
                         fixparams[key] = val[1]
                elif val[1] == 'auxiliary':
                    assert (key in fit.dataset or
                            key in fit.fixed_dict), (f'auxiliary dataset for {key} ' +
                                                     'not found in fit.dataset or ' +
                                                     'fit.fixed_dict')
                    if key in fit.dataset:
                        minparams[key] = fit.dataset[key].min()
                        maxparams[key] = fit.dataset[key].max()
                        startparams[key] = fit.dataset[key].mean()

                    elif key in fit.fixed_dict:
                        minparams[key] = fit.fixed_dict[key].min()
                        maxparams[key] = fit.fixed_dict[key].max()
                        startparams[key] = fit.fixed_dict[key].mean()

        if 'bsf' not in defdict:
            startparams['bsf']  = fit.R.bsf
            fixparams['bsf']  = fit.R.bsf

        modelresult = _getbackscatter(fit=fit,
                                     set_V_SRF=set_V_SRF,
                                     int_Q=int_Q,
                                     inc=inc,
                                     params=startparams,
                                     dB=dB, sig0=sig0,
                                     _fnevals_input = _fnevals_input,
                                     return_fnevals = True)

        _fnevals_input = modelresult.pop('_fnevals')

        f = plt.figure(figsize=(12,9))
        f.subplots_adjust(top=0.93, right=0.98, left=0.07)
                      # generate figure grid and populate with axes
        gs = GridSpec(1 + len(minparams)//2, 1 + 3 ,
                      height_ratios=[8] + [1]*(len(minparams) // 2),
                      width_ratios=[.75, 1, 1, 1])
        gs.update(wspace=.3)

        gsslider = GridSpec(1 + len(minparams)//2, 1 + 3 ,
                            height_ratios=[8] + [1]*(len(minparams) // 2),
                            width_ratios=[.75, 1, 1, 1])
        gsslider.update(wspace=.3, bottom=0.05)

        gsbutton = GridSpec(1 + len(minparams)//2, 1 + 3 ,
                            height_ratios=[8] + [1]*(len(minparams)//2),
                            width_ratios=[.75, 1, 1, 1])
        gsbutton.update(hspace=0.75, wspace=.1, bottom=0.05)


        ax = f.add_subplot(gs[0,0:])
        paramaxes = {}
        col = 0
        for i, key in enumerate(minparams):
            if i%3 == 0: col += 1
            paramaxes[key] = f.add_subplot(gsslider[col, 1 + i%3])

        buttonax = f.add_subplot(gsbutton[1:, 0])
        # hide frame of button-axes
        buttonax.axis('off')
        # add values of fixed parameters
        if len(fixparams) > 0:
            ax.text(.01, .98, 'fixed parameters:\n' +
                    ''.join([f'{key}={round(val, 5)}   ' for
                             key, val in fixparams.items()]))

        # overplot data used in fit
        try:
            ax.plot(fit.R.t_0.T,
                    dBsig0convert(fit.data.T, fit.R.t_0.T,
                                  dB, sig0, fit.dB, fit.sig0), '.', zorder=0)
        except:
            pass

        # plot initial curves
        ltot, = ax.plot(inc, modelresult['tot'], 'k',
                        label = 'total contribution')

        lsurf, = ax.plot(inc, modelresult['surf'], 'b',
                         label = 'surface contribution')

        lvol, = ax.plot(inc, modelresult['vol'], 'g',
                        label = 'volume contribution')

        if int_Q is True:
            lint, = ax.plot(inc, modelresult['inter'], 'y',
                            label = 'interaction contribution')


        if dB is True: ax.set_ylim(-35, 5)
        ax.set_xticks(np.deg2rad(np.arange(5,95, 10)))
        ax.set_xticklabels(np.arange(5,95, 10))

        # a legend for the lines
        leg0 = ax.legend(ncol=4, bbox_to_anchor=(.5, 1.1), loc='upper center')
        # add the line-legend as individual artist
        ax.add_artist(leg0)

        if dB is True and sig0 is True: ax.set_ylabel(r'$\sigma_0$ [dB]')
        if dB is True and sig0 is False: ax.set_ylabel(r'$I/I_0$ [dB]')
        if dB is False and sig0 is True: ax.set_ylabel(r'$\sigma_0$')
        if dB is False and sig0 is False: ax.set_ylabel(r'$I/I_0$')

        ax.set_xlabel(r'$\theta_0$ [deg]')


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

            paramslider[key] = Slider(paramaxes[key], # axes object for the slider
                              keylabel,               # name of the slider
                              minparams[key],         # minimal value
                              maxparams[key],         # maximal value
                              startval,               # initial value
                              #valfmt="%i"            # slider-value as integer
                              color='gray')
            paramslider[key].label.set_position([.05, 0.5])
            paramslider[key].label.set_bbox(dict(boxstyle="round,pad=0.5",
                                                 facecolor='w'))
            paramslider[key].label.set_horizontalalignment('left')
            paramslider[key].valtext.set_position([.8, 0.5])


        buttons = CheckButtons(buttonax, buttonlabels,
                               [False for i in buttonlabels])

        params = startparams.copy()
        # define function to update lines based on slider-input
        def animate(value, key):
            #params = copy.deepcopy(startparams)

            params[key] = value
            modelresult = _getbackscatter(fit=fit, set_V_SRF=set_V_SRF,
                                         int_Q=int_Q, inc=inc,
                                         params=params, dB=dB, sig0=sig0,
                                         _fnevals_input = _fnevals_input)
            # update the data
            ltot.set_ydata(modelresult['tot'].T)
            lsurf.set_ydata(modelresult['surf'].T)
            lvol.set_ydata(modelresult['vol'].T)
            if int_Q is True: lint.set_ydata(modelresult['inter'].T)

            # poverprint boundaries
            hatches = ['//', '\\\ ', '+', 'oo', '--', '..']
            colors = ['C' + str(i) for i in range(10)]
            ax.collections.clear()
            legendhandles = []
            for i, [key_i, key_Q] in enumerate(printvariationQ.items()):
                # replace label of key_i with provided label
                if key_i in labels:
                    keylabel = labels[key_i]
                else:
                    keylabel = key_i

                # reset color of text-backtround
                paramslider[key_i].label.get_bbox_patch().set_facecolor('w')
                if key_Q is True:
                    # set color of text-background to hatch-color
                    #paramslider[key_i].label.set_color(colors[i%len(colors)])
                    paramslider[key_i].label.get_bbox_patch(
                        ).set_facecolor(colors[i%len(colors)])

                    fillparams = params.copy()
                    fillparams[key_i] = minparams[key_i]

                    modelresultmin = _getbackscatter(
                        fit=fit, set_V_SRF=set_V_SRF, int_Q=int_Q, inc=inc,
                        params=fillparams, dB=dB, sig0=sig0,
                        _fnevals_input = _fnevals_input)

                    fillparams[key_i] = maxparams[key_i]

                    modelresultmax = _getbackscatter(
                        fit=fit, set_V_SRF=set_V_SRF, int_Q=int_Q, inc=inc,
                        params=fillparams, dB=dB, sig0=sig0,
                        _fnevals_input = _fnevals_input)

                    legendhandles += [ax.fill_between(
                        inc, modelresultmax['tot'], modelresultmin['tot'],
                        facecolor='none', hatch=hatches[i%len(hatches)],
                        edgecolor=colors[i%len(colors)],
                        label = 'total variability (' + keylabel + ')')]

                    if fillcomponents is True:

                        legendhandles += [ax.fill_between(
                            inc, modelresultmax['surf'], modelresultmin['surf'],
                            color='b', alpha = 0.1,
                            label = 'surf variability (' + keylabel + ')')]

                        legendhandles += [ax.fill_between(
                            inc, modelresultmax['vol'], modelresultmin['vol'],
                            color='g', alpha = 0.1,
                            label = 'vol variability (' + keylabel + ')')]

                        if int_Q is True:
                            legendhandles += [ax.fill_between(
                                inc, modelresultmax['inter'],
                                modelresultmin['inter'], color='y', alpha = 0.1,
                                label = 'int variability (' + keylabel + ')')]

                # a legend for the hatches
                leg1 = ax.legend(handles=legendhandles,
                                 labels=[i.get_label() for i in legendhandles])

                if len(legendhandles) == 0: leg1.remove()

        printvariationQ = {key: False for key in minparams}
        def buttonfunc(label):
            # if labels of the buttons have been changed by the labels-argument
            # set the name to the corresponding key (= the actual parameter name)
            for key, val in labels.items():
                if label == val:
                    label = key

            #ax.collections.clear()
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
        bounds = dict(zip(minparams.keys(),
                          np.array([list(minparams.values()),
                                    list(maxparams.values())]).T))

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

            axbox0 = plt.axes([val.ax.get_position().x0,
                               val.ax.get_position().y1,
                               0.05, 0.025])
            text_box0 = TextBox(axbox0, '', initial=str(round(bounds[key][0], 4)))
            text_box0.on_submit(partial(submit, key=key, minmax=0))


            axbox1 = plt.axes([val.ax.get_position().x1 - 0.05,
                               val.ax.get_position().y1,
                               0.05, 0.025])
            text_box1 = TextBox(axbox1, '', initial=str(round(bounds[key][1], 4)))
            text_box1.on_submit(partial(submit, key=key, minmax=1))


            textboxes_buttons[key + '_min'] = text_box0
            textboxes_buttons[key + '_max'] = text_box1

        textboxes_buttons['buttons'] = buttons

        return f, paramslider, textboxes_buttons

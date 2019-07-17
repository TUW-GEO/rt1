"""
Class for quick visualization of results and used phasefunctions

polarplot() ... plot p and the BRDF as polar-plot

"""

import sympy as sp
import numpy as np
import datetime
try:
    import pandas as pd
except ImportError:
    print('pandas could not be found! ... printsig0timeseries()\
           will not work!')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# plot of 3d scattering distribution
#import mpl_toolkits.mplot3d as plt3d


def rectangularize(array):
    '''
    return a rectangularized (masked array) version of the input-array by
    appending masked values to obtain the smallest possible rectangular shape.

        input-array = [[1,2,3], [1], [1,2]]
        output-array = [[1,2,3], [1,--,--], [1,2,--]]

    Parameters:
    ------------
    array : array-like
            the input-data that is intended to be rectangularized

    Returns:
    ----------
    new_array: np.ma.masked_array
               a rectangularized (masked-array) version of the input-array
    '''
    dim  = len(max(array, key=len))
    newarray = []
    for s in array:
        adddim = dim - len(s)
        if adddim > 0:
            s = np.append(s, np.full(adddim, np.nan))
        newarray +=[s]
    return np.ma.masked_array(newarray, mask=np.isnan(newarray))


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
                    'phi_0', 'phi_s'),
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
                    radapprox = phasefunktapprox(np.pi - ti,
                                                 thetass, 0., 0.)
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
                    ('theta_ex', 'theta_s', 'phi_ex', 'phi_s'),
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
                    radapprox = brdffunktapprox(ti, thetass, 0., 0.)
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

def logmono(inc, Itot=None, Isurf=None, Ivol=None, Iint=None,
            ylim=None, sig0=False, noint=False, fractions=True,
            label=None, label_frac=None):
    """
    Generate a plot of the monostatic backscattered Intensity or sigma_0
    in dB as well as a plot showing the fractional contributions to the
    total signal.

    Parameters
    -----------
    inc : float-array
          Incidence-angle range used for calculating the intensities
    Itot, Ivol, Isurf, Iint : array_like(float)
                              individual monostatic signal contributions
                              i.e. outputs from RT1.calc()  with
                              RT1.geometry = 'mono'  and    RT1.t_0 = inc
                              At least one of the arrays must be provided
                              and it must be of the same length as inc!
    Other Parameters
    -----------------
    ylim : [float , float]
           Manual entry of plot-boundaries as [ymin, ymax]
    sig0 : boolean (default = False)
           Indicator to choose whether sigma_0 (True) or intensity (False)
           is plotted. If True, the relation
           sigma_0 = 4 * Pi * cos(inc) * intensity    is applied.
    noint : boolean (default = False)
            If True, the zero-order contribution
            (i.e. I_tot0 = I_vol + I_surf) is plotted as a dashed line
    fractions : boolean (default = True)
                If True, a plot of the fractional contributions to the
                total signal is generated.
                (i.e. volume_fraction = I_vol / I_tot    etc.)
    label : string
            Manual label of backscatter-plot
    label_frac : string
            Manual label of the fractional contribution-plot

    Returns
    -------
    f : figure
        a matplotlib figure showing a log-plot of the given input-arrays
    """

    # define a mask to avoid log(0) errors (used to crop numpy arrays)
    def mask(x):
        return (x <= 0.)

    assert isinstance(inc, np.ndarray), 'Error, inc must be a numpy array'

    if Itot is not None:
        assert isinstance(Itot, np.ndarray), 'Error, Itot must be a ' + \
            'numpy array'
        assert len(inc) == len(Itot), 'Error: Length of inc and Itot ' + \
            'is not equal'
        Itot = np.ma.array(Itot, mask=mask(Itot))

    if Isurf is not None:
        assert isinstance(Isurf, np.ndarray), 'Error, Isurf must be a ' + \
            'numpy array'
        assert len(inc) == len(Isurf), 'Error: Length of inc and ' + \
            'Isurf is not equal'
        Isurf = np.ma.array(Isurf, mask=mask(Isurf))

    if Ivol is not None:
        assert isinstance(Ivol, np.ndarray), 'Error, Ivol must be a ' + \
            'numpy array'
        assert len(inc) == len(Ivol), 'Error: Length of inc and Ivol' + \
            ' is not equal'
        Ivol = np.ma.array(Ivol, mask=mask(Ivol))

    if Iint is not None:
        assert isinstance(Iint, np.ndarray), 'Error, Iint must be' + \
            ' a numpy array'
        assert len(inc) == len(Iint), 'Error: Length of inc and Iint' + \
            ' is not equal'
        Iint = np.ma.array(Iint, mask=mask(Iint))

    if label is not None:
        assert isinstance(label, str), 'Error, Label must be a string'

    if ylim is not None:
        assert len(ylim) == 2, 'Error: ylim must be an array ' + \
            'of length 2!   ylim = [ymin, ymax]'
    if ylim is not None:
        assert isinstance(ylim[0], (int, float)), 'Error: ymin must' + \
            ' be a number'
    if ylim is not None:
        assert isinstance(ylim[1], (int, float)), 'Error: ymax must' + \
            ' be a number'

    if noint is True:
        assert Isurf is not None, 'Isurf must be provided if noint = True'
        assert Ivol is not None, 'Ivol must be provided if noint = True'

    assert isinstance(sig0, bool), 'Error: sig0 must be True or False'
    assert isinstance(fractions, bool), 'Error: fractions must' + \
        ' be either True or False'

    ctot = 'black'
    csurf = 'red'
    cvol = 'green'
    cint = 'blue'

    if sig0 is True:
        #  I..  will be multiplied with signorm to get sigma0 values
        # instead of normalized intensity
        signorm = 4. * np.pi * np.cos(np.deg2rad(inc))
    else:
        signorm = 1.

    if fractions is True:
        f = plt.figure(figsize=(14, 7))
        ax = f.add_subplot(121)
        ax2 = f.add_subplot(122)
    else:
        f = plt.figure(figsize=(7, 7))
        ax = f.add_subplot(111)

    ax.grid()
    ax.set_xlabel('$\\theta_0$ [deg]')

    if sig0 is True:

        if Itot is not None:
            ax.plot(inc, 10. * np.log10(signorm * Itot),
                    color=ctot, label='$\\sigma_0^{tot}$')
        if Isurf is not None:
            ax.plot(inc, 10. * np.log10(signorm * Isurf),
                    color=csurf, label='$\\sigma_0^{surf}$')
        if Ivol is not None:
            ax.plot(inc, 10. * np.log10(signorm * Ivol),
                    color=cvol, label='$\\sigma_0^{vol}$')
        if Iint is not None:
            ax.plot(inc, 10. * np.log10(signorm * Iint),
                    color=cint, label='$\\sigma_0^{int}$')
        if noint is True:
            ax.plot(inc,
                    10. * np.log10(signorm * (Ivol + Isurf)),
                    color=ctot, linestyle='--',
                    label='$\\sigma_0^{surf}+\\sigma_0^{vol}$')

        if label is None:
            ax.set_title('Bacscattering Coefficient')
        else:
            ax.set_title(label)

        ax.set_ylabel('$\\sigma_0$ [dB]')

    else:

        if Itot is not None:
            ax.plot(inc, 10. * np.log10(signorm * Itot),
                    color=ctot, label='$I_{tot}$')
        if Isurf is not None:
            ax.plot(inc, 10. * np.log10(signorm * Isurf),
                    color=csurf, label='$I_{surf}$')
        if Ivol is not None:
            ax.plot(inc, 10. * np.log10(signorm * Ivol),
                    color=cvol, label='$I_{vol}$')
        if Iint is not None:
            ax.plot(inc, 10. * np.log10(signorm * Iint),
                    color=cint, label='$I_{int}$')
        if noint is True:
            ax.plot(inc, 10. * np.log10(signorm * (Ivol + Isurf)),
                    color=ctot, linestyle='--',
                    label='$I_0^{surf}+I_0^{vol}$')

        if label is None:
            ax.set_title('Normalized Intensity')
        else:
            ax.set_title(label)

        ax.set_ylabel('$I^+$ [dB]')
    legend = ax.legend()
    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_alpha(.5)

    if ylim is None:
        Itotmax = np.nan
        Isurfmax = np.nan
        Ivolmax = np.nan
        Iintmax = np.nan

        # set minimum y to the smallest value of the maximas of
        # the individual contributions -5.
        if Itot is not None:
            Itotmax = np.nanmax(10. * np.log10(signorm * Itot))
        if Isurf is not None:
            Isurfmax = np.nanmax(10. * np.log10(signorm * Isurf))
        if Ivol is not None:
            Ivolmax = np.nanmax(10. * np.log10(signorm * Ivol))
        if Iint is not None:
            Iintmax = np.nanmax(10. * np.log10(signorm * Iint))

        ymin = np.nanmin([Itotmax, Isurfmax, Ivolmax, Iintmax]) - 5.

        # set maximum y to the maximum value of the maximas
        # of the individual contributions + 5.
        ymax = np.nanmax([Itotmax, Isurfmax, Ivolmax, Iintmax]) + 5.

        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(ylim[0], ylim[1])

    if fractions is True:
        # plot fractions
        if Itot is not None and Isurf is not None:
            ax2.plot(inc, Isurf / Itot, label='surface', color=csurf)
        if Itot is not None and Ivol is not None:
            ax2.plot(inc, Ivol / Itot, label='volume', color=cvol)
        if Itot is not None and Iint is not None:
            ax2.plot(inc, Iint / Itot, label='interaction', color=cint)

        if label_frac is None:
            ax2.set_title('Fractional contributions to total signal')
        else:
            ax2.set_title(label_frac)

        ax2.set_xlabel('$\\theta_0$ [deg]')
        if sig0 is True:
            ax2.set_ylabel('$\\sigma_0 / \\sigma_0^{tot}$')
        else:
            ax2.set_ylabel('$I / I_{tot}$')
        ax2.grid()
        legend2 = ax2.legend()
        legend2.get_frame().set_facecolor('w')
        legend2.get_frame().set_alpha(.5)

    plt.show()
    return f

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

def printsig0timeseries(fit,
                        dB=True,
                        sig0=True,
                        params=None,
                        printtot = True,
                        printsurf = True,
                        printvol = True,
                        printint = True,
                        printorig = True,
                        months = None,
                        years = None,
                        ylim=None,
                        printinc = True):
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
         indicator if the plot should display sigma_0 (sig0) or intensity (I)
         The applied relation is: sig0 = 4.*pi*cos(theta) * I
    params: list
            a list of parameter-names that should be overprinted
            (the names must coincide with the arguments of set_V_SRF())
    printtot, printsurf, printvol, printint, printorig : bool
            indicators if the corresponding components should be plotted
    months : list (default = None)
             a list of months to plot (if None, all will be plotted)
    years : list (default = None)
            a list of years to select (if None, all will be plotted)
    ylim : tuple
           a tuple of (ymin, ymax) that will be used as boundaries for the
           y-axis
    printinc : bool (default = True)
               indicator if the incidence-angle dependency should be plotted
               (in a separate plot alongside the timeseries)
    '''
    # get mask
    (_, _, data, _, mask, _, _, _, _) = fit.result
    # get incidence-angles
    inc_array = np.ma.masked_array(fit.result[1].t_0, mask)
    inc = inc_array.compressed()
    # get input dataset
    data = np.ma.masked_array(data, mask)

    def dBsig0convert(val):
        # if results are provided in dB convert them to linear units
        if fit.dB is True: val = 10**(val/10.)
        # convert sig0 to intensity
        if sig0 is False and fit.sig0 is True:
            val = val/(4.*np.pi*np.cos(inc))
        # convert intensity to sig0
        if sig0 is True and fit.sig0 is False:
            val = 4.*np.pi*np.cos(inc)*val
        # if dB output is required, convert to dB
        if dB is True: val = 10.*np.log10(val)
        return val

    res_dict = {}
    # add fitted parameters
    res_dict.update(fit.result[6])
    # add constant values
    res_dict.update(fit.result[-1])

    # calculate individual contributions
    contrib_array = fit._calc_model(R=fit.result[1],
                                    res_dict=res_dict,
                                    return_components=True)

    # apply mask and convert to pandas dataframe
    contrib_array = [np.ma.masked_array(con, mask) for con in contrib_array]
    contrib_array += [data, inc_array]

    contrib = []
    for i, cont in enumerate(contrib_array):
        contrib += [pd.concat([pd.DataFrame(i, index = fit.index) for i in cont.T])[0]]

    contrib = pd.concat(contrib,
                        keys=['tot', 'surf', 'vol', 'inter',
                              '$\\sigma_0$ dataset', 'inc'], axis=1).dropna()

    # convert units
    contrib[['tot', 'surf', 'vol', 'inter',
             '$\\sigma_0$ dataset']] = contrib[[
                     'tot', 'surf', 'vol', 'inter', '$\\sigma_0$ dataset'
                     ]].apply(dBsig0convert)

    # drop unneeded columns
    if fit.result[1].int_Q is False or printint is False:
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
            a=np.rad2deg(rectangularize([x.values for _, x in groupedcontrib['inc']])).T
            b=np.array(rectangularize([x.values for _, x in groupedcontrib[label]])).T
            x = (np.array([a,b]).T)

            l_col = mpl.collections.LineCollection(x,linewidth =.25, label='x',
                                      color=color[label], alpha = 0.5)
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
        paramdf_dict.update(fit.result[6])
        # add constant values
        paramdf_dict.update(fit.result[-1])

        paramdf = pd.DataFrame(paramdf_dict,
                               index = fit.index).sort_index()
        if years is not None:
            paramdf = paramdf.loc[paramdf.index.year.isin(years)]
        if months is not None:
            paramdf = paramdf.loc[paramdf.index.month.isin(months)]

        pax = ax.twinx()
        for k in params:
            pax.plot(paramdf[k], lw=1, marker='.', ms=2)
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



def plot_interres(fit, dynparam = ['SM'],
                  cmaps=['viridis_r', 'RdBu','inferno', 'magma_r']):
    '''
    a function to plot the intermediate-results
    (the data is only available if rtfits.monofit has been called with
    the argument intermediate_results=True)

    Parameters:
    -------------
    fit : rtfits object
          the rtfits-object containing the fit-results

    dynparam : a list of parameter-names that are intended to be plotted
               as timeseries.
    '''
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
    for key in dynparam:
        interres_p = pd.concat([pd.DataFrame(valdict[key],
                                             fit.index.drop_duplicates(),
                                columns=[i])
                                for i, valdict in enumerate(
                                        fit.intermediate_results['parameters'])
                                ], axis=1)
        interres_params[key] = interres_p


    f = plt.figure(figsize=(15,10))
    f.subplots_adjust(top=0.98, left=0.05, right=0.95)
    gs = mpl.gridspec.GridSpec(4, len(interjacs),
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


        cb = mpl.colorbar.ColorbarBase(axcb, cmap=cmap,
                                 orientation = 'vertical',
                                 boundaries = [0] + cbbounds + [cbbounds[-1]+1],
                                 spacing='proportional',
                                 norm=mpl.colors.BoundaryNorm(cbbounds, cmap.N)
                                 )
        if nparam > 0: cb.set_ticks([])

        smhandles += [mpl.lines.Line2D([],[], color=cmap(0))]
        smlabels += [parameter]
    axsm.legend(handles=smhandles, labels=smlabels, loc='upper left')

    axsmbounds = list(axsm.get_position().bounds)
    axsmbounds[2] = axsmbounds[2] - 0.015*len(interres_params)
    axsm.set_position(axsmbounds)

    for [pax, jax, [key, val]] in zip(paramaxes, jacaxes, interparams.items()):
        if key not in interjacs: continue
        pax.plot(*val, label=key, marker='.', ms=3, lw=0.5)
        pax.legend(loc='upper center')
        jax.plot(interjacs[key][0], interjacs[key][1], label=key, marker='.', ms=3, lw=0.5)
        jax.legend(loc='upper center')

    for key, val in intererrs.items():
        if key == 'abserr':
            axerr.semilogy(val[0],np.abs(val[1]), label=key, marker='.', ms=3, lw=0.5, c='r')
            axerr.legend(ncol=5, loc='upper left')
        if key == 'relerr':
            axrelerr = axerr.twinx()
            axrelerr.semilogy(val[0],np.abs(val[1]), label=key, marker='.', ms=3, lw=0.5, c='g')
            axrelerr.legend(ncol=5, loc='upper right')



def printresults(fit, truevals=None, startvals=False,
                 legends=True, result_selection='all'):
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
    ---------
    fig : matplotlib.figure object
    '''
    # this is done to allow the usage of monofit-outputs as well
    (res_lsq, R, data, inc, mask, weights,
     res_dict, start_dict, fixed_dict) = fit.result

    if result_selection == 'all':
        result_selection = range(len(data))

    # assign colors
    colordict = {key:f'C{i%10}' for i, key in enumerate(res_dict.keys())}

    # reset incidence-angles in case they have been altered beforehand
    R.t_0 = inc
    R.p_0 = np.zeros_like(inc)

    # evaluate number of measurements
    Nmeasurements = len(inc)

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
    for i, j in enumerate(np.ma.masked_array(data, mask)[result_selection]):
        ax.plot(inc[result_selection[i]], j, '.')

    # reset color-cycle
    plt.gca().set_prop_cycle(None)

    # define incidence-angle range for plotting
    incplot = np.array([np.linspace(np.min(inc), np.max(inc), 100)]
                       * Nmeasurements)
    # set new incidence-angles
    R.t_0 = incplot
    R.p_0 = np.zeros_like(incplot)

    # get parameter-values
    # for python > 3.4
    # calc_dict = dict(**res_dict, **fixed_dict)
    calc_dict = {**res_dict, **fixed_dict}

    # calculate results
    fitplot = fit._calc_model(R, calc_dict)

    # generate a mask that hides all measurements where no data has
    # been provided (i.e. whose parameter-results are still the startvals)
    newmask = np.ones_like(incplot) * np.all(mask, axis=1)[:, np.newaxis]
    fitplot = np.ma.masked_array(fitplot, newmask)

    for i, val in enumerate(fitplot[result_selection]):
        ax.plot(incplot[i], val, alpha=0.4, label=result_selection[i])

    # ----------- plot start-values ------------
    if startvals is True:
        startplot = fit._calc_model(R, {**start_dict, **fixed_dict})
        for i, val in enumerate(startplot[result_selection]):
            if i == 0:
                label = 'fitstart'
            else:
                label = ''
            ax.plot(incplot[result_selection[i]], val, 'k--', linewidth=1,
                    alpha=0.5, label=label)

    if legends is True:
        ax.legend(loc=1)

    mintic = np.round(np.rad2deg(np.min(inc)) + 4.9, -1)
    if mintic < 0.:
        mintic = 0.
    maxtic = np.round(np.rad2deg(np.max(inc)) + 4.9, -1)
    if maxtic > 360.:
        maxtic = 360.

    ticks = np.arange(np.rad2deg(np.min(inc)),
                      np.rad2deg(np.max(inc)) + 1.,
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
            param_errs[key] = res_dict[key] - truevals[key]

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
    for key in res_dict:
        ax2.plot(fit.index,
                 np.ma.masked_array(res_dict[key], np.all(mask, axis=1)),
                 alpha=1., label=key, color=colordict[key])

    if len(result_selection) < len(data):
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



def printerr(fit, newcalc=False, relative=False, result_selection='all'):
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
    '''

    (res_lsq, R, data, inc, mask, weights,
     res_dict, start_dict, fixed_dict) = fit.result

    if result_selection == 'all':
        result_selection = range(len(data))

    Nmeasurements = len(inc)

    if newcalc is False:
        # get residuals from fit into desired shape for plotting
        # Attention -> incorporate weights and mask !
        res = np.ma.masked_array(np.reshape(res_lsq.fun, data.shape), mask)

        if relative is True:
            res = np.ma.abs(res / (res + np.ma.masked_array(data, mask)))
        else:
            res = np.ma.abs(res)
    else:
        # Alternative way of calculating the residuals
        # (based on R, inc and res_dict)

        R.t_0 = inc
        R.p_0 = np.zeros_like(inc)

        # for python > 3.4
        # calc_dict = dict(**res_dict, **fixed_dict)
        calc_dict = dict((k, v) for k, v in list(res_dict.items())
                         + list(fixed_dict.items()))

        estimates = fit._calc_model(R, calc_dict)
        # calculate the residuals based on masked arrays
        masked_estimates = np.ma.masked_array(estimates, mask=mask)
        masked_data = np.ma.masked_array(data, mask=mask)

        res = np.ma.sqrt((masked_estimates - masked_data)**2)

        if relative is True:
            res = res / masked_estimates

    # apply mask to data and incidence-angles (and convert to degree)
    inc = np.ma.masked_array(np.rad2deg(inc), mask=mask)
    data = np.ma.masked_array(data, mask=mask)

    # make new figure
    figres = plt.figure(figsize=(14, 10))
    axres = figres.add_subplot(212)
    if relative is True:
        axres.set_title('Mean relative residual per measurement')
    else:
        axres.set_title('Mean absolute residual per measurement')

    axres2 = figres.add_subplot(211)
    if relative is True:
        axres2.set_title('Relative residuals per incidence-angle')
    else:
        axres2.set_title('Residuals per incidence-angle')

    # the use of masked arrays might cause python 2 compatibility issues!
    axres.plot(fit.index[result_selection], res[result_selection], '.', alpha=0.5)

    # plot mean residual for each measurement
    axres.plot(fit.index[result_selection], np.ma.mean(res[result_selection], axis=1),
               'k', linewidth=3, marker='o', fillstyle='none')

    # plot total mean of mean residuals per measurement
    axres.plot(fit.index[result_selection],
               [np.ma.mean(np.ma.mean(res[result_selection], axis=1))] * len(result_selection),
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

    handles, labels = axres.get_legend_handles_labels()
    axres.legend(handles=handles + [res_h_dots] + [res_h] + [res_h_dash],
                 loc=1)

    axres.set_ylabel('Residual')

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
        axres2.plot(inc[result_selection[i]][sortpattern], resval[sortpattern],
                    ':', alpha=0.5, marker='.')

    # plot mean residual per incidence-angle
    axres2.plot(meanincs, mean,
                'k', linewidth=3, marker='o', fillstyle='none')

    # add some legends
    res_h2 = mlines.Line2D(
        [], [], color='black', label='Mean res.  per inc-angle',
        linestyle='-', linewidth=3, marker='o', fillstyle='none')
    res_h_lines = mlines.Line2D(
        [], [], color='black', label='Residuals',
        linestyle=':', alpha=0.5)

    handles2, labels2 = axres2.get_legend_handles_labels()
    axres2.legend(handles=handles2 + [res_h_lines] + [res_h2], loc=1)

    axres2.set_xlabel('$\\theta_0$ [deg]')
    axres2.set_ylabel('Residual')

    # find minimum and maximum incidence angle
    maxinc = np.max(inc)
    mininc = np.min(inc)

    axres2.set_xlim(np.floor(mininc) - 1,
                    np.ceil(maxinc) + 1)

    # set major and minor ticks
    axres2.xaxis.set_major_locator(plt.MultipleLocator(1))
    axres2.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    axres2.xaxis.set_minor_locator(plt.MultipleLocator(.25))

    # set ticks
    if isinstance(fit.index[0], datetime.datetime):
        plt.setp(axres.get_xticklabels(), rotation=30, ha='right')

    figres.tight_layout()

    return figres
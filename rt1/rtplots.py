"""
Class for quick visualization of results and used phasefunctions

polarplot() ... plot p and the BRDF as polar-plot

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# plot of 3d scattering distribution
import mpl_toolkits.mplot3d as plt3d
from .scatter import Scatter


class Plots(Scatter):
    """
    The class provides pre-defined plotting-modules to easily visualize the
    used definitions and results.
    """

    def __init__(self, **kwargs):
        pass

    def polarplot(self, R=None, SRF=None, V=None, incp=[15., 35., 55., 75.],
                  incBRDF=[15., 35., 55., 75.], pmultip=2., BRDFmultip=1.,
                  plabel='Volume-Scattering Phase Function',
                  BRDFlabel='Surface-BRDF', paprox=True, BRDFaprox=True,
                  plegend=True, plegpos=(0.75, 0.5), BRDFlegend=True,
                  BRDFlegpos=(0.285, 0.5), groundcolor="none"):
        """
        Generation of polar-plots of the volume- and the surface scattering
        phase function as well as the used approximations in terms of
        legendre-polynomials.


        Parameters
        -----------
        R : RT1-class object
            If R is provided, SRF and V are taken from it
            as V = R.RV and SRF = R.SRF
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
            V = R.RV

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
            for V in V:
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

                pmax = pmultip * np.max(V.p(plottis, np.pi - plottis, 0., 0.))

                if plegend is True:
                    legend_lines = []

                # set color-counter to 0
                i = 0
                for ti in plottis:
                    color = colors[i]
                    i = i + 1
                    thetass = np.arange(0., 2. * np.pi, .01)
                    rad = V.p(ti, thetass, 0., 0.)
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
            for SRF in SRF:
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

                brdfmax = BRDFmultip * np.max(SRF.brdf(plottis,
                                                       plottis, 0., 0.))

                # set color-counter to 0
                i = 0
                for ti in plottis:
                    color = colors[i]
                    i = i + 1
                    thetass = np.arange(-np.pi / 2., np.pi / 2., .01)
                    rad = SRF.brdf(ti, thetass, 0., 0.)
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

    def logmono(self, inc, Itot=None, Isurf=None, Ivol=None, Iint=None,
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

    def linplot3d(self, theta, phi, Itot=None, Isurf=None, Ivol=None,
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

    def hemreflect(self, R=None, SRF=None, phi_0=0., t_0_step=5., t_0_min=0.,
                   t_0_max=90., simps_N=1000, showpoints=True,
                   returnarray=False):
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
            NormBRDF = R.SRF.NormBRDF
        elif SRF is not None:
            BRDF = SRF.brdf
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
        for theta_0 in np.deg2rad(incnum):
            # define the function that has to be integrated
            # (i.e. Eq.20 in the paper)
            # notice the additional  np.sin(thetas)  which oritinates from
            # integrating over theta_s instead of mu_s
            def integfunkt(theta_s, phi_s):
                return np.sin(theta_s) * np.cos(theta_s) * BRDF(theta_0,
                                                                theta_s,
                                                                phi_0, phi_s)
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

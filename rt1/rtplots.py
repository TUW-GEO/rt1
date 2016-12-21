"""
Class for quick visualization of results and used phasefunctions

polarplot() ... plot p and the BRDF as polar-plot

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
#plot of 3d scattering distribution
import mpl_toolkits.mplot3d as plt3d
from scatter import Scatter

class Plots(Scatter):
    """
    Parameters:
        R ... RT1 class object
    """
    def __init__(self, **kwargs):
        pass
    

    def polarplot(self, R = None, SRF = None, V = None, incp = [15.,35.,55.,75.], incBRDF = [15.,35.,55.,75.], pmultip = 2., BRDFmultip = 1. , plabel = 'Volume-Scattering Phase Function', BRDFlabel = 'Surface-BRDF', paprox = True, BRDFaprox = True):
        """
        generate a polar-plot of the volume- and the surface scattering phase function
        together or in separate plots. 
        The used approximation in terms of legendre-polynomials is also plotted as dashed line.
        
        
        Parameters:
        
            R	   ... a RT1 object 
            SRF     ... a rt1.surface module
            V       ... a rt1.volume module
        
        Optional Parameters:
            incp        ... incidence-angles [deg] at which the volume-scattering phase-function will be plotted 
            incBRDF     ... incidence-angles [deg] at which the BRDF will be plotted
        
            plabel      ... label for the plot of p
            BRDFlabel   ... label for the plot of the BRDF
                     
            paprox      ... True/False whether to print the approximation of p or not
            BRDFaprox   ... True/False whether to print the approximation of the BRDF or not
            
            pmultip     ... multiplicator of max-plotrange for p     (in case the chosen plotranges are not satisfying)
            BRDFmultip  ... multiplicator of max-plotrange for BRDF
        
            max-plotrange for p is given by the max. value of p in forward-direction (for the chosen incp)
            and for the BRDF by the max. value of the BRDF in specular direction (for the chosen incBRDF)
        """
        
       
        assert isinstance(incp,list), 'Error: incidence-angles for polarplot of p must be a list'
        assert isinstance(incBRDF,list), 'Error: incidence-angles for polarplot of the BRDF must be a list'
        for i in incBRDF: assert i<=90, 'ERROR: the incidence-angle of the BRDF in polarplot must be < 90'
        
        assert isinstance(pmultip,float), 'Error: plotrange-multiplier for polarplot of p must be a floating-point number'
        assert isinstance(BRDFmultip,float), 'Error: plotrange-multiplier for polarplot of the BRDF must be a floating-point number'

        assert isinstance(plabel,str), 'Error: plabel of polarplot must be a string'
        assert isinstance(BRDFlabel,str), 'Error: plabel of polarplot must be a string'



        if R==None and SRF==None and V==None:
            assert False, 'Error: You must either provide R or SRF and/or V'
            
                
        # if R is provided, use it to define SRF and V, else use the provided functions
        if R !=None:
            SRF = R.SRF
            V = R.RV
            
        # define functions for plotting that evaluate the used approximations in terms of legendre-polynomials
        n = sp.Symbol('n')        
        


        
        
        
        if V != None:
            # if V is a scalar, make it a list
            if np.ndim(V)==0:
                V = [V]

            # make new figure
            if SRF == None:
                # if SRF is None, plot only a single plot of p
                polarfig = plt.figure(figsize=(7,7))
                polarax = polarfig.add_subplot(111,projection='polar')
            else:
                # plot p and the BRDF together
                polarfig = plt.figure(figsize=(14,7))
                polarax = polarfig.add_subplot(121,projection='polar')

            # plot of volume-scattering phase-function's
            for V in V:
                # define a plotfunction of the analytic form of p
                phasefunkt = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), V._func,"numpy") 
                # define a plotfunction of the legendre-approximation of p
                if paprox == True: phasefunktapprox = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), sp.Sum(V.legcoefs*sp.legendre(n,self.thetap('theta_i','theta_s','phi_i','phi_s', V.a)),(n,0,V.ncoefs-1)).doit(),"numpy") 
                   
                # set incidence-angles for which p is calculated
                plottis=np.deg2rad(incp)
                colors = ['k','r','g','b','c','m','y']*(len(plottis)/7+1)
                # reset color-counter
                i=0
                
                pmax = pmultip*np.max(phasefunkt(plottis, np.pi-plottis, 0., 0.))
                
                for ti in plottis:
                    color = colors[i]
                    i=i+1
                    thetass = np.arange(0.,2.*np.pi,.01)
                    rad= [phasefunkt(ti, ts, 0., 0.) for ts in thetass]
                    if paprox == True: radapprox = phasefunktapprox(ti, thetass, 0., 0.)
                    
                    polarax.set_theta_direction(-1)   # set theta direction to clockwise
                    polarax.set_theta_offset(np.pi/2.) # set theta to start at z-axis
                    
                    polarax.plot(thetass,rad, color)
                    if paprox == True: polarax.plot(thetass,radapprox, color+'--')
                    polarax.arrow(-ti,pmax*1.2  ,0.,-pmax*0.8, head_width = .0, head_length=.0, fc = color, ec = color, lw=1, alpha=0.3)
                    polarax.fill_between(thetass,rad,alpha=0.2, color=color)
                    polarax.set_xticklabels(['$0^\circ$','$45^\circ$','$90^\circ$','$135^\circ$','$180^\circ$'])
                    polarax.set_yticklabels([])
                    polarax.set_rmax(pmax*1.2)
                    polarax.set_title(plabel + '\n')
            
   
                
        if SRF !=None:
            # if SRF is a scalar, make it a list
            if np.ndim(SRF)==0:
                SRF = [SRF]

            # append to figure or make new figure
            if V == None:
                # if V is None, plot only a single plot of the BRDF
                polarfig = plt.figure(figsize=(7,7))
                polarax = polarfig.add_subplot(111,projection='polar')
            else: 
                # plot p and the BRDF together
                polarax = polarfig.add_subplot(122,projection='polar')



            # plot of BRDF
            for SRF in SRF:
                # define a plotfunction of the analytic form of the BRDF
                brdffunkt = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), SRF._func,"numpy") 
                # define a plotfunction of the analytic form of the BRDF
                if BRDFaprox == True: brdffunktapprox = sp.lambdify(('theta_i', 'theta_s', 'phi_i', 'phi_s'), sp.Sum(SRF.legcoefs*sp.legendre(n,self.thetaBRDF('theta_i','theta_s','phi_i','phi_s', SRF.a)),(n,0,SRF.ncoefs-1)).doit(),"numpy") 
          
                
                # set incidence-angles for which the BRDF is calculated
                plottis=np.deg2rad(incBRDF)
                colors = ['k', 'r','g','b', 'c','m','y']*(len(plottis)/7+1)
                i=0
                
                brdfmax = BRDFmultip*np.max(brdffunkt(plottis, plottis, 0., 0.))
                
                for ti in plottis:
                    color = colors[i]
                    i=i+1
                    thetass = np.arange(-np.pi/2.,np.pi/2.,.01)
                    rad=[brdffunkt(ti, ts, 0., 0.) for ts in thetass]
                    if BRDFaprox == True: radapprox = brdffunktapprox(ti, thetass, 0., 0.)
                    
                    polarax.set_theta_direction(-1)   # set theta direction to clockwise
                    polarax.set_theta_offset(np.pi/2.) # set theta to start at z-axis
                    
                    polarax.plot(thetass,rad, color)
                    if BRDFaprox == True: polarax.plot(thetass,radapprox, color + '--')
                    polarax.fill(np.arange(np.pi/2.,3.*np.pi/2.,.01),np.ones_like(np.arange(np.pi/2.,3.*np.pi/2.,.01))*brdfmax*1.2,'k')
                
                
                    polarax.arrow(-ti,brdfmax*1.2  ,0.,-brdfmax*0.8, head_width =.0, head_length=.0, fc = color, ec = color, lw=1, alpha=0.3)
                    polarax.fill_between(thetass,rad,alpha=0.2, color=color)
                    polarax.set_xticklabels(['$0^\circ$','$45^\circ$','$90^\circ$'])
                    polarax.set_yticklabels([])
                    polarax.set_rmax(brdfmax*1.2)
                    polarax.set_title(BRDFlabel + '\n')
        return polarfig


        
        


        
        
    def logmono(self, inc, Itot=None, Isurf= None, Ivol = None, Iint=None, ylim = None, sig0=False, fractions = True, label = None):
        """
        plot either backscattered intensity or sigma_0 in dB
        
        Parameters:
             
        inc ...     incidence-angle range used for calculating the intensities
                    (array)
           
        Itot, Ivol, Isurf, Iint ... individual signal contributions, i.e. outputs from RT1.calc()
                    (array's of same length as inc)        
        
        sig0 ... If True, sigma0 will be plotted which is related to I via:  sigma_0 = 4 Pi cos(inc) * I             
                    (True, False)
                    
        ylim ... manual entry of plot-boundaries as [ymin, ymax]
                
        label ...   manual label of plot        
        """
        
        assert isinstance(inc,np.ndarray), 'Error, inc must be a numpy array'
    
        if Itot is not None: assert isinstance(Itot,np.ndarray), 'Error, Itot must be a numpy array'
        if Itot is not None: assert len(inc) == len(Itot), 'Error: Length of inc and Itot is not equal'

        if Isurf is not None: assert isinstance(Isurf,np.ndarray), 'Error, Isurf must be a numpy array'
        if Isurf is not None: assert len(inc) == len(Isurf), 'Error: Length of inc and Isurf is not equal'

        if Ivol is not None: assert isinstance(Ivol,np.ndarray), 'Error, Ivol must be a numpy array'
        if Ivol is not None: assert len(inc) == len(Ivol), 'Error: Length of inc and Ivol is not equal'
        
        if Iint is not None: assert isinstance(Iint,np.ndarray), 'Error, Iint must be a numpy array'
        if Iint is not None: assert len(inc) == len(Iint), 'Error: Length of inc and Iint is not equal'
        
        if label is not None:assert isinstance(label,str), 'Error, Label must be a string'
     
        
        if ylim is not None: assert len(ylim)!=2, 'Error: ylim must be an array of length 2!   ylim = [ymin, ymax]'
        if ylim is not None: assert isinstance(ylim[0],(int,float)), 'Error: ymin must be a number'
        if ylim is not None: assert isinstance(ylim[1],(int,float)), 'Error: ymax must be a number'
        
        assert isinstance(sig0,bool), 'Error: sig0 must be either True or False'
        assert isinstance(fractions,bool), 'Error: fractions must be either True or False'


        ctot='black'
        csurf='red'
        cvol='green'
        cint='blue'      
            
        
        if sig0 == True:
            #  I..  will be multiplied with sig0  to get sigma0 values instead of normalized intensity
            signorm = 4.*np.pi*np.cos(np.deg2rad(inc))
        else:
            signorm = 1.
        
        if fractions == True:
            f = plt.figure(figsize=(14,7))
            ax = f.add_subplot(121)
            ax2 = f.add_subplot(122)
        else:
            f = plt.figure(figsize=(7,7))
            ax = f.add_subplot(111)

        ax.grid()
        ax.set_xlabel('$\\theta_0$ [deg]')
        
        if sig0 == True:
            if Itot is not None: ax.plot(inc, 10.*np.log10(signorm*Itot), color=ctot, label='$\\sigma_0^{tot}$')
            if Isurf is not None: ax.plot(inc, 10.*np.log10(signorm*Isurf), color=csurf, label='$\\sigma_0^{surf}$')
            if Ivol is not None: ax.plot(inc, 10.*np.log10(signorm*Ivol), color=cvol, label='$\\sigma_0^{vol}$')
            if Iint is not None: ax.plot(inc, 10.*np.log10(signorm*Iint), color=cint, label='$\\sigma_0^{int}$')
            
            if label == None:
                ax.set_title('Bacscattering Coefficient')
            else:
                ax.set_title(label)
            
            ax.set_ylabel('$\\sigma_0$ [dB]')

                
        else:
            if Itot is not None: ax.plot(inc, 10.*np.log10(signorm*Itot), color=ctot, label='$I_{tot}$')
            if Isurf is not None: ax.plot(inc, 10.*np.log10(signorm*Isurf), color=csurf, label='$I_{surf}$')
            if Ivol is not None: ax.plot(inc, 10.*np.log10(signorm*Ivol), color=cvol, label='$I_{vol}$')
            if Iint is not None: ax.plot(inc, 10.*np.log10(signorm*Iint), color=cint, label='$I_{int}$')
            
            if label == None:
                ax.set_title('Normalized Intensity')
            else:
                ax.set_title(label)
            
            ax.set_ylabel('$I^+$ [dB]')
        ax.legend()
        
        if ylim == None:
            if Itot is not None and Iint is not None: ax.set_ylim(np.nanmax(10.*np.log10(signorm*Iint))-5.,np.nanmax(10.*np.log10(signorm*Itot))+5)
        else:
            ax.set_ylim(ylim[0],ylim[1])
        
        if fractions == True:    
            # plot fractions
            if Itot is not None and Isurf is not None: ax2.plot(inc, Isurf/Itot, label='surface', color=csurf)
            if Itot is not None and Ivol is not None: ax2.plot(inc, Ivol/Itot, label='volume', color=cvol)
            if Itot is not None and Iint is not None: ax2.plot(inc, Iint/Itot, label='interaction', color=cint)
            ax2.set_title('Fractional contributions to total signal')
            ax2.set_xlabel('$\\theta_0$ [deg]')
            if sig0 == True:
                ax2.set_ylabel('$\\sigma_0 / \\sigma_0^{tot}$')
            else:
                ax2.set_ylabel('$I / I_{tot}$')
            ax2.grid()
            ax2.legend()
        
        plt.show()
        return f

        
        
        
        
        
        
    def linplot3d(self, theta, phi, Itot3d=None, Isurf3d=None, Ivol3d=None, Iint3d=None, surfmultip = 1., zoom = 2.):
        """
        Input arrays must be numpy arrays with dtype = float and must have the following shape:
            
        Parameters:
            theta = [...]
            phi   = [...]            
            I...  = [ [theta[0], phi[:] ] , [theta[1], phi[:] ] , ... , [theta[N], phi[:] ]]
            
            
        Optional Parameters:
            surfmultip ... scaling factor for the plotted surface
            zoom       ... scaling factor for the whole plot
        """

        assert isinstance(theta,np.ndarray), 'Error: theta must be a numpy-array'
        assert isinstance(phi,np.ndarray), 'Error: phi must be a numpy-array'
        if Itot3d is not None: assert isinstance(Itot3d,np.ndarray), 'Error: Itot3d must be a numpy-array'
        if Isurf3d is not None: assert isinstance(Isurf3d,np.ndarray), 'Error: Isurf3d must be a numpy-array'
        if Ivol3d is not None: assert isinstance(Ivol3d,np.ndarray), 'Error: Ivol3d must be a numpy-array'
        if Iint3d is not None: assert isinstance(Iint3d,np.ndarray), 'Error: Iint3d must be a numpy-array'
        assert isinstance(surfmultip, float), 'Error: surfmultip must be a floating-point number'
        assert surfmultip>0., 'Error: surfmultip must be larger than 0.'

        assert isinstance(zoom, float), 'Error: zoom must be a floating-point number'
        assert zoom>0., 'Error: zoom must be larger than 0.'




        
        # make plot-grid of theta/phi
        THETA,PHI = np.meshgrid(theta,phi)
        
        # transform values to spherical coordinate system
        def sphericaltransform(r):
            if r is None:
                return None
                
            X = r * np.sin(THETA) * np.cos(PHI)
            Y = r * np.sin(THETA) * np.sin(PHI)
            Z = r * np.cos(THETA)
            return X,Y,Z
        
                        
        fig = plt.figure(figsize=(7,7))
        ax3d = fig.add_subplot(1,1,1,projection='3d')
        
        #ax3d.view_init(elev=20.,azim=45)
        
        # calculate maximum value of all given imput-arrays        
        m = []
        if Itot3d is not None: m = m + [np.max(sphericaltransform(Itot3d)), np.abs(np.min(sphericaltransform(Itot3d)))]
        if Isurf3d is not None: m = m + [np.max(sphericaltransform(Isurf3d)), np.abs(np.min(sphericaltransform(Isurf3d)))]
        if Ivol3d is not None: m = m + [np.max(sphericaltransform(Ivol3d)), np.abs(np.min(sphericaltransform(Ivol3d)))]
        if Iint3d is not None: m = m + [np.max(sphericaltransform(Iint3d)), np.abs(np.min(sphericaltransform(Iint3d)))]
        maximum = np.max(m)
        
        
        
        xx=np.array([- surfmultip*maximum, surfmultip*maximum ])
        yy=np.array([0., surfmultip*maximum])
        xxx,yyy = np.meshgrid(xx,yy)
        zzz = np.ones_like(xxx)*(0.)
        
        ax3d.plot_surface(xxx,yyy,zzz, alpha=0.2, color='k')

        
        if Itot3d is not None: 
            ax3d.plot_surface(
            sphericaltransform(Itot3d)[0],sphericaltransform(Itot3d)[1],sphericaltransform(Itot3d)[2] ,rstride=1, cstride=1, color='Gray',
            linewidth=0, antialiased=True, alpha=.3)
        
        if Isurf3d is not None:
            ax3d.plot_surface(
            sphericaltransform(Isurf3d)[0],sphericaltransform(Isurf3d)[1],sphericaltransform(Isurf3d)[2], rstride=1, cstride=1, color='Red',
            linewidth=0, antialiased=True, alpha=.5)
        
        if Ivol3d is not None:
            ax3d.plot_surface(
            sphericaltransform(Ivol3d)[0],sphericaltransform(Ivol3d)[1],sphericaltransform(Ivol3d)[2], rstride=1, cstride=1, color='Green',
            linewidth=0, antialiased=True, alpha=.5)
        
        if Iint3d is not None:
            ax3d.plot_surface(
            sphericaltransform(Iint3d)[0],sphericaltransform(Iint3d)[1],sphericaltransform(Iint3d)[2], rstride=1, cstride=1, color='Blue',
            linewidth=0, antialiased=True, alpha=.5)
        
        ax3d.w_xaxis.set_pane_color((1.,1.,1.,0.))
        ax3d.w_xaxis.line.set_color((1.,1.,1.,0.))
        ax3d.w_yaxis.set_pane_color((1.,1.,1.,0.))
        ax3d.w_yaxis.line.set_color((1.,1.,1.,0.))
        #ax3d.w_zaxis.set_pane_color((0.,0.,0.,.1))
        ax3d.w_zaxis.set_pane_color((1.,1.,1.,.0))
        ax3d.w_zaxis.line.set_color((1.,1.,1.,0.))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        
        
        ax3d.set_xlim(-maximum/zoom, maximum/zoom)
        ax3d.set_ylim(-maximum/zoom,maximum/zoom)
        ax3d.set_zlim(0,2*maximum/zoom)
        
        
        # ensure display of correct aspect ratio (bug in mplot3d)
        # due to the bug it is only possible to have equally sized axes (without changing the mplotlib code itself)
        
        #ax3d.auto_scale_xyz([np.min(sphericaltransform(Itot3d)),np.max(sphericaltransform(Itot3d))],
        #                   [0.,np.max(sphericaltransform(Itot3d))+np.abs(np.min(sphericaltransform(Itot3d)))],
        #                     [-np.max(sphericaltransform(Itot3d))+np.abs(np.min(sphericaltransform(Itot3d))),np.max(sphericaltransform(Itot3d))+np.abs(np.min(sphericaltransform(Itot3d)))])
         
        plt.show()
        return fig
        
        
        
        
        
        
        
        
        
        
        
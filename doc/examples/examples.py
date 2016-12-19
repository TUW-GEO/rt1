"""
Reproduce examples like given in the paper
"""

import sys
sys.path.append('..')


import matplotlib.pyplot as plt
import numpy as np
import timeit

from rt1.rt1 import RT1

from rt1.rtplots import Plots


from rt1.volume import Rayleigh
from rt1.volume import HenyeyGreenstein
from rt1.volume import HGRayleigh
from rt1.volume import DoubleHG


from rt1.surface import CosineLobe
from rt1.surface import LafortuneLobe
from rt1.surface import HenyeyGreenstein as HenyeyGreensteinsurface

from rt1.rt1 import RT1



# EVALUATION OF BISTATIC PLOTS
# if set to true, 3dplots will be generated,
# otherwise the code is stopped after monostatic evaluation
bistaticplot = False


#plt.close('all')


# start timer
tic = timeit.default_timer()

I0=1.
inc = np.arange(0.,90.,2.)

if True:
    # Example 1
    V = Rayleigh(tau=0.7, omega=0.3)
    SRF = CosineLobe(ncoefs=10, i=5)
    label = 'Example 1'
else:
    V = HenyeyGreenstein(tau=0.7, omega=0.3, t=0.7, ncoefs=20)
    SRF = CosineLobe(ncoefs=10, i=5)
    label = 'Example 2'


V = Rayleigh(tau=0.7, omega=0.3)
SRF = LafortuneLobe(ncoefs=10, i=5, a=[1.,1.,1.])


# initialize output fields for faster processing
Itot = np.ones_like(inc)*np.nan
Isurf = np.ones_like(inc)*np.nan
Iint = np.ones_like(inc)*np.nan
Ivol = np.ones_like(inc)*np.nan



fn = None
for i in xrange(len(inc)):
    # set geometries
    mu_0 = np.cos(np.deg2rad(inc[i]))
    mu_ex = mu_0*1.
    phi_0 = np.deg2rad(0.)
    phi_ex = phi_0 + np.pi

    
    R = RT1(I0, mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
    fn = R.fn  # store coefficients for faster itteration
    #Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()

toc = timeit.default_timer()
print('time for coefficient evaluation: ' + str(toc-tic))


# todo this separation in two loops is actually not needed and only for debugging
tic = timeit.default_timer()
for i in xrange(len(inc)):
    mu_0 = np.cos(np.deg2rad(inc[i]))
    mu_ex = mu_0*1.
    #phi_0 = 10.
    #phi_ex = np.pi   # todo ???

    #print inc[i], mu_0, mu_ex, phi_0, phi_ex

    R = RT1(I0, mu_0, mu_0, phi_0, phi_ex, RV=V, SRF=SRF, fn=fn, geometry='mono')
    Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()

toc = timeit.default_timer()
print('evaluating print-values took ' + str(toc-tic))


Plots(R).polarplot()

ctot='black'
csurf='red'
cvol='green'
cint='blue'

# multiply  I..  with  signorm  to get sigma0 values instead of normalized intensity
#signorm = 4.*np.pi*np.cos(np.deg2rad(inc))

f = plt.figure()
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax.plot(inc, 10.*np.log10(Itot), color=ctot, label='$I_{tot}$')
ax.plot(inc, 10.*np.log10(Isurf), color=csurf, label='$I_{surf}$')
ax.plot(inc, 10.*np.log10(Ivol), color=cvol, label='$I_{vol}$')
ax.plot(inc, 10.*np.log10(Iint), color=cint, label='$I_{int}$')
ax.grid()
ax.legend()
ax.set_xlabel('$\\theta_0$ [deg]')
ax.set_ylabel('$I^+$ [dB]')
ax.set_title(label)
ax.set_ylim(-40.,0.)

# plot fractions
ax2.plot(inc, Isurf/Itot, label='surf', color=csurf)
ax2.plot(inc, Ivol/Itot, label='volume', color=cvol)
ax2.plot(inc, Iint/Itot, label='interaction', color=cint)
ax2.set_title('fractional contributions on total signal')
ax2.set_xlabel('$\\theta_0$ [deg]')
ax2.set_ylabel('$I / I_{tot}$ [-]')
ax2.set_title(label)
ax2.grid()
ax2.legend()

plt.show()


assert bistaticplot, 'Manual stop before bistatic evaluation'

# ---------------------- generation 3d plots

# definition of incidence-angles
thetaplot = 55.
phiplot = 0.



# define plot-grid
theta,phi = np.linspace(np.deg2rad(1.),np.deg2rad(89.),25),np.linspace(0.,np.pi,25)
THETA,PHI = np.meshgrid(theta,phi)


# calculate bistatic fn coefficients
print('start of bistatic coefficient generation')
testfn = RT1(1., np.cos(np.deg2rad(thetaplot)), np.cos(np.deg2rad(45)), phiplot, np.pi, RV=V, SRF=SRF, fn=None, geometry='fvfv').fn

# initialize arrays
tot3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]
surf3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]
vol3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]
int3d=[[0 for i in range(0,len(theta))] for j in range(0,len(phi))]


# definition of function to evaluate bistatic contributions
# since this step might take a while an estimation of the calculation-time was included

def Rad(tt,pp):
    for i in range(0,len(tt)):
        if i == 0: tic = timeit.default_timer()
        if i == 0: print('... estimating evaluation-time...')
        for j in range(0,len(pp)):
            test = RT1(1., np.cos(np.deg2rad(thetaplot)), np.cos(theta[i]), phiplot,phi[j], RV=V, SRF=SRF, fn=testfn, geometry='ffff')
            tot3d[j][i],surf3d[j][i],vol3d[j][i],int3d[j][i] = test.calc()
        if i == 0: toc = timeit.default_timer()
        if i == 0: print('evaluation of 3dplot will take approximately ' + str(round((toc-tic)*len(tt)/60.,2)) + ' minutes')
    return np.array(tot3d,dtype=float) ,  np.array(surf3d,dtype=float),  np.array(vol3d,dtype=float),  np.array(int3d,dtype=float)


# evaluation of bistatic contributions
tic = timeit.default_timer()
tot3dplot, surf3dplot, vol3dplot, int3dplot = Rad(theta,phi)
toc = timeit.default_timer()
print('evaluation finished, it took ' + str(round((toc-tic)/60.,2)) + ' minutes')


# transform values to spherical coordinate system
def sphericaltransform(r):
    X = r * np.sin(THETA) * np.cos(PHI)
    Y = r * np.sin(THETA) * np.sin(PHI)
    Z = r * np.cos(THETA)
    return X,Y,Z



# plot of 3d scattering distribution
import mpl_toolkits.mplot3d as plt3d


fig = plt.figure(figsize=plt.figaspect(1.))
ax3d = fig.add_subplot(1,1,1,projection='3d')

#ax3d.view_init(elev=20.,azim=45)

maximumx = np.max(sphericaltransform(tot3dplot)[0])

xx=np.array([-maximumx,maximumx])
yy=np.array([0.,maximumx])
xxx,yyy = np.meshgrid(xx,yy)
zzz = np.ones_like(xxx)*(0.)

ax3d.plot_surface(xxx,yyy,zzz, alpha=0.2, color='k')




plot = ax3d.plot_surface(
    sphericaltransform(tot3dplot)[0],sphericaltransform(tot3dplot)[1],sphericaltransform(tot3dplot)[2] ,rstride=1, cstride=1, color='Gray',
    linewidth=0, antialiased=True, alpha=.3)

plot = ax3d.plot_surface(
    sphericaltransform(surf3dplot)[0],sphericaltransform(surf3dplot)[1],sphericaltransform(surf3dplot)[2], rstride=1, cstride=1, color='Red',
    linewidth=0, antialiased=True, alpha=.5)

plot = ax3d.plot_surface(
    sphericaltransform(vol3dplot)[0],sphericaltransform(vol3dplot)[1],sphericaltransform(vol3dplot)[2], rstride=1, cstride=1, color='Green',
    linewidth=0, antialiased=True, alpha=.5)

plot = ax3d.plot_surface(
    sphericaltransform(int3dplot)[0],sphericaltransform(int3dplot)[1],sphericaltransform(int3dplot)[2], rstride=1, cstride=1, color='Blue',
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

zoom=2.
ax3d.set_xlim(-np.max(sphericaltransform(tot3dplot))/zoom,np.max(sphericaltransform(tot3dplot))/zoom)
ax3d.set_ylim(-np.max(sphericaltransform(tot3dplot))/zoom,np.max(sphericaltransform(tot3dplot))/zoom)
ax3d.set_zlim(0,2*np.max(sphericaltransform(tot3dplot))/zoom)


# ensure display of correct aspect ratio (bug in mplot3d)
# due to the bug it is only possible to have equally sized axes (without changing the mplotlib code itself)

ax3d.auto_scale_xyz([np.min(sphericaltransform(tot3dplot)),np.max(sphericaltransform(tot3dplot))],
                    [0.,np.max(sphericaltransform(tot3dplot))+np.abs(np.min(sphericaltransform(tot3dplot)))],
                      [-np.max(sphericaltransform(tot3dplot))+np.abs(np.min(sphericaltransform(tot3dplot))),np.max(sphericaltransform(tot3dplot))+np.abs(np.min(sphericaltransform(tot3dplot)))])





plt.show()



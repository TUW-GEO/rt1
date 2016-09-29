"""
Reproduce examples like given in the paper
"""

import sys
sys.path.append('..')

import matplotlib.pyplot as plt

from rt1.volume import Rayleigh
from rt1.coefficients import RayleighIsotropic
from rt1.surface import Isotropic
from rt1.rt1 import RT1

import numpy as np

plt.close('all')

# Example1, Fig.7
I0=1.
inc = np.arange(0.,90.,1.)

V = Rayleigh(tau=0.7, omega=0.3)
SRF = Isotropic()

Itot = np.ones_like(inc)*np.nan
Isurf = np.ones_like(inc)*np.nan
Iint = np.ones_like(inc)*np.nan
Ivol = np.ones_like(inc)*np.nan

C = RayleighIsotropic()  # todo: this is not the combination in the paper!

for i in xrange(len(inc)):
    mu_0 = np.cos(np.deg2rad(inc[i]))
    mu_ex = mu_0*1.
    phi_0 = 0.
    phi_ex = np.pi

    R = RT1(I0, mu_0, mu_ex, phi_0, phi_ex, RV=V, SRF=SRF, Fn=C)
    Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()

ctot='black'
csurf='red'
cvol='green'
cint='blue'


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
ax.set_title('Figure 7')
ax.set_ylim(-40.,0.)

# plot fractions
ax2.plot(inc, Isurf/Itot, label='surf', color=csurf)
ax2.plot(inc, Ivol/Itot, label='volume', color=cvol)
ax2.plot(inc, Iint/Itot, label='interaction', color=cint)
ax2.set_title('fractional contributions on total signal')
ax2.set_xlabel('$\\theta_0$ [deg]')
ax2.set_ylabel('$I / I_{tot}$ [-]')
ax2.set_title('Figure 7')
ax2.grid()
ax2.legend()

plt.show()

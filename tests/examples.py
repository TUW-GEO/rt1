from volume import Rayleigh
from coefficients import RayleighIsotropic

# Example1, Fig.7
I0=1.
inc = np.arange(0.,90.,5.)

V = Rayleigh(tau=0.7, omega=0.3)
SRF = Isotropic()

Itot = np.ones_like(inc)*np.nan
Isurf = np.ones_like(inc)*np.nan
Iint = np.ones_like(inc)*np.nan
Ivol = np.ones_like(inc)*np.nan
for i in xrange(len(inc)):
    mu_0 = inc[i]
    mu_ex = mu_0*1.

    R = RT1(I0, mu_0, mu_ex, RV=V, SRF=SRF)
    Itot[i], Isurf[i], Ivol[i], Iint[i] = R.calc()

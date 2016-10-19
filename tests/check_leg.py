import sympy as sp
import numpy as np

"""
check sympy.Sum function for a Legendre series
it iss shown that he Sum function results in a different result
than when using summing up the arguments individually!
"""



n=sp.Symbol('n')

f1 = []
f2 = []
cum = 0.
#  results in differences after coefficient #8
for nmax in range(20):

    fak1 = (((2.*n+1.)*15.*sp.sqrt(sp.pi))/(16.*sp.gamma(((sp.Rational(7.)-n)/sp.Rational(2.)))*sp.gamma((sp.Rational(8.)+n)/sp.Rational(2.)))).expand()
    fak2=sp.legendre(n, 7.37721344327046e-17)  # use a tiny number of X which then should result in basically zero terms for every second polynomial
    r1=fak1.xreplace({n:nmax}).evalf()
    r2=fak2.xreplace({n:nmax}).evalf()
    f1.append(r1)
    f2.append(r2)
    cum += r1*r2  #... here we sum up the products individually --> approach A

    expr=sp.Sum((fak1*fak2).doit(), (n, 0, nmax))  # ... here we sum up the product of the factors using sympy.Sum

    print 'n, sympy.sum(), cum: ', nmax, expr.evalf(), cum

f1 = np.asarray(f1).astype('float')
f2 = np.asarray(f2).astype('float')
res = f1*f2
#~ print res

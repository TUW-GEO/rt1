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


print sp.__version__

"""
results in:
n, sympy.sum(), cum:  0 0.0833333333333333 0.0833333333333333
n, sympy.sum(), cum:  1 0.0833333333333333 0.0833333333333333
n, sympy.sum(), cum:  2 -0.0468750000000000 -0.0468750000000000
n, sympy.sum(), cum:  3 -0.0468750000000000 -0.0468750000000000
n, sympy.sum(), cum:  4 0.00585937500000000 0.00585937500000000
n, sympy.sum(), cum:  5 0.00585937500000001 0.00585937500000001
n, sympy.sum(), cum:  6 0.000569661458333343 0.000569661458333343
n, sympy.sum(), cum:  7 0.000569661458333343 0.000569661458333343
n, sympy.sum(), cum:  8 0.000569661458333343 0.000137329101562509    <<---- look at differences here!  solved! was due to old sympy version
n, sympy.sum(), cum:  9 0.000569661458333343 0.000137329101562509
n, sympy.sum(), cum:  10 0.000569661458333343 4.72068786621187e-5
n, sympy.sum(), cum:  11 0.000569661458333343 4.72068786621187e-5
n, sympy.sum(), cum:  12 0.000569661458333343 1.98880831400646e-5
n, sympy.sum(), cum:  13 0.000569661458333343 1.98880831400646e-5
n, sympy.sum(), cum:  14 0.000569661458333343 9.58889722825025e-6
n, sympy.sum(), cum:  15 0.000569661458333343 9.58889722825025e-6
n, sympy.sum(), cum:  16 0.000569661458333343 5.09410165251230e-6
n, sympy.sum(), cum:  17 0.000569661458333343 5.09410165251230e-6
n, sympy.sum(), cum:  18 0.000569661458333343 2.91259978743577e-6
n, sympy.sum(), cum:  19 0.000569661458333343 2.91259978743577e-6
"""

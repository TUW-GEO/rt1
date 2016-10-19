import sympy as sp
import numpy as np

n=sp.Symbol('n')


f1 = []
f2 = []
cum = 0.


#  results in differentces after coefficient #8
for nmax in range(20):

    fak1 = (((2.*n+1.)*15.*sp.sqrt(sp.pi))/(16.*sp.gamma(((sp.Rational(7.)-n)/sp.Rational(2.)))*sp.gamma((sp.Rational(8.)+n)/sp.Rational(2.)))).expand()
    fak2=sp.legendre(n, 7.37721344327046e-17)
    r1=fak1.xreplace({n:nmax}).evalf()
    r2=fak2.xreplace({n:nmax}).evalf()
    f1.append(r1)
    f2.append(r2)

    #~ fak1= (1.875*sp.sqrt(sp.pi)*n/(sp.gamma(-n/.2 + 7./2.)*sp.gamma(n/2. + 4)) + 0.9375*sp.sqrt(sp.pi)/(sp.gamma(-n/2. + 7./2.)*sp.gamma(n/2. + 4.)))

    expr=sp.Sum((fak1*fak2).doit(), (n, 0, nmax))  # results in wrong results !!! problem with Sum function ???
    #expr = sp.Sum((1.875*sp.sqrt(sp.pi)*n/(sp.gamma(-n/2 + 7/2)*sp.gamma(n/2 + 4)) + 0.9375*sp.sqrt(sp.pi)/(sp.gamma(-n/2 + 7/2)*sp.gamma(n/2 + 4)))*sp.legendre(n, 7.37721344327046e-17), (n, 0, nmax))
    cum += r1*r2
    print nmax, expr.evalf(), cum

f1 = np.asarray(f1).astype('float')
f2 = np.asarray(f2).astype('float')

res = f1*f2
print res

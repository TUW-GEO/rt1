import sys
sys.path.append('..')

from rt1.rt1 import RT1
from rt1.volume import Rayleigh
from rt1.surface import CosineLobe

import numpy as np

import time





# some speed benchmarking ...

S = CosineLobe(ncoefs=10, i=5)
V = Rayleigh(tau=0.7, omega=0.3)

I0 = 1.

theta_i = np.pi/2.
theta_ex = 0.234234
phi_i = np.pi/2.
phi_ex = 0.

RT = RT1(I0, np.cos(theta_i), np.cos(theta_ex), phi_i, phi_ex, RV=V, SRF=S, geometry='vvvv')
#~ print RT.fn
#~ print len(RT.fn)

#~ start = time.time()
#~ for i in xrange(10):
    #~ for n in xrange(len(RT.fn)):
        #~ RT._get_fn(n, theta_i, phi_i, theta_ex, phi_ex)
#~ end = time.time()
#~ print('Time for get_fn (10x): ', end - start)


start = time.time()
for i in xrange(10):
    RT.interaction()
end = time.time()
print('Time for interaction (10x): ', end - start)

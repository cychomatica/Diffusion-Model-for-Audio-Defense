import numpy as np
import matplotlib.pyplot as plt

N = 1000
sigma = 0.25

radius_array = np.arange(0,4.05,0.05)
diffusion = np.load('records/num_examples=100_N={}_sigma={}_defense=diffusion.npy'.format(N,sigma))
diffusion_2 = np.load('records/2ShotsRev/num_examples=100_N={}_sigma={}_defense=diffusion.npy'.format(N,sigma))
randsmooth = np.load('records/num_examples=100_N={}_sigma={}_defense=randsmooth.npy'.format(N,sigma))

plt.figure(dpi=1000)
plt.title('sigma={}, N={}'.format(sigma,N))
plt.xlabel('radius')
plt.ylabel('certified accuracy')
plt.plot(radius_array, diffusion, label='diffusion smoothing (one-shot-rev)')
plt.plot(radius_array, diffusion_2, label='diffusion smoothing (two-shots-rev)')
plt.plot(radius_array, randsmooth, label='randomized smoothing')
plt.legend()
plt.savefig('figures/num_examples=100_N={}_sigma={}.png'.format(N,sigma))

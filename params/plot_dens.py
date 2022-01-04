import numpy as np
import matplotlib.pyplot as plt

tng_dir = '/mnt/gosling1/boryanah/TNG300/'
#delta_R = np.load(tng_dir+'smoothed_density_R5_tng300-3_99.npy')
delta_R = np.load(tng_dir+'smoothed_density_R1_tng300-3_99.npy')
N_dim = delta_R.shape[0]
delta_s = 0.25
p = 2.

# obtain mark
mark = (1+delta_s/(1+delta_s+delta_R))**p

# plot a 2d array (the log function just makes it easier to see the overdensities by eye)
plt.figure(1)
plt.imshow(np.log10(mark[:, :, 20]), origin='lower')
plt.colorbar()
plt.gca().set_aspect('equal')
plt.xlim([0, N_dim])
plt.ylim([0, N_dim])
plt.savefig("figs/log_mark.png")

plt.figure(2)
plt.imshow(np.log10(delta_R[:, :, 20]+1.), origin='lower')
plt.colorbar()
plt.gca().set_aspect('equal')
plt.xlim([0, N_dim])
plt.ylim([0, N_dim])
plt.savefig("figs/log_overdensity.png")
plt.show()

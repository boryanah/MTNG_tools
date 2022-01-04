import numpy as np
import matplotlib.pyplot as plt
from tools import *
from nbodykit.lab import ArrayMesh, FieldMesh

# simulation parameters
Lbox = 500. # Mpc/h
N_dim = 1024
grid_size = Lbox/N_dim

# load particles
tng_dir = "/mnt/alan1/boryanah/MTNG/"
snap = 184; sim_type = 'dm'
#snap = 179; sim_type = 'fp'

# WE ARE JUST GONNA LOAD THE TSC DENSITY FIELD DIRECTLY
density = np.load(tng_dir+f"density_ngrid_{N_dim:d}_snap_{snap:d}_{sim_type:s}.npy")
mean_dens = np.mean(density)
density = ArrayMesh(density, Lbox)
print("mean density = ", mean_dens)

# I AM COMMENTING OUT CAUSE I HAVE THE TSC DENSITY FIELD ALREADY
"""
#pos_parts = np.load(tng_dir+"data_parts/pos_down_1000_snap_179_fp.npy")
pos_parts = np.load(tng_dir+"data_parts/pos_down_1000_snap_184_dm.npy")
dtype = [('Position', ('f8', 3))]
data = np.empty(pos_parts.shape[0], dtype=dtype)
data['Position'] = pos_parts

# total number of particles 
N_parts = pos_parts.shape[0]
del pos_parts

# mean density
mean_dens = N_parts/Lbox**3

# get unsmoothed density
'''
strides = np.zeros(3, dtype=int)
strides[2] = 1
for j in range(1, -1, -1):
    strides[j] = strides[j + 1] * N_dim
print(strides)
density = np.zeros((N_dim**3))
print(paint_tsc(pos=pos_parts, meshflat=density, mesh_strides=strides, period=N_dim))
density = density.reshape((N_dim, N_dim, N_dim))
print(density[:3])
'''
'''
density = get_density(pos_parts, N_dim, Lbox)
'''
density = ArrayCatalog(data).to_mesh(window='tsc', Nmesh=N_dim, BoxSize=Lbox)
mean_dens = 1. # automatically does it for you
"""

want_show = 1
if want_show:
    #density_fft = FieldMesh(density.paint(mode='complex'))
    #filter_tophat = TopHat(12.)
    #density = (density_fft.apply(filter_tophat, mode='complex', kind='wavenumber')).paint(mode='real')
    print("minimum divided by mean = ", np.min(density.preview()))
    print("mean = ", np.mean(density.preview()), mean_dens)

    plt.imshow(np.log10(density.preview()[:, :, 1000]))
    plt.show()
    quit()

# density condition
min_dens = 0.2*mean_dens

# starting smoothing scale
R_max = 15. # Mpc/h

# minimum void size
R_min = 3. # Mpc/h

# smoothing scales
Rs = np.arange(R_min, R_max, 0.5 * grid_size)[::-1]

# mask for all regions
mask = np.zeros((N_dim, N_dim, N_dim), dtype=np.float32)

for i in range(len(Rs)):
    print("counter = ", i, len(Rs)-1)

    # tophat radius
    R_th = Rs[i]

    # smooth map: convolve with tophat filter
    density_fft = FieldMesh(density.paint(mode='complex'))
    filter_tophat = TopHat(R_th)
    smooth_density = (density_fft.apply(filter_tophat, mode='complex', kind='wavenumber')).paint(mode='real')
    #smooth_density = smooth_tophat(density, R_th)
    
    # identify regions that are below density minimum and are not part of the mask
    active_void = (smooth_density.preview() <= min_dens) & (mask == 0.)
    print("number of voids = ", np.sum(active_void))
    mask[active_void] = R_th

np.save(tng_dir+f"mask_void_ngrid_{N_dim:d}_snap_{snap:d}_{sim_type:s}.npy", mask)

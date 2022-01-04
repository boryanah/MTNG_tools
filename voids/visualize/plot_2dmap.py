import numpy as np
import matplotlib.pyplot as plt

tng_dir = "/mnt/alan1/boryanah/MTNG/"
Lbox = 500
N_dim = 512
gr_size = Lbox/N_dim

density = np.load(tng_dir+"smoothed_density_R1_mtng_179.npy")
#density = np.load(tng_dir+"smoothed_density_R2_mtng_179.npy")
#void = np.loadtxt("data/pos_down_10000_snap_179_fp.SVF_recen_ovl0.5")
void = np.loadtxt("data/SubhaloPos_fp_179_1e11.SVF_recen_ovl0.5")
xyz_void = void[:, :3].astype(np.float32)/gr_size
rad_void = void[:, 3].astype(np.float32)/gr_size

slab_id = 20
slab_min = slab_id-8
slab_max = slab_id+8

# draw circles
choice = (slab_min < xyz_void[:, 2]) & (slab_max > xyz_void[:, 2])
inds = np.arange(xyz_void.shape[0], dtype=int)[choice]
print(len(inds))

#plt.imshow(np.log10(1+density[:, :, slab_id]), origin='lower left')
plt.imshow(np.log10(1+density[:, :, slab_id]), origin='lower left')
ax = plt.gca()
for i in range(len(inds)):
    
    circle = plt.Circle((xyz_void[inds[i], 1], xyz_void[inds[i], 0]), radius=rad_void[inds[i]], color="r", fill=False)
    # draw circle
    ax.add_artist(circle)
    continue

# clear things for fresh plot
#ax.cla()

plt.axis("equal")
plt.show()

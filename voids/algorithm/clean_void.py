import numpy as np
from tools import dist

# voids
tng_dir = "/mnt/alan1/boryanah/MTNG/"
Lbox = 500. # Mpc/h
N_dim = 1024
#snap = 184; sim_type = "dm"
snap = 179; sim_type = "fp"
mask = np.load(tng_dir+f"mask_void_ngrid_{N_dim:d}_snap_{snap:d}_{sim_type:s}.npy")
grid_size = Lbox/N_dim
centers = np.arange(0., Lbox, grid_size) + grid_size/2.
assert len(centers) == N_dim
max_overlap = 0.5

# isolate the voids
ind_x, ind_y, ind_z = np.where(mask > 0.)
void_pos = np.vstack((centers[ind_x], centers[ind_y], centers[ind_z])).T
void_size = mask[mask > 0.]

# order them by size
i_sort = np.argsort(void_size)[::-1]
void_pos = void_pos[i_sort]
void_size = void_size[i_sort]
print(void_pos[:5])
print(void_size[:5])

# array with maximum overlap for each void
void_overlap = np.ones(len(void_size))
void_inds = np.arange(len(void_size), dtype=int)
void_volume = 4/3.*np.pi*void_size**3./Lbox**3
volume_min = 0.5 # if 50% is covered in voids, you can stop

volume = 0
for i in range(len(void_size)):
    if i % 100 == 1: print(i, np.sum(select), volume)
    if volume > volume_min: break
    if void_overlap[i] <= max_overlap:
        void_size[i] = 0.
        continue
    else:
        volume += void_volume[i]
    # compute distances between voids
    select = (void_inds > i) & (void_overlap > max_overlap)
    r = dist(void_pos[select], void_pos[i], L=Lbox)
    R_sum = void_size[i]+void_size[select]
    ratio = r/R_sum

    # update overall overlap for the remaining active void centers
    tmp = void_overlap[select]
    condition = (ratio < tmp)
    tmp[condition] = ratio[condition]
    void_overlap[select] = tmp

# save data
viable = void_size > 0.
void_overlap = void_overlap[viable]
void_pos = void_pos[viable]
void_size = void_size[viable]
dtype = [('Position', ('f4', 3)), ('Size', 'f4'), ('Overlap', 'f4')]
data = np.empty(len(void_size), dtype=dtype)
data['Position'] = void_pos
data['Size'] = void_size
data['Overlap'] = void_overlap
np.save(tng_dir+f"voids_ngrid_{N_dim:d}_snap_{snap:d}_{sim_type:s}.npy", data)

# could form a small 27 cube thing and interpolate to find the center

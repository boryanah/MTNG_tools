import glob
import numpy as np

data_dir = "/home/boryanah/MTNG/voids/glamdring/data/"
overlap_factor = 0.2
ncentres = 5000000
Lbox = 500.
snap_type = '184_dm'

handle = glob.glob(data_dir+f"pos_down_1000_snap_{snap_type:s}_ncentres_{ncentres:d}_*voids_overlap{overlap_factor:.1f}.dat")[0]
print(handle)
voids = np.loadtxt(handle)
pos = voids[:, :3]
r = voids[:, 3]

print(r[:10])
print(pos[:10])

voids = np.load("/mnt/alan1/boryanah/MTNG/voids_ngrid_1024_snap_184_dm.npy")
r = voids['Size']
pos = voids['Position']

print(r[:10])
print(pos[:10])

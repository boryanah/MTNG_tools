import glob
import os

import numpy as np
import matplotlib.pyplot as plt
#import plotparams
#plotparams.buba()

data_dir = "/home/boryanah/MTNG/voids/glamdring/data/"
overlap_factors = [0.2]#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#ncentreses = [100000]
ncentreses = [1000000, 5000000]
Lbox = 500.
bins = np.linspace(1, 30, 51)
binc = (bins[1:] + bins[:-1]) * 0.5
snap_types = ['179_fp', '184_dm']
#snap_types = ['179_fp']
#snap_types = ['184_dm']

plt.figure(figsize=(9, 7))
for snap_type in snap_types:
    for ncentres in ncentreses:
        for overlap_factor in overlap_factors:
            handle = glob.glob(data_dir+f"pos_down_1000_snap_{snap_type:s}_ncentres_{ncentres:d}_*voids_overlap{overlap_factor:.1f}.dat")[-1]
            print(handle)
            if os.path.exists(handle) is not True: print("non existent"); continue
            voids = np.loadtxt(handle)
            pos = voids[:, :3]
            r = voids[:, 3]

            print("volume covered = ", np.sum(4./3*np.pi*r**3)*100./Lbox**3)
            print("number of voids = ", len(r))
            print("-----------------")
            
            hist, bin_edges = np.histogram(r, bins)

            #chist = np.cumsum(hist)
            chist = np.cumsum(hist[::-1])[::-1]

            plt.plot(binc, chist, label=f"{snap_type:s}, ncentres = {ncentres:d}, overlap = {overlap_factor:.1f}")

r = np.load("/mnt/alan1/boryanah/MTNG/voids_ngrid_1024_snap_184_dm.npy")['Size']
print("volume covered = ", np.sum(4./3*np.pi*r**3)*100./Lbox**3)
print("number of voids = ", len(r))
hist, bin_edges = np.histogram(r, bins)
chist = np.cumsum(hist[::-1])[::-1]
plt.plot(binc, chist, label=r"New algorithm DM")

r = np.load("/mnt/alan1/boryanah/MTNG/voids_ngrid_1024_snap_179_fp.npy")['Size']
print("volume covered = ", np.sum(4./3*np.pi*r**3)*100./Lbox**3)
print("number of voids = ", len(r))
hist, bin_edges = np.histogram(r, bins)
chist = np.cumsum(hist[::-1])[::-1]
plt.plot(binc, chist, label=r"New algorithm FP")

plt.yscale('log')
plt.xlabel(r'${\rm Radius}$')
plt.ylabel(r'${\rm Cumulative number}$')
plt.xlim([1, 14.])
plt.legend()
plt.savefig("1mln_nshifts_128.png")
plt.show()

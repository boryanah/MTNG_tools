import os

import numpy as np
import matplotlib.pyplot as plt

tng_dir = "/mnt/alan1/boryanah/MTNG/"
snapshot = 179
str_snap = f"_{snapshot:d}"
Lbox = 500. # Mpc/h

# load other halo properties
GroupPos_fp = np.load(tng_dir+'data_fp/GroupPos_fp'+str_snap+'.npy')
GroupEnv_fp = np.load(tng_dir+'data_fp/GroupEnv_R2_fp'+str_snap+'.npy')
GrMcrit_fp = np.load(tng_dir+'data_fp/Group_M_TopHat200_fp'+str_snap+'.npy')*1.e10
SubhaloSFR = np.load(tng_dir+f"data_fp/SubhaloSFR_fp_{snapshot:d}.npy")
SubhaloMstar = np.load(tng_dir+f"data_fp/SubhaloMassType_fp_{snapshot:d}.npy")[:, 4]*1.e10

# specific star formation
SubhalosSFR = SubhaloSFR/SubhaloMstar

# 7.4e-4
mstar_thresh = 8.44e9
ssfr_thresh = 8e-10
mstar_lrg = 7.2e10 # fyi

# 5.9e-4
#mstar_thresh = 9e9
#ssfr_thresh = 8.5e-10
#mstar_lrg = 8.4e10 # fyi

# 9.7e-4
#mstar_thresh = 7.9e9
#ssfr_thresh = 7.4e-10
#mstar_lrg = 6.0e10 # fyi

# 2.0e-3
mstar_thresh = 6.e9
ssfr_thresh = 6e-10
mstar_lrg = 3.3e10 # fyi

index_ELG = np.arange(len(SubhaloMstar), dtype=int)
index_ELG = index_ELG[(SubhaloMstar > mstar_thresh) & (SubhalosSFR > ssfr_thresh)]
index_LRG = np.argsort(SubhaloMstar)[::-1][:len(index_ELG)]
n_LRG = len(index_LRG)/Lbox**3
n_ELG = len(index_ELG)/Lbox**3

print("number of ELG galaxies = ", len(index_ELG))
print(f"number density of ELG galaxies = {n_ELG:.1e}")
print(f"number density of LRG galaxies = {n_LRG:.1e}")
print("min mass LRG = %.1e" %SubhaloMstar[index_LRG].min())

np.save(f"data/index_ELG_{n_ELG:.1e}_{snapshot:d}.npy", index_ELG)
np.save(f"data/index_LRG_{n_LRG:.1e}_{snapshot:d}.npy", index_LRG)

# reduce number of things to be plotted
choice = SubhaloMstar > 5.e9

plt.figure(figsize=(9, 7))
plt.scatter(SubhaloMstar[choice], SubhalosSFR[choice], s=4, color="gray", alpha=0.1, marker="*")
plt.scatter(SubhaloMstar[index_LRG], SubhalosSFR[index_LRG], s=50, color="orangered", marker="*", alpha=0.2)
plt.scatter(SubhaloMstar[index_ELG], SubhalosSFR[index_ELG], s=50, color="dodgerblue", marker="*", alpha=0.2)
plt.gca().axvline(x=mstar_thresh, color='k', ls='--')
plt.gca().axhline(y=ssfr_thresh, color='k', ls='--')
plt.ylim([1.e-14, 1.e-7])
plt.xscale('log')
plt.yscale('log')
plt.show()

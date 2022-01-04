import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

#fp_dm = 'dm'
fp_dm = 'fp'
snap = 55

parent = np.load(f"data/galaxy_parent_r200m_{fp_dm:s}_{snap:d}.npy") # should come from the query assignment
subind = np.load(f"data/galaxy_subind_r200m_{fp_dm:s}_{snap:d}.npy")
m200_infall = np.load(f"data/MissGal_Infall_M200m_{snap:d}_{fp_dm:s}.npy")
infall_grnr = np.load(f"data/MissGal_Infall_SubGrNr_{snap:d}_{fp_dm:s}.npy") 

subind = subind[parent == -1]
#parent = parent[parent == -1] # so parent is -1, hence you should not use it

infall_time = infall_grnr[:, 0]
infall_grnr = infall_grnr[:, 1]

subind = subind[infall_time > 0]
#parent = parent[infall_time > 0]
m200_infall = m200_infall[infall_time > 0]
infall_grnr = infall_grnr[infall_time > 0]
infall_time = infall_time[infall_time > 0]


print(subind.shape)

tng_dir = "/mnt/gosling1/boryanah/TNG300/"
SubhaloMass = np.load(tng_dir + f"SubhaloMass_{fp_dm:s}_{snap:d}.npy")*1.e10
SubhaloPos = np.load(tng_dir + f"SubhaloPos_{fp_dm:s}_{snap:d}.npy")*1.e-3
SubhaloGrNr = np.load(tng_dir + f"SubhaloGrNr_{fp_dm:s}_{snap:d}.npy")
GroupPos = np.load(tng_dir + f"GroupPos_{fp_dm:s}_{snap:d}.npy")*1.e-3

sub_mass = SubhaloMass[subind]
parent = SubhaloGrNr[subind]
sub_pos = SubhaloPos[subind]
par_pos = GroupPos[parent]

Lbox = 205.
dist = par_pos-sub_pos
dist[dist >= Lbox/2.] -= Lbox
dist[dist < -Lbox/2.] += Lbox
dist = np.sqrt(np.sum(dist**2, axis=1))

print(dist[:10])
print(dist.max())

ratio = m200_infall/sub_mass

bins = np.geomspace(ratio.min(), ratio.max(), 21)
binc = bins[1:]+bins[:-1]

alpha = 0.1
s=75
hist_rat, _ = np.histogram(ratio, bins=bins)

plt.figure(1, figsize=(9, 7))
plt.plot(binc, hist_rat, color='dodgerblue')
plt.xscale('log')
plt.xlabel('SubMass(now)/M200(infall)')

plt.figure(2, figsize=(9, 7))
plt.scatter(ratio, dist, s=s, alpha=alpha, color='dodgerblue')
plt.xscale('log')
plt.xlabel('SubMass(now)/M200(infall)')
plt.ylabel('Distance to center (now)')

plt.figure(3, figsize=(9, 7))
plt.scatter(ratio, infall_time, s=s, alpha=alpha, color='dodgerblue')
plt.xscale('log')
plt.xlabel('SubMass(now)/M200(infall)')
plt.ylabel('Snapshot (infall)')

plt.figure(4, figsize=(9, 7))
plt.scatter(dist, infall_time, s=s, alpha=alpha, color='dodgerblue')
#plt.xscale('log')
plt.xlabel('Distance to center (now)')
plt.ylabel('Snapshot (infall)')
plt.show()

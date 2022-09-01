"""
Plot True and Shuffled sample
"""
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','#BBBBBB','#33BBEE','#EE3377','#0099BB']

# 0077BB is dark blue; EE7733 is orange; EE3377 is cyclamen; 33BBEE is blue; CC3311 is brick; 0099BB is dark green-blue; BBBBBB is silver

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
fp_dm = 'dm'
snapshots = [179, 264]
zs = [1., 0.]

# k nearest neighbors
dtype = np.int64
ks = np.array([1, 2, 4, 8], dtype=dtype)
ks_wanted = np.array([8], dtype=dtype)
lw = [1, 2, 2.5, 3]

# definitions for the axes
left, width = 0.14, 0.85#0.1, 0.65
bottom, height = 0.1, 0.25#0.2#65
spacing = 0.005

rect_scatter = [left, bottom + (height + spacing), width, 0.6]
rect_histx = [left, bottom, width, height]

# start with a rectangular Figure
plt.figure(figsize=(9, 10))
ax_scatter = plt.axes(rect_scatter)
#(direction='in', top=True, right=True)
ax_histx = plt.axes(rect_histx)

counter = 0
for i, snapshot in enumerate(snapshots):
    z = zs[i]
    z_label = f"z = {z:.1f}"
    if fp_dm == 'dm':
        snapshot_dm = snapshot+5
    else:
        snapshot_dm = snapshot
    
    for gal_type in gal_types:
        gal_label = "{\\rm "+f"{gal_type}s"+"}"

        # load correlation function
        data_true_mean = np.load(f"{gal_type:s}/kNN_true_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
        data_shuff_mean = np.load(f"{gal_type:s}/kNN_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
        data_rat_shuff_mean = np.load(f"{gal_type:s}/kNN_rat_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
        data_rat_shuff_err = np.load(f"{gal_type:s}/kNN_rat_shuff_err_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
        binc = np.load(f"{gal_type:s}/rbinc.npy")

        # go to each order
        for j, k in enumerate(ks):
            if k not in ks_wanted: continue 
            kNN_true_mean = data_true_mean[:, j]
            kNN_shuff_mean = data_shuff_mean[:, j]
            kNN_true_mean[kNN_true_mean > 0.5] = 1 - kNN_true_mean[kNN_true_mean > 0.5]
            kNN_shuff_mean[kNN_shuff_mean > 0.5] = 1 - kNN_shuff_mean[kNN_shuff_mean > 0.5]
            rat_shuff_mean = data_rat_shuff_mean[:, j]
            rat_shuff_err = data_rat_shuff_err[:, j]
            rbinc = binc[:, j]

            if k == ks_wanted[0]:
                ax_scatter.errorbar(rbinc/k**(1./3), kNN_true_mean, capsize=4, color=hexcolors_bright[counter], lw=lw[j], ls='-', label=rf"${z_label}, \ {gal_label}$")
            else:
                ax_scatter.errorbar(rbinc/k**(1./3), kNN_true_mean, capsize=4, color=hexcolors_bright[counter], lw=lw[j], ls='-')
            ax_scatter.errorbar(rbinc/k**(1./3), kNN_shuff_mean, capsize=4, color=hexcolors_bright[counter], lw=lw[j], ls='--')
            ax_histx.errorbar(rbinc/k**(1./3), rat_shuff_mean, yerr=rat_shuff_err, capsize=4, color=hexcolors_bright[counter], lw=lw[j], ls='-')
        
        counter += 1
ax_scatter.set_xscale('log')
ax_scatter.set_yscale('log')
ax_scatter.plot([], [], color='black', ls='-', label='MTNG')
#ax_scatter.plot([], [], color='black', ls='-', lw=2, label='$k=2$')
#ax_scatter.plot([], [], color='black', ls='-', lw=4, label='$k=8$')
ax_scatter.plot([], [], color='black', ls='--', label='Shuffled')
ax_scatter.legend(ncol=2, fontsize=22)
ax_scatter.set_xticks([])
ax_scatter.set_ylim([1.e-6, 2])
ax_scatter.set_xlabel(r'$r/k^{1 \over 3} \ [{\rm Mpc}/h]$')
ax_scatter.set_ylabel(r'${\rm Peaked \ CDF}$')
ax_histx.axhline(y=1, color='black', ls='--')
ax_histx.set_xlabel(r'$r/k^{1 \over 3} \ [{\rm Mpc}/h]$')
ax_histx.set_ylabel(r'${\rm Ratio}$')
ax_histx.set_xscale('log')
ymin, ymax = ax_histx.get_ylim()
ymin = np.floor(ymin*10.)/10.
ymax = np.ceil(ymax*10.)/10.
ax_histx.set_yticks(np.arange(ymin, ymax, 0.1))
#ax_histx.set_ylim([0.25, 1.75])
#ax_histx.set_yscale('log')
plt.savefig(f"figs/kNN.png")
plt.show()

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
#gal_types = ['ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
#n_gal = '7.0e-04'
#n_gal = '7.8e-04'
#n_gal = '6.0e-04'
p1, p2 = n_gal.split('e-0')
fp_dm = 'fp'
snapshots = [179, 264]
zs = [1., 0.]

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
        if gal_type == 'ELG':
            drad_str = "_drad"
        else:
            drad_str = ""
        corr_true_mean = np.load(f"{gal_type:s}/corr_mean_{n_gal}{drad_str}_{fp_dm:s}_{snapshot_dm:d}.npy")
        corr_shuff_mean = np.load(f"{gal_type:s}/corr_shuff_mean_{n_gal}{drad_str}_{fp_dm:s}_{snapshot_dm:d}.npy")
        rat_shuff_mean = np.load(f"{gal_type:s}/corr_rat_shuff_mean_{n_gal}{drad_str}_{fp_dm:s}_{snapshot_dm:d}.npy")
        rat_shuff_err = np.load(f"{gal_type:s}/corr_rat_shuff_err_{n_gal}{drad_str}_{fp_dm:s}_{snapshot_dm:d}.npy")
        rbinc = np.load(f"{gal_type:s}/rbinc.npy")
        
        ax_scatter.errorbar(rbinc, corr_true_mean*rbinc**2, capsize=4, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
        ax_scatter.errorbar(rbinc, corr_shuff_mean*rbinc**2, capsize=4, color=hexcolors_bright[counter], ls='--')
        ax_histx.errorbar(rbinc, rat_shuff_mean, yerr=rat_shuff_err, capsize=4, color=hexcolors_bright[counter], ls='-')
        
        counter += 1
ax_scatter.set_xscale('log')
#ax_scatter.set_yscale('log')
label = r"$n_{\rm gal} = %s \times 10^{-%s}$"%(p1, p2)
ax_scatter.text(0.6, 0.1, s=label, transform=ax_scatter.transAxes)
ax_scatter.plot([], [], color='black', ls='-', label='MTNG')
ax_scatter.plot([], [], color='black', ls='--', label='Shuffled')
ax_scatter.legend(ncol=2, fontsize=20)
ax_scatter.set_xticks([])
ax_scatter.set_xlabel(r'$r \ [{\rm Mpc}/h]$')
ax_scatter.set_ylabel(r'$\xi(r) r^2$')
ax_histx.plot(rbinc, np.ones(len(rbinc)), color='black', ls='--')
ax_histx.set_xlabel(r'$r \ [{\rm Mpc}/h]$')
ax_histx.set_ylabel(r'${\rm Ratio}$')
ax_histx.set_xscale('log')
ymin, ymax = ax_histx.get_ylim()
ymin = np.floor(ymin*10.)/10.
ymax = np.ceil(ymax*10.)/10.
ax_histx.set_yticks(np.arange(ymin, ymax, 0.2))
#ax_histx.set_ylim([0.25, 1.75])
#ax_histx.set_yscale('log')
plt.savefig(f"figs/corr_{n_gal}.png")
plt.show()
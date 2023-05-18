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
#tng_dir = "/mnt/alan1/boryanah/MTNG/"
tng_dir = "/mnt/alan1/boryanah/MTNG/dm_arepo/"
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
fp_dm = 'dm'
snapshots = [179, 264]
zs = [1., 0.]

for l0l2_type in ['l0', 'l2']:
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
        if fp_dm == 'dm' and "arepo" not in dm:
            snapshot_dm = snapshot+5
        else:
            snapshot_dm = snapshot

        for gal_type in gal_types:
            gal_label = "{\\rm "+f"{gal_type}s"+"}"

            # load correlation function
            xil0_true_mean = np.load(f"{gal_type:s}/xi{l0l2_type}_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
            xil0_shuff_mean = np.load(f"{gal_type:s}/xi{l0l2_type}_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
            ratl0_shuff_mean = np.load(f"{gal_type:s}/rat{l0l2_type}_shuff_mean_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
            ratl0_shuff_err = np.load(f"{gal_type:s}/rat{l0l2_type}_shuff_err_{n_gal}_{fp_dm:s}_{snapshot_dm:d}.npy")
            rbinc = np.load(f"{gal_type:s}/rbinc.npy")

            ax_scatter.errorbar(rbinc, xil0_true_mean*rbinc**2, capsize=4, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
            ax_scatter.errorbar(rbinc, xil0_shuff_mean*rbinc**2, capsize=4, color=hexcolors_bright[counter], ls='--')
            ax_histx.errorbar(rbinc, ratl0_shuff_mean, yerr=ratl0_shuff_err, capsize=4, color=hexcolors_bright[counter], ls='-')

            counter += 1
    ax_scatter.set_xscale('log')
    #ax_scatter.set_yscale('log')
    ax_scatter.plot([], [], color='black', ls='-', label='MTNG')
    ax_scatter.plot([], [], color='black', ls='--', label='Shuffled')
    ax_scatter.legend(ncol=2, fontsize=22)
    ax_scatter.set_xticks([])
    ax_scatter.set_xlabel(r'$r \ [{\rm Mpc}/h]$')
    ax_scatter.set_ylabel(r'$\xi_{\ell=%s}(r) r^2$'%(l0l2_type.split('l')[-1]))
    ax_histx.plot(rbinc, np.ones(len(rbinc)), color='black', ls='--')
    ax_histx.set_xlabel(r'$r \ [{\rm Mpc}/h]$')
    ax_histx.set_ylabel(r'${\rm Ratio}$')
    ax_histx.set_xscale('log')
    ymin, ymax = ax_histx.get_ylim()
    if ymin < 0.: ymin = 0.
    if ymax > 2.: ymax = 2.
    ax_histx.set_ylim([ymin, ymax])
    ymin = np.floor(ymin*10.)/10.
    ymax = np.ceil(ymax*10.)/10.
    ax_histx.set_yticks(np.arange(ymin, ymax, 0.2))
    #ax_histx.set_ylim([0.25, 1.75])
    #ax_histx.set_yscale('log')
    plt.savefig(f"figs/xi{l0l2_type}.png", bbox_inches='tight', pad_inches=0.)
    plt.show()

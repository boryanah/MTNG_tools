"""
Plot the galaxy assembly bias signature in different mass bins
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# matplotlib settings
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)

import plotparams
plotparams.buba()

# colors
hexcolors_bright = ['#CC3311','#0077BB','#EE7733','limegreen','#BBBBBB','#33BBEE','#EE3377','#0099BB']
greysafecols = ['#809BC8', 'black', '#FF6666', '#FFCC66', '#64C204']
# 0077BB is dark blue; EE7733 is orange; EE3377 is cyclamen; 33BBEE is blue; CC3311 is brick; 0099BB is dark green-blue; BBBBBB is silver

# simulation parameters
tng_dir = "/mnt/alan1/boryanah/MTNG/"
gal_types = ['LRG', 'ELG']
n_gal = '2.0e-03' #['7.0e-04', '2.0e-03']
#n_gal = '7.0e-04'
p1, p2 = n_gal.split('e-0')
fp_dm = 'fp'
snapshots = [179, 264]
zs = [1., 0.]

nrow, ncol = 2, 2
if ncol == 3:
    fig, axes = plt.subplots(nrow, ncol, figsize=(21, 5.*2.))
elif ncol == 4:
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 5.5*2.))
else:
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.5*2.))
counter = 0
for j, snapshot in enumerate(snapshots):
    z = zs[j]
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
            offset = ncol
        else:
            drad_str = ""
            offset = 0

        """
        if gal_type == 'ELG':
            nbins = np.arange(0, 3) # 12, 12.5, 13., 13.5
        else:
            nbins = np.arange(1, 4) # 12.5, 13., 13.5, 14
        """
        nbins = np.array([0, 1]) # 12.5, 13.5

        for i in range(len(nbins)):

            plt.subplot(nrow, ncol, i+offset+1)

            if gal_type == 'ELG':
                drad_str = "_drad"
            else:
                drad_str = ""

            plt.axhline(y=1, color='k', ls='--')

            # all
            f = np.load(f'{gal_type:s}/{gal_type:s}_corr_bin_{nbins[i]:d}_gals{drad_str}_{n_gal}_{fp_dm}_{snapshot:d}.npz')
            rat_mean, rat_err, rbinc, logm = f['mean'], f['err'], f['binc'], f['logm']
            
            if i == len(nbins)-1:
                #plt.errorbar(rbinc, rat_mean, yerr=rat_err, capsize=4, color=hexcolors_bright[counter], ls='-', label=rf"${z_label}, \ {gal_label}$")
                plt.errorbar(rbinc, rat_mean, capsize=4, color=hexcolors_bright[counter], ls='-', lw=1.5, label=rf"${z_label}, \ {gal_label}$")
                plt.fill_between(rbinc, rat_mean+rat_err, rat_mean-rat_err, color=hexcolors_bright[counter], alpha=0.3)
            else:
                #plt.errorbar(rbinc, rat_mean, yerr=rat_err, capsize=4, color=hexcolors_bright[counter], ls='-')
                plt.errorbar(rbinc, rat_mean, capsize=4, color=hexcolors_bright[counter], ls='-', lw=1.5)
                plt.fill_between(rbinc, rat_mean+rat_err, rat_mean-rat_err, color=hexcolors_bright[counter], alpha=0.3)
            
            # centrals
            f = np.load(f'{gal_type:s}/{gal_type:s}_corr_bin_{nbins[i]:d}_cent{drad_str}_{n_gal}_{fp_dm}_{snapshot:d}.npz')
            rat_mean, rat_err, rbinc, logm = f['mean'], f['err'], f['binc'], f['logm']
            #plt.errorbar(rbinc, rat_mean, yerr=rat_err, capsize=4, color=hexcolors_bright[counter], ls='-.')
            plt.errorbar(rbinc, rat_mean, capsize=4, color=hexcolors_bright[counter], ls='-.')

            # satellites
            f = np.load(f'{gal_type:s}/{gal_type:s}_corr_bin_{nbins[i]:d}_sats{drad_str}_{n_gal}_{fp_dm}_{snapshot:d}.npz')
            rat_mean, rat_err, rbinc, logm = f['mean'], f['err'], f['binc'], f['logm']
            #plt.errorbar(rbinc*(1.+0.03), rat_mean, yerr=rat_err, capsize=4, color=hexcolors_bright[counter], ls='--')
            plt.errorbar(rbinc, rat_mean, capsize=4, color=hexcolors_bright[counter], ls='--')
            #plt.fill_between(rbinc, rat_mean+rat_err, rat_mean-rat_err, color=hexcolors_bright[counter], alpha=0.5)

            if counter == 0:
                plt.text(x=0.5, y=0.1, s=r"$\log (M) = %.1f$"%logm, transform=plt.gca().transAxes)

            plt.xscale('log')
            plt.ylim([0.5, 1.5])

        counter += 1

plt.subplot(nrow, ncol, 1)
plt.plot([], [], color='k', ls='-.', label='centrals')
plt.plot([], [], color='k', ls='--', label='satellites')
plt.legend(fontsize=22)

plt.subplot(nrow, ncol, ncol)
plt.legend(fontsize=22, frameon=False)

plt.subplot(nrow, ncol, ncol*nrow)
plt.legend(fontsize=22, frameon=False)

for i in range(ncol*nrow):
    plt.subplot(nrow, ncol, i+1)
    if i >= ncol:
        plt.xlabel(r'$r \ [{\rm Mpc}/h]$')
    if i in [0, ncol]:
        plt.ylabel(r'$\xi_{\rm pred}(r)/\xi_{\rm true}(r)$')
    else:
        plt.gca().set_yticklabels([])


plt.savefig(f"figs/corr_mass_{n_gal}.pdf", bbox_inches='tight', pad_inches=0.)
plt.show()
